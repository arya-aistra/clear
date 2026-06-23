"""Silero VAD v5 compatibility shim — the *teacher* wrapper.

This module gives Silero VAD v5 a single, stable, well-documented interface that the
rest of ClearVAD (dissection, distillation teacher, benchmark baseline) consumes. It
deliberately makes **no assumptions** about Silero's internal tensor shapes — those are
discovered and reported by ``scripts/dissect_silero.py``. The wrapper only standardizes
the *behavioral* contract:

    forward(chunk[B, 512]) -> prob[B]          # speech probability in [0, 1]
    reset_states(batch_size)                   # clear recurrent state
    streaming_step(chunk) == forward(chunk)    # streaming alias
    get_state() -> dict                        # introspect carried state (h/c/context)
    probabilities(wav) -> np.ndarray           # per-chunk probs over a whole signal

torch is imported lazily so that merely importing this module in an inference-only
(ONNX, no-torch) environment does not fail. torch is only required to *instantiate*
``SileroVAD`` (i.e. teacher-side / training-side).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE

ArrayLike = Union["np.ndarray", "Any"]  # np.ndarray or torch.Tensor


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "SileroVAD requires PyTorch (teacher/training side). "
            "Install with: pip install -r requirements.txt"
        ) from exc


def load_silero(onnx: bool = False, force_reload: bool = False, **kwargs) -> "SileroVAD":
    """Convenience constructor. See :class:`SileroVAD`."""
    return SileroVAD(onnx=onnx, force_reload=force_reload, **kwargs)


class SileroVAD:
    """Behavioral wrapper around Silero VAD v5 (JIT or ONNX backend).

    Parameters
    ----------
    onnx:
        If True, load the ONNX backend via the ``silero-vad`` package's ``OnnxWrapper``
        (forces CPU). If False, load the TorchScript (JIT) backend.
    force_reload:
        Only used for the torch.hub fallback path.
    sample_rate:
        Operating sample rate. Silero v5 supports 8000 and 16000; ClearVAD fixes 16000.
    """

    def __init__(
        self,
        onnx: bool = False,
        force_reload: bool = False,
        sample_rate: int = SAMPLE_RATE,
        device: str = "cpu",
    ) -> None:
        self.torch = _require_torch()
        self.onnx = bool(onnx)
        self.sample_rate = int(sample_rate)
        self.device = device
        self.chunk_samples = CHUNK_SAMPLES if sample_rate == 16_000 else 256
        self._backend = None  # "pkg-jit" | "pkg-onnx" | "hub-jit"
        self._model = self._load(onnx=onnx, force_reload=force_reload)
        self.reset_states(batch_size=1)

    # ------------------------------------------------------------------ loading
    def _load(self, onnx: bool, force_reload: bool):
        """Load Silero v5. Prefer the pip ``silero-vad`` package; fall back to torch.hub."""
        # Path 1: official pip package (recommended, pinned reproducible version).
        try:
            from silero_vad import load_silero_vad  # type: ignore

            model = load_silero_vad(onnx=onnx)
            self._backend = "pkg-onnx" if onnx else "pkg-jit"
            return model
        except Exception as pkg_exc:  # noqa: BLE001 - we intentionally try the fallback
            warnings.warn(
                f"silero-vad package load failed ({pkg_exc!r}); falling back to torch.hub.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Path 2: torch.hub fallback.
        model, _utils = self.torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=force_reload,
            onnx=onnx,
        )
        self._backend = "hub-jit" if not onnx else "pkg-onnx"
        return model

    # ------------------------------------------------------------------ core API
    def _to_input_tensor(self, chunk: ArrayLike):
        """Coerce a chunk to a float32 torch tensor of shape [B, N]."""
        torch = self.torch
        if isinstance(chunk, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        elif torch.is_tensor(chunk):
            t = chunk.to(torch.float32)
        else:
            t = torch.as_tensor(np.asarray(chunk, dtype=np.float32))
        if t.ndim == 1:
            t = t.unsqueeze(0)  # [N] -> [1, N]
        if t.ndim != 2:
            raise ValueError(f"chunk must be 1D or 2D [B, N]; got shape {tuple(t.shape)}")
        return t.to(self.device)

    def forward(self, chunk: ArrayLike) -> "Any":
        """Run one streaming chunk through Silero. Returns prob tensor of shape [B].

        State is carried internally by the Silero model across consecutive calls,
        exactly as in production streaming. Call :meth:`reset_states` between utterances.
        """
        torch = self.torch
        x = self._to_input_tensor(chunk)
        with torch.no_grad():
            out = self._model(x, self.sample_rate)
        out = torch.as_tensor(out)
        # Silero returns [B, 1]; normalize to [B].
        return out.reshape(out.shape[0])

    # streaming alias (matches ClearVAD's own naming used in later phases)
    def streaming_step(self, chunk: ArrayLike) -> "Any":
        return self.forward(chunk)

    def __call__(self, chunk: ArrayLike) -> "Any":
        return self.forward(chunk)

    def reset_states(self, batch_size: int = 1) -> None:
        """Reset Silero's recurrent state (and left-context buffer).

        The v5 JIT backend's ``reset_states()`` takes no argument (it infers batch size
        from the next input) and raises ``RuntimeError`` on extra args; the ONNX wrapper
        accepts ``batch_size``. Try with-arg first, fall back to no-arg.
        """
        try:
            self._model.reset_states(batch_size)
        except (TypeError, RuntimeError):
            self._model.reset_states()
        self._last_batch_size = batch_size

    # ------------------------------------------------------------------ introspection
    def get_state(self) -> Dict[str, Any]:
        """Return a dict of Silero's carried internal state for inspection.

        Keys present depend on the backend/version; we probe defensively. Typical v5:
        ``state`` (the combined LSTM h/c, shape ~[2, B, 128]) and ``context``
        (the prepended left-context samples, shape ~[B, 64]).
        """
        state: Dict[str, Any] = {"backend": self._backend}
        m = self._model

        def _store(key, val):
            if val is None:
                return
            try:
                state[key] = val.detach().cpu().clone() if hasattr(val, "detach") else val
            except Exception:  # noqa: BLE001
                state[key] = val

        # 1. direct attributes (ONNX wrapper exposes these; some JIT versions too)
        for attr in ("_state", "_context", "_h", "_c", "_last_sr", "_last_batch_size"):
            _store(attr.lstrip("_"), getattr(m, attr, None))

        # 2. registered buffers (the v5 JIT often keeps recurrent state as buffers
        #    nested inside submodules, invisible to top-level getattr).
        named_buffers = getattr(m, "named_buffers", None)
        if callable(named_buffers):
            try:
                for name, buf in named_buffers():
                    low = name.lower()
                    if any(k in low for k in ("state", "context", "hidden", "cell", "_h", "_c")):
                        _store(f"buf:{name}", buf)
            except Exception:  # noqa: BLE001
                pass
        return state

    def describe(self) -> Dict[str, Any]:
        """Structured summary of the loaded teacher (for the dissection report)."""
        info: Dict[str, Any] = {
            "backend": self._backend,
            "onnx": self.onnx,
            "sample_rate": self.sample_rate,
            "chunk_samples": self.chunk_samples,
            "model_repr_type": type(self._model).__name__,
        }
        # ONNX backend: surface the underlying ORT session I/O if present.
        session = getattr(self._model, "session", None)
        if session is not None:
            try:
                info["onnx_inputs"] = [
                    {"name": i.name, "shape": list(i.shape), "type": i.type}
                    for i in session.get_inputs()
                ]
                info["onnx_outputs"] = [
                    {"name": o.name, "shape": list(o.shape), "type": o.type}
                    for o in session.get_outputs()
                ]
            except Exception:  # noqa: BLE001
                pass
        return info

    def parameter_count(self) -> Optional[int]:
        """Total parameter count for the JIT backend (None for ONNX)."""
        params = getattr(self._model, "parameters", None)
        if params is None:
            return None
        try:
            return int(sum(p.numel() for p in self._model.parameters()))
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ batch helpers
    def probabilities(self, wav: ArrayLike, reset: bool = True) -> np.ndarray:
        """Stream a full waveform and return one speech probability per 512-sample chunk.

        This is the canonical *streaming* path used by evaluation and failure-mode
        scripts: it resets state, then feeds non-overlapping `chunk_samples` windows.
        Trailing samples shorter than one chunk are dropped (Silero requires exact size).

        Returns
        -------
        np.ndarray, shape [num_chunks], dtype float32
        """
        torch = self.torch
        wav = self._as_mono_np(wav)
        if reset:
            self.reset_states(batch_size=1)
        n = self.chunk_samples
        num_chunks = len(wav) // n
        probs: List[float] = []
        with torch.no_grad():
            for i in range(num_chunks):
                seg = wav[i * n : (i + 1) * n]
                p = self.forward(seg)
                probs.append(float(p.reshape(-1)[0]))
        return np.asarray(probs, dtype=np.float32)

    @staticmethod
    def _as_mono_np(wav: ArrayLike) -> np.ndarray:
        if hasattr(wav, "detach"):  # torch tensor
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            # assume [channels, samples] or [samples, channels]; collapse to mono
            ch_axis = 0 if wav.shape[0] < wav.shape[1] else 1
            wav = wav.mean(axis=ch_axis)
        return wav.reshape(-1)
