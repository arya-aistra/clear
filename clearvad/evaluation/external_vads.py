"""Adapters for third-party VADs, each emitting per-32ms-chunk speech probabilities on ClearVAD's
eval grid so they can be scored identically.

Each adapter: `.load()` (raises if the package/model is unavailable) then `.probs(audio_1d)` →
np.ndarray[K] aligned to 512-sample (32 ms) chunks. The benchmark wraps `.load()` in try/except so
an uninstalled model is reported "unavailable" rather than crashing the whole run.

Install notes (only what you have needs to import):
  pyannote   : pip install pyannote.audio ; export HF_TOKEN=...  (accept the model's HF license)
  ten-vad    : pip install ten-vad         (needs libc++; may be blocked on some boxes)
  fireredvad : project-specific; wire the import in FireRedVAD.load() if you have it
  nemo       : pip install nemo_toolkit[asr]  (uses the existing NeMoMarbleTeacher)
"""

from __future__ import annotations

import numpy as np

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE


def align_to_chunks(src_probs, src_hop_samples: float, n_samples: int,
                    chunk: int = CHUNK_SAMPLES) -> np.ndarray:
    """Map probabilities sampled every `src_hop_samples` to K=n_samples//chunk chunk-probs
    (mean of the source frames overlapping each chunk)."""
    K = n_samples // chunk
    out = np.zeros(K, dtype=np.float32)
    src = np.asarray(src_probs, dtype=np.float32).reshape(-1)
    if src.size == 0 or src_hop_samples <= 0:
        return out
    for k in range(K):
        f0 = int((k * chunk) // src_hop_samples)
        f1 = max(f0 + 1, int(np.ceil(((k + 1) * chunk) / src_hop_samples)))
        seg = src[f0:min(f1, src.size)]
        out[k] = float(seg.mean()) if seg.size else float(src[min(f0, src.size - 1)])
    return out


class PyannoteVAD:
    name = "pyannote"
    params = None

    def __init__(self, model_id: str = "pyannote/segmentation-3.0") -> None:
        self.model_id = model_id

    def load(self) -> None:
        import os
        import torch
        from pyannote.audio import Inference, Model
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        model = Model.from_pretrained(self.model_id, use_auth_token=token)
        self.inf = Inference(model, batch_size=32)
        self.torch = torch

    def probs(self, audio: np.ndarray) -> np.ndarray:
        wav = self.torch.from_numpy(np.ascontiguousarray(audio, np.float32)).unsqueeze(0)
        out = self.inf({"waveform": wav, "sample_rate": SAMPLE_RATE})  # SlidingWindowFeature
        data = np.asarray(out.data, dtype=np.float32)
        if data.ndim == 3:                       # [chunks, frames, classes] -> collapse
            data = data.reshape(-1, data.shape[-1])
        if data.ndim == 2 and data.shape[1] > 1:
            # powerset output: softmax to probs if these look like logits, then speech = 1 - P(empty)
            if data.min() < 0.0 or data.max() > 1.0001:
                e = np.exp(data - data.max(axis=-1, keepdims=True))
                data = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
            speech = 1.0 - data[:, 0]            # class 0 = empty set (no speaker)
        else:
            speech = data.reshape(-1)
        # derive frame hop from COVERAGE (robust; do not trust sliding_window metadata)
        hop = len(audio) / max(len(speech), 1)
        return align_to_chunks(speech, hop, len(audio))


class TenVAD:
    name = "tenvad"
    params = None

    def __init__(self, hop: int = 256) -> None:
        self.hop = hop

    def load(self) -> None:
        from ten_vad import TenVad
        self.vad = TenVad(self.hop) if self._accepts_arg(TenVad) else TenVad()

    @staticmethod
    def _accepts_arg(cls) -> bool:
        import inspect
        try:
            return len(inspect.signature(cls.__init__).parameters) > 1
        except (ValueError, TypeError):
            return False

    def probs(self, audio: np.ndarray) -> np.ndarray:
        x = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        n = len(x) // self.hop
        p = np.empty(n, dtype=np.float32)
        for i in range(n):
            frame = x[i * self.hop:(i + 1) * self.hop]
            r = self.vad.process(frame)
            p[i] = float(r[0] if isinstance(r, (tuple, list)) else r)
        return align_to_chunks(p, self.hop, len(audio))


class FireRedVAD:
    name = "fireredvad"
    params = None

    def load(self) -> None:
        # Project-specific. Wire your import + per-frame inference here; left as a graceful stub.
        try:
            import fireredvad  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "FireRedVAD not wired: install/clone it and implement FireRedVAD.load()/probs() "
                "(map its frame output via align_to_chunks).") from exc
        raise ImportError("FireRedVAD import found but inference not implemented; wire probs().")

    def probs(self, audio: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class NeMoVAD:
    name = "nemo"
    params = None

    def load(self) -> None:
        import torch
        from clearvad.distill.nemo_teacher import NeMoMarbleTeacher
        self.teacher = NeMoMarbleTeacher()
        self.torch = torch

    def probs(self, audio: np.ndarray) -> np.ndarray:
        wav = self.torch.from_numpy(np.ascontiguousarray(audio, np.float32)).unsqueeze(0)
        out = self.teacher.label(wav)                       # [1, K] aligned to 512-chunks
        return np.asarray(out, dtype=np.float32).reshape(-1)


REGISTRY = {a.name: a for a in (PyannoteVAD, TenVAD, FireRedVAD, NeMoVAD)}
