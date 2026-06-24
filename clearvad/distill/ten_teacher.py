"""TEN VAD as a teacher — clean frame-level alternative to Pyannote.

TEN VAD (TEN-framework) outranks Silero on precision AND speech→non-speech latency (our weak
dimension), is lightweight, and exposes a simple per-frame API: `TenVad(hop_size).process(int16)`
returns `(probability, flag)`. With hop_size=256 (16 ms), a 512-sample ClearVAD chunk is EXACTLY
2 TEN frames → a 2-frame average lands on our 32 ms grid with **zero alignment guesswork** (the
fragility that broke the Pyannote teacher).

Still label-free: the teacher is a model, not human annotation. Stateful per clip (fresh TenVad
per clip = reset, mirroring how Silero resets between utterances).

Install: pip install git+https://github.com/TEN-framework/ten-vad.git
"""

from __future__ import annotations

import numpy as np

from clearvad import CHUNK_SAMPLES, CONTEXT_SAMPLES


class TenVadTeacher:
    def __init__(self, hop_size: int = 256, agg: str = "mean") -> None:
        try:
            from ten_vad import TenVad
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TenVadTeacher needs ten-vad: "
                "pip install git+https://github.com/TEN-framework/ten-vad.git"
            ) from exc
        if CHUNK_SAMPLES % hop_size != 0:
            raise ValueError(f"hop_size {hop_size} must divide CHUNK_SAMPLES {CHUNK_SAMPLES}")
        self._TenVad = TenVad
        self.hop = hop_size
        self.frames_per_chunk = CHUNK_SAMPLES // hop_size   # 512/256 = 2
        if agg not in ("mean", "max"):
            raise ValueError("agg must be 'mean' or 'max'")
        self.agg = agg

    @property
    def torch(self):
        import torch
        return torch

    @staticmethod
    def _to_int16(x: np.ndarray) -> np.ndarray:
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)

    def label(self, audio):
        """audio: [B, L] (tensor/ndarray) -> speech probs [B, K] torch float32, K = L//512."""
        import torch

        if hasattr(audio, "detach"):
            a = audio.detach().cpu().numpy()
        else:
            a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        B, L = a.shape
        K = L // CHUNK_SAMPLES
        fpc = self.frames_per_chunk
        out = np.zeros((B, K), dtype=np.float32)
        for b in range(B):
            vad = self._TenVad(hop_size=self.hop)   # fresh state per clip
            xi = self._to_int16(a[b])
            n_frames = K * fpc
            fr = np.empty(n_frames, dtype=np.float32)
            for f in range(n_frames):
                prob, _flag = vad.process(xi[f * self.hop:(f + 1) * self.hop])
                fr[f] = float(prob)
            fr = fr.reshape(K, fpc)
            out[b] = fr.max(axis=1) if self.agg == "max" else fr.mean(axis=1)
        return torch.from_numpy(out)

    def build_student_windows(self, audio):
        """[B, L] -> student windows [B, K, 576] (64 ctx + 512), zeros ctx at chunk 0."""
        torch = self.torch
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        audio = audio.to(torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        K = L // CHUNK_SAMPLES
        win = CONTEXT_SAMPLES + CHUNK_SAMPLES
        padded = torch.cat([audio.new_zeros(B, CONTEXT_SAMPLES), audio], dim=1)
        windows = audio.new_zeros(B, K, win)
        for i in range(K):
            windows[:, i] = padded[:, i * CHUNK_SAMPLES:i * CHUNK_SAMPLES + win]
        return windows
