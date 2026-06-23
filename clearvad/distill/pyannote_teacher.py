"""Pyannote segmentation as a stronger second teacher for multi-teacher distillation.

Pyannote's `segmentation-3.0` outranks Silero on several VAD benchmarks. We use it to produce
per-512-chunk speech probabilities aligned to ClearVAD's 32 ms grid, so a student distilled
from Silero+Pyannote can exceed Silero (its decision surface is no longer the ceiling).

Still label-free: the teacher is a model, not human annotation.

Implementation: call the model's forward DIRECTLY on a fixed-duration (model.specifications)
zero-padded waveform — avoids pyannote's Inference aggregation (which returns 3-D window stacks)
and the torchcodec file-decode path (we pass in-memory tensors). Speech prob = 1 - P(silence)
over the powerset classes, resampled to the 32 ms chunk grid. Runs on GPU when available.

Gating: `pyannote/segmentation-3.0` is gated — accept its terms and pass an HF token
(env HF_TOKEN or --hf-token).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE


class PyannoteTeacher:
    def __init__(self, hf_token: Optional[str] = None, device: Optional[str] = None,
                 model_name: str = "pyannote/segmentation-3.0", sub_batch: int = 32) -> None:
        try:
            import torch
            from pyannote.audio import Model
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyannoteTeacher needs pyannote.audio + torch") from exc

        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        try:
            self.model = Model.from_pretrained(model_name, use_auth_token=token)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load {model_name}. Accept terms at https://hf.co/{model_name} "
                f"and pass --hf-token / set HF_TOKEN. ({exc!r})"
            ) from exc
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(self.device)
        self.sr = SAMPLE_RATE
        self.sub_batch = sub_batch
        # model's training chunk duration (seconds) -> fixed input length in samples
        spec = self.model.specifications
        spec = spec[0] if isinstance(spec, (list, tuple)) else spec
        self.duration = float(getattr(spec, "duration", 10.0))
        self.n_samples = int(round(self.duration * self.sr))

    @staticmethod
    def _resample(vec: np.ndarray, k: int) -> np.ndarray:
        f = len(vec)
        if f == 0:
            return np.zeros(k, dtype=np.float32)
        src = (np.arange(f) + 0.5) / f
        tgt = (np.arange(k) + 0.5) / k
        return np.interp(tgt, src, vec).astype(np.float32)

    def label(self, audio):
        """audio: [B, L] (tensor/ndarray) -> speech probs [B, K] torch float32, K = L//512."""
        import torch

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        audio = audio.to(torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        K = L // CHUNK_SAMPLES
        N = self.n_samples
        # pad/truncate every clip to the model's fixed duration
        if L >= N:
            fixed = audio[:, :N]
            real_frac = 1.0
        else:
            fixed = torch.cat([audio, audio.new_zeros(B, N - L)], dim=1)
            real_frac = L / N

        out = np.zeros((B, K), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, B, self.sub_batch):
                wav = fixed[i:i + self.sub_batch].unsqueeze(1).to(self.device)  # [b,1,N]
                scores = self.model(wav)                                        # [b,F,C]
                scores = torch.as_tensor(scores).float()
                if scores.dim() == 2:
                    scores = scores.unsqueeze(0)
                p = torch.softmax(scores, dim=-1)
                speech = (1.0 - p[..., 0]).cpu().numpy()                        # [b,F]
                F = speech.shape[1]
                real_F = max(1, int(round(F * real_frac)))
                for j in range(speech.shape[0]):
                    out[i + j] = self._resample(speech[j, :real_F], K)
        return torch.from_numpy(out)
