"""Pyannote segmentation as a stronger second teacher for multi-teacher distillation.

Pyannote's `segmentation-3.0` outranks Silero on several VAD benchmarks. We use it to produce
per-512-chunk speech probabilities aligned to ClearVAD's 32 ms grid, so a student distilled
from Silero+Pyannote can exceed Silero (its decision surface is no longer the ceiling).

Still label-free: the teacher is a model, not human annotation.

Gating: `pyannote/segmentation-3.0` is a gated HF model — accept its terms and pass an HF
token (env HF_TOKEN or --hf-token). Speech probability is derived from the powerset output as
`1 - P(silence-class)`, resampled to the 32 ms chunk grid.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE


class PyannoteTeacher:
    def __init__(self, hf_token: Optional[str] = None, device: str = "cpu",
                 model_name: str = "pyannote/segmentation-3.0") -> None:
        try:
            import torch  # noqa: F401
            from pyannote.audio import Model
            from pyannote.audio.core.inference import Inference
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyannoteTeacher needs pyannote.audio: pip install pyannote.audio"
            ) from exc

        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        try:
            self.model = Model.from_pretrained(model_name, use_auth_token=token)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load {model_name}. Accept its terms at "
                f"https://hf.co/{model_name} and pass --hf-token / set HF_TOKEN. ({exc!r})"
            ) from exc
        self.model.eval().to(device)
        self.device = device
        # frame-level scores over the whole clip
        self.inference = Inference(self.model, skip_aggregation=False)
        self.sr = SAMPLE_RATE

    @staticmethod
    def _powerset_to_speech(scores: np.ndarray) -> np.ndarray:
        """scores [F, C] powerset activations -> speech prob [F] = 1 - P(silence class 0)."""
        s = scores.astype(np.float64)
        # softmax over classes (robust whether inputs are logits or already-normalized)
        s = s - s.max(axis=-1, keepdims=True)
        p = np.exp(s)
        p = p / (p.sum(axis=-1, keepdims=True) + 1e-9)
        silence = p[:, 0]            # class 0 = no speaker active
        return np.clip(1.0 - silence, 0.0, 1.0).astype(np.float32)

    def _label_one(self, wav_1d, n_chunks: int):
        import torch

        wav = wav_1d.reshape(1, -1)
        out = self.inference({"waveform": torch.as_tensor(wav, dtype=torch.float32),
                              "sample_rate": self.sr})
        scores = np.asarray(out.data)        # [F, C]
        if scores.ndim == 1:
            scores = scores[:, None]
        speech = self._powerset_to_speech(scores) if scores.shape[1] > 1 else \
            np.clip(scores[:, 0], 0.0, 1.0).astype(np.float32)
        # resample frame scores -> n_chunks at 32 ms grid
        F = len(speech)
        if F == 0:
            return np.zeros(n_chunks, dtype=np.float32)
        src_t = np.linspace(0.0, 1.0, F, endpoint=False) + 0.5 / F
        tgt_t = (np.arange(n_chunks) + 0.5) / n_chunks
        return np.interp(tgt_t, src_t, speech).astype(np.float32)

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
        out = np.zeros((B, K), dtype=np.float32)
        a = audio.cpu().numpy()
        for b in range(B):
            out[b] = self._label_one(a[b], K)
        return torch.from_numpy(out)
