"""Silero teacher — produces per-chunk soft labels for distillation.

Uses Silero's `audio_forward` (fast internal chunked pass) to label a whole batch of clips
in one call, returning per-chunk probabilities aligned to the student's chunks. Also builds
the student's 576-sample windows (64-sample left context + 512 chunk) from the same clips.

Alignment: teacher chunk i = clip[i*512:(i+1)*512] (context handled internally); student
window i = clip[i*512-64 : i*512+512] (zeros for the first chunk). Both start from a reset
state with zero context, so the two are aligned chunk-for-chunk.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clearvad import CHUNK_SAMPLES, CONTEXT_SAMPLES, SAMPLE_RATE
from clearvad.model.silero_compat import SileroVAD


class SileroTeacher:
    def __init__(self, device: str = "cpu") -> None:
        self.vad = SileroVAD(onnx=False, device=device)
        self.sr = SAMPLE_RATE

    @property
    def torch(self):
        return self.vad.torch

    def label(self, audio):
        """audio: [B, L] float (tensor or ndarray) -> teacher probs [B, K] (torch float32).

        K = L // 512. Uses Silero audio_forward for speed; trims to K chunks.
        """
        torch = self.torch
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        audio = audio.to(torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        K = L // CHUNK_SAMPLES
        self.vad.reset_states(B)
        with torch.no_grad():
            model = self.vad._model
            if hasattr(model, "audio_forward"):
                probs = torch.as_tensor(model.audio_forward(audio, self.sr)).reshape(B, -1)
            else:
                # fallback: manual per-chunk streaming
                cols = []
                for i in range(K):
                    seg = audio[:, i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
                    cols.append(self.vad.forward(seg).reshape(B, 1))
                probs = torch.cat(cols, dim=1)
        return probs[:, :K].contiguous().to(torch.float32)

    @staticmethod
    def probs_to_logits(probs, eps: float = 1e-6):
        """Recover pseudo-logits from teacher probabilities (Silero outputs post-sigmoid)."""
        p = probs.clamp(eps, 1.0 - eps)
        return (p / (1.0 - p)).log()

    def build_student_windows(self, audio):
        """audio: [B, L] -> student windows [B, K, 576] (64 ctx + 512 chunk), zeros ctx@i=0."""
        torch = self.torch
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        audio = audio.to(torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        K = L // CHUNK_SAMPLES
        win = CONTEXT_SAMPLES + CHUNK_SAMPLES  # 576
        # left-pad by CONTEXT_SAMPLES so chunk 0's context is zeros
        padded = torch.cat([audio.new_zeros(B, CONTEXT_SAMPLES), audio], dim=1)
        windows = audio.new_zeros(B, K, win)
        for i in range(K):
            start = i * CHUNK_SAMPLES  # in padded coords this maps to [start : start+576]
            windows[:, i] = padded[:, start:start + win]
        return windows
