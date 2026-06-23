"""Multi-teacher: weighted ensemble of Silero + Pyannote soft labels.

Drop-in replacement for SileroTeacher in the trainer: exposes ``.label(audio) -> [B,K]``,
``.build_student_windows(audio)`` (delegated to the Silero wrapper), and ``.torch``. The
ensemble target is sharper than Silero alone, so the student can exceed Silero on independent
labels (breaking the single-teacher distillation ceiling).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from clearvad.distill.teacher import SileroTeacher


def combine_probs(prob_arrays: List, weights: List[float]):
    """Weighted average of aligned per-chunk prob tensors (trimmed to common length)."""
    import torch

    arrs = [torch.as_tensor(p, dtype=torch.float32) for p in prob_arrays]
    K = min(a.shape[-1] for a in arrs)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = None
    for a, wi in zip(arrs, w):
        term = a[..., :K] * float(wi)
        out = term if out is None else out + term
    return out


class MultiTeacher:
    def __init__(self, silero_weight: float = 0.3, pyannote_weight: float = 0.7,
                 hf_token: Optional[str] = None, device: str = "cpu") -> None:
        self.silero = SileroTeacher(device=device)
        from clearvad.distill.pyannote_teacher import PyannoteTeacher
        self.pyannote = PyannoteTeacher(hf_token=hf_token, device=device)
        self.w_silero = float(silero_weight)
        self.w_pyannote = float(pyannote_weight)

    @property
    def torch(self):
        return self.silero.torch

    def build_student_windows(self, audio):
        return self.silero.build_student_windows(audio)

    def label(self, audio):
        ps = self.silero.label(audio)        # [B, K]
        pp = self.pyannote.label(audio)      # [B, K]
        return combine_probs([ps, pp], [self.w_silero, self.w_pyannote])
