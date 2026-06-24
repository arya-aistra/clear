"""Multi-teacher: weighted ensemble of any frame-level teachers' soft labels.

Drop-in replacement for SileroTeacher in the trainer: exposes ``.label(audio) -> [B,K]``,
``.build_student_windows(audio)`` (delegated to the primary teacher), and ``.torch``. The
ensemble target can be sharper than Silero alone, so the student may exceed Silero on
independent labels.

Construct directly with a teacher list, or via the convenience class methods.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

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
    def __init__(self, teachers: Sequence, weights: Sequence[float]) -> None:
        if len(teachers) != len(weights):
            raise ValueError("teachers and weights must be the same length")
        self.teachers = list(teachers)
        self.weights = list(weights)
        self._primary = self.teachers[0]   # provides windows + torch

    @property
    def torch(self):
        return self._primary.torch

    def build_student_windows(self, audio):
        return self._primary.build_student_windows(audio)

    def label(self, audio):
        return combine_probs([t.label(audio) for t in self.teachers], self.weights)

    # ------------------------------------------------------------ convenience
    @classmethod
    def silero_nemo(cls, silero_weight: float = 0.5, nemo_weight: float = 0.5,
                    device: str = "cpu") -> "MultiTeacher":
        from clearvad.distill.nemo_teacher import NeMoMarbleTeacher
        return cls([SileroTeacher(device=device), NeMoMarbleTeacher()],
                   [silero_weight, nemo_weight])

    @classmethod
    def silero_firered(cls, silero_weight: float = 0.5, firered_weight: float = 0.5,
                       device: str = "cpu") -> "MultiTeacher":
        from clearvad.distill.firered_teacher import FireRedVADTeacher
        return cls([SileroTeacher(device=device), FireRedVADTeacher(device=device)],
                   [silero_weight, firered_weight])
