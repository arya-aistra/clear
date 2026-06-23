"""Distillation losses: soft-label KL + boundary-weighted BCE + temporal smoothness.

    L_total = λ1·L_soft + λ2·L_boundary + λ3·L_smooth

L_soft     : temperature-T KL between the teacher's and student's Bernoulli speech
             distributions (scaled by T² as in Hinton KD).
L_boundary : BCE(student_logit, teacher_hard) upweighted within ±boundary_width frames of a
             teacher decision change — trains crisp, fast onsets/endpoints.
L_smooth   : mean |Δ| of consecutive student probabilities — suppresses steady-state jitter.

Inputs are per-chunk sequences: student_logits [B, K] (pre-sigmoid), teacher_probs [B, K].
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def bernoulli_kl(p: Tensor, q: Tensor, eps: float = 1e-6) -> Tensor:
    """KL(Bernoulli(p) || Bernoulli(q)), elementwise."""
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def transition_weight(teacher_hard: Tensor, width: int, weight: float) -> Tensor:
    """Per-frame weight = 1 + weight·(within ±width frames of a teacher decision change)."""
    B, K = teacher_hard.shape
    if K < 2:
        return torch.ones_like(teacher_hard)
    changes = (teacher_hard[:, 1:] != teacher_hard[:, :-1]).float()  # [B, K-1]
    change_frames = teacher_hard.new_zeros(B, K)
    change_frames[:, :-1] = torch.maximum(change_frames[:, :-1], changes)
    change_frames[:, 1:] = torch.maximum(change_frames[:, 1:], changes)
    if width > 0:
        dil = F.max_pool1d(change_frames.unsqueeze(1), kernel_size=2 * width + 1,
                           stride=1, padding=width).squeeze(1)
    else:
        dil = change_frames
    return 1.0 + weight * dil


class DFKDLoss(nn.Module):
    def __init__(
        self,
        lambda_soft: float = 1.0,
        lambda_boundary: float = 2.0,
        lambda_smooth: float = 0.5,
        temperature: float = 2.0,
        boundary_width: int = 3,
        boundary_weight: float = 5.0,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.l_soft = lambda_soft
        self.l_boundary = lambda_boundary
        self.l_smooth = lambda_smooth
        self.T = temperature
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.threshold = threshold
        self.eps = eps

    def forward(self, student_logits: Tensor, teacher_probs: Tensor
                ) -> Tuple[Tensor, Dict[str, float]]:
        teacher_probs = teacher_probs.clamp(self.eps, 1 - self.eps)
        teacher_logits = (teacher_probs / (1 - teacher_probs)).log()

        # --- L_soft: temperature-scaled Bernoulli KL (teacher || student) ---
        pt = torch.sigmoid(teacher_logits / self.T)
        ps = torch.sigmoid(student_logits / self.T)
        l_soft = bernoulli_kl(pt, ps, self.eps).mean() * (self.T ** 2)

        # --- L_boundary: transition-weighted BCE to hard teacher labels ---
        teacher_hard = (teacher_probs > self.threshold).float()
        if self.l_boundary > 0:
            w = transition_weight(teacher_hard, self.boundary_width, self.boundary_weight)
            bce = F.binary_cross_entropy_with_logits(student_logits, teacher_hard,
                                                     reduction="none")
            l_boundary = (bce * w).sum() / w.sum().clamp_min(1.0)
        else:
            l_boundary = student_logits.new_zeros(())

        # --- L_smooth: temporal L1 on student probabilities ---
        if self.l_smooth > 0 and student_logits.shape[1] > 1:
            sp = torch.sigmoid(student_logits)
            l_smooth = (sp[:, 1:] - sp[:, :-1]).abs().mean()
        else:
            l_smooth = student_logits.new_zeros(())

        total = self.l_soft * l_soft + self.l_boundary * l_boundary + self.l_smooth * l_smooth
        parts = {
            "total": float(total.detach()),
            "soft": float(l_soft.detach()),
            "boundary": float(l_boundary.detach()),
            "smooth": float(l_smooth.detach()),
        }
        return total, parts


@torch.no_grad()
def agreement_rate(student_probs: Tensor, teacher_probs: Tensor, threshold: float = 0.5
                   ) -> float:
    """Fraction of frames where student and teacher agree at the threshold."""
    s = (student_probs > threshold)
    t = (teacher_probs > threshold)
    return float((s == t).float().mean())
