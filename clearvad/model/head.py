"""Projection + sigmoid head producing one speech probability per chunk.

Takes the G-SSM output [B, T_enc, C] for a chunk, projects to a single logit per frame
(`Conv1d(C→1, k=1)`), applies sigmoid, then reduces the chunk's frames to one probability.
Default reduction is 'mean' (matches Silero's final ReduceMean); 'last' is available for a
more strictly causal/low-latency operating point.

Output: [B, 1] speech probability in (0, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


class VADHead(nn.Module):
    def __init__(self, in_channels: int = 128, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "last"):
            raise ValueError("reduction must be 'mean' or 'last'")
        self.reduction = reduction
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, feats: Tensor, return_logit: bool = False) -> Tensor:
        """feats: [B, T, C] -> prob [B, 1] (or logit [B, 1] if return_logit).

        Reduces the chunk's per-frame logits to one logit (mean/last) BEFORE sigmoid, so a
        clean pre-sigmoid logit is available for temperature-scaled distillation (Phase 3).
        """
        if feats.dim() != 3:
            raise ValueError(f"head expects [B,T,C]; got {tuple(feats.shape)}")
        x = feats.transpose(1, 2)            # [B, C, T]
        logits = self.conv(x).squeeze(1)     # [B, T]
        if self.reduction == "mean":
            logit = logits.mean(dim=1, keepdim=True)   # [B, 1]
        else:
            logit = logits[:, -1:]                     # [B, 1]
        if return_logit:
            return logit
        return torch.sigmoid(logit)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
