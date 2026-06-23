"""Convolutional encoder — depthwise-separable (lighter than Silero's plain convs).

Follows Silero's measured channel schedule 129→128→64→64→128 (Phase 0 dissection) but
replaces each plain Conv1d with a depthwise-separable block (depthwise k=3 + pointwise
k=1 + BatchNorm + ReLU). 'same' padding (k//2) preserves the per-chunk frame count so the
encoder is applied independently per chunk (cross-chunk temporal modeling is the G-SSM's
job). A plain-conv variant is available for the Silero-parity ablation.

Input : [B, 129, T]   magnitude features
Output: [B, 128, T]   encoded features
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

Tensor = torch.Tensor


class DWSepConv1d(nn.Module):
    """Depthwise (k) + pointwise (1) + BN + ReLU, 'same' padding."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, kernel_size, padding=kernel_size // 2,
                            groups=c_in, bias=False)
        self.pw = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class PlainConv1d(nn.Module):
    """Plain Conv1d (k) + BN + ReLU, 'same' padding — Silero-parity ablation block."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvEncoder(nn.Module):
    def __init__(
        self,
        channel_schedule: List[int] = (129, 128, 64, 64, 128),
        depthwise_separable: bool = True,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.channel_schedule = list(channel_schedule)
        self.depthwise_separable = depthwise_separable
        Block = DWSepConv1d if depthwise_separable else PlainConv1d
        blocks = []
        for c_in, c_out in zip(self.channel_schedule[:-1], self.channel_schedule[1:]):
            blocks.append(Block(c_in, c_out, kernel_size))
        self.blocks = nn.ModuleList(blocks)

    @property
    def in_channels(self) -> int:
        return self.channel_schedule[0]

    @property
    def out_channels(self) -> int:
        return self.channel_schedule[-1]

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"encoder expects [B,C,T]; got {tuple(x.shape)}")
        for blk in self.blocks:
            x = blk(x)
        return x

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
