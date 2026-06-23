"""STFT-style learned front-end (mirrors Silero v5's fixed-STFT magnitude front-end).

Silero's front-end (from the Phase 0 dissection) is a *fixed* `Conv1d(1→258, k=256,
stride=128)` whose 258 channels are 129 cos + 129 sin DFT-basis filters; a per-bin
magnitude `sqrt(re² + im²)` collapses them to 129 features. ClearVAD reproduces this
structure but makes the basis **learnable** (default), with an optional warm-start from
Silero's basis for faster distillation convergence (Phase 3).

Input : [B, T_samples]  (T_samples = 576 = 512 chunk + 64 left context)
Output: [B, 129, L]      magnitude features (L = (T_samples - k)//stride + 1)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor


class STFTFrontend(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 258,
        kernel_size: int = 256,
        stride: int = 128,
        learnable: bool = True,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("out_channels must be even (real+imag halves)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_freq = out_channels // 2 + 1  # 258 -> 130? see note below
        # Silero packs 258 = 129 real + 129 imag -> 129 magnitude features.
        self.n_mag = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, bias=False)
        if not learnable:
            self.conv.weight.requires_grad_(False)

    def forward(self, wav: Tensor) -> Tensor:
        """wav: [B, T] or [B, 1, T] -> magnitude [B, n_mag, L]."""
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)                  # [B, 1, T]
        elif wav.dim() != 3:
            raise ValueError(f"frontend expects [B,T] or [B,1,T]; got {tuple(wav.shape)}")
        spec = self.conv(wav)                       # [B, 2*n_mag, L]
        real, imag = spec[:, : self.n_mag], spec[:, self.n_mag :]
        mag = torch.sqrt(real ** 2 + imag ** 2 + self.eps)  # [B, n_mag, L]
        return mag

    def output_frames(self, t_samples: int) -> int:
        return (t_samples - self.kernel_size) // self.stride + 1

    @torch.no_grad()
    def load_basis(self, weight: Tensor) -> None:
        """Warm-start the conv basis from an external [out_channels,1,k] tensor."""
        if tuple(weight.shape) != tuple(self.conv.weight.shape):
            raise ValueError(f"basis shape {tuple(weight.shape)} != "
                             f"{tuple(self.conv.weight.shape)}")
        self.conv.weight.copy_(weight)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
