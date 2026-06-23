"""Full ClearVAD model: frontend + encoder + G-SSM + head.

Streaming contract (ONNX-friendly, tensor state):
    forward(chunk[B, 576], state[B, d_inner, d_state] | None)
        -> (prob[B, 1], new_state[B, d_inner, d_state])

Only the G-SSM carries cross-chunk state; the frontend+encoder run per-chunk and rely on
the 64-sample left context baked into the 576-sample input (exactly like Silero v5). This
is what makes chunked streaming == offline exact (see tests/test_full_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from clearvad.model.encoder import ConvEncoder
from clearvad.model.frontend import STFTFrontend
from clearvad.model.gssm import GSSM
from clearvad.model.head import VADHead

Tensor = torch.Tensor


@dataclass
class ClearVADConfig:
    # frontend
    frontend_in_channels: int = 1
    frontend_out_channels: int = 258
    frontend_kernel_size: int = 256
    frontend_stride: int = 128
    frontend_learnable: bool = True
    # encoder
    encoder_channel_schedule: List[int] = field(default_factory=lambda: [129, 128, 64, 64, 128])
    encoder_depthwise_separable: bool = True
    encoder_kernel_size: int = 3
    # gssm
    gssm_d_model: int = 128
    gssm_d_inner: int = 256
    gssm_d_state: int = 16
    gssm_dt_rank: int = 8
    gssm_tie_projections: bool = False
    # head
    head_reduction: str = "mean"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ClearVADConfig":
        """Build from a nested config dict (as loaded from configs/model/*.yaml)."""
        fe = cfg.get("frontend", {})
        en = cfg.get("encoder", {})
        gs = cfg.get("gssm", {})
        hd = cfg.get("head", {})
        return cls(
            frontend_in_channels=fe.get("in_channels", 1),
            frontend_out_channels=fe.get("out_channels", 258),
            frontend_kernel_size=fe.get("kernel_size", 256),
            frontend_stride=fe.get("stride", 128),
            frontend_learnable=fe.get("learnable", True),
            encoder_channel_schedule=list(en.get("channel_schedule", [129, 128, 64, 64, 128])),
            encoder_depthwise_separable=en.get("depthwise_separable", True),
            encoder_kernel_size=en.get("kernel_size", 3),
            gssm_d_model=gs.get("d_model", 128),
            gssm_d_inner=gs.get("d_inner", 256),
            gssm_d_state=gs.get("d_state", 16),
            gssm_dt_rank=gs.get("dt_rank", 8),
            gssm_tie_projections=gs.get("tie_projections", False),
            head_reduction=hd.get("reduction", "mean"),
        )


class ClearVADModel(nn.Module):
    def __init__(self, config: Optional[ClearVADConfig] = None) -> None:
        super().__init__()
        cfg = config or ClearVADConfig()
        self.config = cfg

        self.frontend = STFTFrontend(
            in_channels=cfg.frontend_in_channels,
            out_channels=cfg.frontend_out_channels,
            kernel_size=cfg.frontend_kernel_size,
            stride=cfg.frontend_stride,
            learnable=cfg.frontend_learnable,
        )
        self.encoder = ConvEncoder(
            channel_schedule=cfg.encoder_channel_schedule,
            depthwise_separable=cfg.encoder_depthwise_separable,
            kernel_size=cfg.encoder_kernel_size,
        )
        self.gssm = GSSM(
            d_model=cfg.gssm_d_model,
            d_inner=cfg.gssm_d_inner,
            d_state=cfg.gssm_d_state,
            dt_rank=cfg.gssm_dt_rank,
            tie_projections=cfg.gssm_tie_projections,
        )
        self.head = VADHead(in_channels=cfg.gssm_d_model, reduction=cfg.head_reduction)

        self._validate_wiring()

    # ------------------------------------------------------------ wiring checks
    def _validate_wiring(self) -> None:
        if self.frontend.n_mag != self.encoder.in_channels:
            raise ValueError(
                f"frontend magnitude features ({self.frontend.n_mag}) != encoder "
                f"in_channels ({self.encoder.in_channels})"
            )
        if self.encoder.out_channels != self.gssm.d_model:
            raise ValueError(
                f"encoder out_channels ({self.encoder.out_channels}) != gssm d_model "
                f"({self.gssm.d_model})"
            )

    # ------------------------------------------------------------ state
    @property
    def d_inner(self) -> int:
        return self.gssm.d_inner

    @property
    def d_state(self) -> int:
        return self.gssm.d_state

    def reset_state(self, batch_size: int = 1, device=None, dtype=None) -> Tensor:
        """Return a fresh zero G-SSM state [B, d_inner, d_state]."""
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        return self.gssm.initial_state(batch_size, device=device, dtype=dtype)

    # ------------------------------------------------------------ forward
    def features(self, chunk: Tensor) -> Tensor:
        """[B, 576] -> encoder features [B, T_enc, C] (frontend + encoder, per-chunk)."""
        mag = self.frontend(chunk)          # [B, 129, L]
        enc = self.encoder(mag)             # [B, 128, L]
        return enc.transpose(1, 2)          # [B, L, 128]

    def forward(self, chunk: Tensor, state: Optional[Tensor] = None,
                return_logit: bool = False) -> Tuple[Tensor, Tensor]:
        feats = self.features(chunk)                 # [B, T_enc, C]
        y, new_state = self.gssm(feats, state)       # [B, T_enc, C], [B, E, N]
        out = self.head(y, return_logit=return_logit)  # [B, 1]
        return out, new_state

    def streaming_step(self, chunk: Tensor, state: Optional[Tensor] = None
                       ) -> Tuple[Tensor, Tensor]:
        return self.forward(chunk, state)

    def forward_sequence(self, windows: Tensor, return_logit: bool = True) -> Tensor:
        """Efficient training forward over a sequence of chunks.

        windows: [B, K, 576] — K consecutive 576-sample student windows (each = 64-sample
        left context + 512 chunk). Computes per-chunk encoder features, runs the G-SSM ONCE
        over the concatenated feature sequence (state flows across chunks, == streaming by
        the equivalence property), then the head per chunk.

        Returns [B, K] — per-chunk logit (or prob if return_logit=False).
        """
        if windows.dim() != 3:
            raise ValueError(f"forward_sequence expects [B,K,576]; got {tuple(windows.shape)}")
        B, K, S = windows.shape
        feats = self.features(windows.reshape(B * K, S))   # [B*K, T_enc, C]
        t_enc, C = feats.shape[1], feats.shape[2]
        feats = feats.reshape(B, K * t_enc, C)             # [B, K*T_enc, C]
        y, _ = self.gssm(feats, None)                      # [B, K*T_enc, C]
        y = y.reshape(B * K, t_enc, C)
        out = self.head(y, return_logit=return_logit)      # [B*K, 1]
        return out.reshape(B, K)

    # ------------------------------------------------------------ introspection
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def count_by_module(self) -> Dict[str, int]:
        return {
            "frontend": self.frontend.parameter_count(),
            "encoder": self.encoder.parameter_count(),
            "gssm": self.gssm.parameter_count(),
            "head": self.head.parameter_count(),
            "total": self.parameter_count(),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ClearVADModel":
        return cls(ClearVADConfig.from_dict(cfg))
