"""LiquidVAD — same proven frontend/encoder/head as ClearVAD, but a CfC temporal core.

Identical streaming contract and forward_sequence/forward interface as ClearVADModel, so it drops
into the existing trainer, compare_models, and ONNX export. The ONLY difference is the temporal
core (CfC instead of G-SSM) — a controlled architecture ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from clearvad.model.cfc import CfCCore
from clearvad.model.encoder import ConvEncoder
from clearvad.model.frontend import STFTFrontend
from clearvad.model.head import VADHead

Tensor = torch.Tensor


@dataclass
class LiquidVADConfig:
    frontend_in_channels: int = 1
    frontend_out_channels: int = 258
    frontend_kernel_size: int = 256
    frontend_stride: int = 128
    frontend_learnable: bool = True
    encoder_channel_schedule: List[int] = field(default_factory=lambda: [129, 128, 64, 64, 128])
    encoder_depthwise_separable: bool = True
    encoder_kernel_size: int = 3
    # CfC core
    cfc_d_model: int = 128
    cfc_hidden: int = 128
    cfc_layers: int = 1
    cfc_backbone_units: Optional[int] = None
    cfc_dropout: float = 0.0
    head_reduction: str = "mean"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "LiquidVADConfig":
        fe = cfg.get("frontend", {})
        en = cfg.get("encoder", {})
        cf = cfg.get("cfc", {})
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
            cfc_d_model=cf.get("d_model", 128),
            cfc_hidden=cf.get("hidden", 128),
            cfc_layers=cf.get("layers", 1),
            cfc_backbone_units=cf.get("backbone_units", None),
            cfc_dropout=cf.get("dropout", 0.0),
            head_reduction=hd.get("reduction", "mean"),
        )


class LiquidVADModel(nn.Module):
    def __init__(self, config: Optional[LiquidVADConfig] = None) -> None:
        super().__init__()
        cfg = config or LiquidVADConfig()
        self.config = cfg
        self.frontend = STFTFrontend(
            in_channels=cfg.frontend_in_channels, out_channels=cfg.frontend_out_channels,
            kernel_size=cfg.frontend_kernel_size, stride=cfg.frontend_stride,
            learnable=cfg.frontend_learnable,
        )
        self.encoder = ConvEncoder(
            channel_schedule=cfg.encoder_channel_schedule,
            depthwise_separable=cfg.encoder_depthwise_separable,
            kernel_size=cfg.encoder_kernel_size,
        )
        self.core = CfCCore(d_model=cfg.cfc_d_model, hidden_size=cfg.cfc_hidden,
                            n_layers=cfg.cfc_layers, backbone_units=cfg.cfc_backbone_units,
                            dropout=cfg.cfc_dropout)
        self.head = VADHead(in_channels=cfg.cfc_d_model, reduction=cfg.head_reduction)
        if self.encoder.out_channels != cfg.cfc_d_model:
            raise ValueError(f"encoder out_channels ({self.encoder.out_channels}) != cfc d_model "
                             f"({cfg.cfc_d_model})")
        if self.frontend.n_mag != self.encoder.in_channels:
            raise ValueError(f"frontend n_mag ({self.frontend.n_mag}) != encoder in_channels "
                             f"({self.encoder.in_channels})")

    # ----- state -----
    def reset_state(self, batch_size: int = 1, device=None, dtype=None) -> Tensor:
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        return self.core.initial_state(batch_size, device=device, dtype=dtype)

    # ----- forward -----
    def features(self, chunk: Tensor) -> Tensor:
        mag = self.frontend(chunk)
        enc = self.encoder(mag)
        return enc.transpose(1, 2)              # [B, L, C]

    def forward(self, chunk: Tensor, state: Optional[Tensor] = None,
                return_logit: bool = False) -> Tuple[Tensor, Tensor]:
        feats = self.features(chunk)
        y, new_state = self.core(feats, state)
        out = self.head(y, return_logit=return_logit)
        return out, new_state

    def streaming_step(self, chunk: Tensor, state: Optional[Tensor] = None
                       ) -> Tuple[Tensor, Tensor]:
        return self.forward(chunk, state)

    def forward_sequence(self, windows: Tensor, return_logit: bool = True,
                         scan_mode: str = "loop") -> Tensor:
        if windows.dim() != 3:
            raise ValueError(f"forward_sequence expects [B,K,576]; got {tuple(windows.shape)}")
        B, K, S = windows.shape
        feats = self.features(windows.reshape(B * K, S))    # [B*K, T_enc, C]
        t_enc, C = feats.shape[1], feats.shape[2]
        feats = feats.reshape(B, K * t_enc, C)              # [B, K*T_enc, C]
        y, _ = self.core(feats, None)                       # [B, K*T_enc, C]
        y = y.reshape(B * K, t_enc, C)
        out = self.head(y, return_logit=return_logit)       # [B*K, 1]
        return out.reshape(B, K)

    # ----- introspection -----
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def count_by_module(self) -> Dict[str, int]:
        return {
            "frontend": self.frontend.parameter_count(),
            "encoder": self.encoder.parameter_count(),
            "cfc": self.core.parameter_count(),
            "head": self.head.parameter_count(),
            "total": self.parameter_count(),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LiquidVADModel":
        return cls(LiquidVADConfig.from_dict(cfg))
