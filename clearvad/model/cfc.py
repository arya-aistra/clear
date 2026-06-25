"""Closed-form Continuous-time (CfC) temporal core — the novel VAD core.

CfC (Hasani, Lechner et al., "Closed-form continuous-time neural networks", Nature MI 2022) is a
continuous-time RNN whose ODE solution is approximated in CLOSED FORM, so there is no ODE solver:
each step interpolates between two input-dependent candidate states via a time-gated sigmoid. It is
tiny, streaming-native (carry a hidden vector), and ONNX-friendly (elementwise ops + matmuls, no
associative scan, no control flow). It is essentially untouched for VAD — which is the point.

Drop-in for the G-SSM core: `CfCCore(x[B,T,C], state) -> (y[B,T,C], new_state)`, so the rest of the
model (frontend, encoder, head) is unchanged and we get a clean SSM-vs-CfC ablation.

Streaming/state: hidden h per layer, stacked as state[B, n_layers, hidden]. Uniform frame spacing
=> elapsed time ts=1 per step (kept explicit so irregular spacing is a future option).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


class CfCCell(nn.Module):
    """One closed-form continuous-time cell.

    h_t = ff1 * (1 - gate) + ff2 * gate,  gate = sigmoid(t_a * ts + t_b)
    where ff1, ff2, t_a, t_b are produced from a shared backbone over [x_t, h_{t-1}].
    """

    def __init__(self, input_size: int, hidden_size: int, backbone_units: Optional[int] = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        bb = backbone_units or hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, bb),
            nn.SiLU(),
        )
        self.ff1 = nn.Linear(bb, hidden_size)      # candidate state A
        self.ff2 = nn.Linear(bb, hidden_size)      # candidate state B
        self.time_a = nn.Linear(bb, hidden_size)   # time gate slope
        self.time_b = nn.Linear(bb, hidden_size)   # time gate bias

    def forward(self, x: Tensor, h: Tensor, ts: float = 1.0) -> Tensor:
        z = torch.cat([x, h], dim=-1)
        bb = self.backbone(z)
        ff1 = torch.tanh(self.ff1(bb))
        ff2 = torch.tanh(self.ff2(bb))
        gate = torch.sigmoid(self.time_a(bb) * ts + self.time_b(bb))
        return ff1 * (1.0 - gate) + ff2 * gate


class CfCCore(nn.Module):
    """Stacked CfC cells over a feature sequence. Mirrors the GSSM core's interface."""

    def __init__(self, d_model: int, hidden_size: int = 128, n_layers: int = 1,
                 backbone_units: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cells = nn.ModuleList([
            CfCCell(d_model if i == 0 else hidden_size, hidden_size, backbone_units)
            for i in range(n_layers)
        ])
        # project last layer's hidden back to d_model so the head (expects d_model) is unchanged
        self.out_proj = nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def state_shape(self) -> Tuple[int, int]:
        return (self.n_layers, self.hidden_size)

    def initial_state(self, batch_size: int, device=None, dtype=None) -> Tensor:
        return torch.zeros(batch_size, self.n_layers, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, state: Optional[Tensor] = None,
                scan_mode: str = "loop") -> Tuple[Tensor, Tensor]:
        """x: [B, T, d_model]. state: [B, n_layers, hidden] or None. Returns ([B,T,d_model], state).

        scan_mode is accepted for interface parity with the G-SSM but ignored (CfC is recurrent;
        the per-step recurrence is identical in train and export, so no parallel/loop duality needed)."""
        B, T, _ = x.shape
        if state is None:
            state = self.initial_state(B, device=x.device, dtype=x.dtype)
        h = [state[:, i] for i in range(self.n_layers)]
        outputs = []
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                h[i] = cell(inp, h[i])
                inp = self.dropout(h[i])
            outputs.append(inp)
        y = torch.stack(outputs, dim=1)            # [B, T, hidden]
        y = self.out_proj(y)                       # [B, T, d_model]
        new_state = torch.stack(h, dim=1)          # [B, n_layers, hidden]
        return y, new_state

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
