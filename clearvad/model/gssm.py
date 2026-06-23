"""Gated SSM (G-SSM) block — ClearVAD's single novel architectural contribution.

This block replaces Silero's LSTMCell temporal core. It is a *selective* state-space
model (Mamba-style) wrapped in a multiplicative gate, implemented so that:

  * it is **causal** (the only cross-time dependency is the carried SSM state h);
  * it carries **explicit state** across streaming chunks via ``forward(x, state)``;
  * its recurrence is an **explicit Python loop over T** (no associative scan), so it
    exports to ONNX opset-17 by unrolling at a fixed T — no Scan/Loop op needed;
  * ``offline(T)`` == ``T × online(T=1)`` to within fp tolerance (state carry is exact).

Block dataflow (matches the GSD Phase 1 spec):

    x:[B,T,D]
      ├─ in_proj ─→ (u:[B,T,E], z:[B,T,E])           # expand D→2E in one matmul
      │                 │            │
      │              SSM(u) = y      g = SiLU(z)      # selective SSM + gate branch
      │                 └──── y * g ──┘
      │                       │
      │                   out_proj ─→ [B,T,D]
      └──────────── + ───────────────┘ residual

Selective SSM core (per channel of E=d_inner, with N=d_state):
    Δ = softplus(dt_proj(low_rank(u)))            # input-dependent timescale [B,T,E]
    B,C = low_rank(u)                             # input-dependent [B,T,N]
    A   = -exp(A_log)                             # diagonal, negative ⇒ stable [E,N]
    Ā_t = exp(Δ_t ⊙ A) ;  recurrence  h_t = Ā_t ⊙ h_{t-1} + (Δ_t·u_t) ⊗ B_t
    y_t = (h_t ⊙ C_t).sum(N) + D ⊙ u_t

Why this targets Silero's documented issues:
  * Input-dependent Δ lets the effective memory window stretch/shrink per frame — useful
    for crisp behavior across short silences and long pauses.
  * Factored (A,B,C) state (vs entangled LSTM gate products) is a cleaner structured-pruning
    and INT8 target later (the recurrence stays FP32; only the dense projections quantize).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class GSSM(nn.Module):
    """Causal Gated Selective SSM block.

    Parameters
    ----------
    d_model : residual/interface width (matches the encoder output; default 128).
    d_inner : expanded SSM width E (default 256).
    d_state : SSM state size N per channel (default 16).
    dt_rank : low-rank dim for the input-dependent Δ projection (default 8).
    u_act   : optional activation applied to the SSM input u ('silu' or None). Default None
              keeps the block faithful to the spec (nonlinearity comes from the gate).
    tie_projections : if True, ties out_proj to the u-half of in_proj (transposed). This
              cuts ~d_inner·d_model params and brings the block under 100K at d_inner=256.
              Default False (faithful, ~115K, still < the LSTM it replaces).
    bias : bias on the big in/out projections (default False; lean).
    """

    def __init__(
        self,
        d_model: int = 128,
        d_inner: int = 256,
        d_state: int = 16,
        dt_rank: int = 8,
        u_act: Optional[str] = None,
        tie_projections: bool = False,
        bias: bool = False,
        scan_mode: str = "loop",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.u_act = u_act
        self.tie_projections = tie_projections
        # 'loop' = explicit unrolled recurrence (ONNX-exportable, used for export/streaming).
        # 'parallel' = log-depth associative scan (training speed; mathematically identical).
        self.scan_mode = scan_mode

        # Expand D -> 2E (u and gate z) in a single matmul.
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=bias)

        # Selective parameters: produce (Δ_lowrank, B, C) from u.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Output projection E -> D (optionally tied to in_proj's u-half).
        if tie_projections:
            self.out_proj = None  # uses in_proj.weight[:d_inner] transposed at runtime
            self.out_bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        else:
            self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
            self.out_bias = None

        # Diagonal A (negative, stable) stored as log; D skip connection.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)  # [E,N]
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self._init_dt_bias(dt_min, dt_max, dt_init_floor)
        self._init_dt_weight()

    # ---------------------------------------------------------------- init
    def _init_dt_bias(self, dt_min: float, dt_max: float, floor: float) -> None:
        """Mamba-style Δ bias init so softplus(bias) lands in [dt_min, dt_max]."""
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=floor)
        # inverse softplus: log(exp(dt) - 1) = dt + log(-expm1(-dt))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _init_dt_weight(self) -> None:
        # constant-ish scaling, keeps Δ stable early in training
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)

    # ---------------------------------------------------------------- helpers
    def initial_state(self, batch_size: int, device=None, dtype=None) -> Tensor:
        """Zero SSM state h of shape [B, d_inner, d_state]."""
        return torch.zeros(batch_size, self.d_inner, self.d_state,
                           device=device, dtype=dtype)

    def _apply_out_proj(self, y: Tensor) -> Tensor:
        if self.out_proj is not None:
            return self.out_proj(y)
        # tied: out = y @ W_u  where in_proj.weight[:E] is W_u of shape [E, D]
        w_u = self.in_proj.weight[: self.d_inner]            # [E, D]
        out = F.linear(y, w_u.t())                           # [B,T,D]
        if self.out_bias is not None:
            out = out + self.out_bias
        return out

    # ---------------------------------------------------------------- scan
    @staticmethod
    def _shift(x: Tensor, offset: int, pad_value: float) -> Tensor:
        """Shift right along time (dim=1) by offset, filling the front with pad_value."""
        B, T = x.shape[0], x.shape[1]
        pad = x.new_full((B, offset) + tuple(x.shape[2:]), pad_value)
        return torch.cat([pad, x[:, : T - offset]], dim=1)

    def _fold_state(self, dA: Tensor, dBu: Tensor, h0: Tensor) -> Tensor:
        """Fold the initial state into the first step: b0 ← a0·h0 + b0 (out-of-place)."""
        b0 = dBu[:, :1] + dA[:, :1] * h0.unsqueeze(1)
        return torch.cat([b0, dBu[:, 1:]], dim=1)

    def _scan_loop(self, a: Tensor, b: Tensor) -> Tensor:
        """Explicit recurrence H_t = a_t·H_{t-1} + b_t (h_{-1}=0). Unrolls for ONNX."""
        T = a.shape[1]
        h = torch.zeros_like(b[:, 0])
        hs = []
        for t in range(T):
            h = a[:, t] * h + b[:, t]
            hs.append(h)
        return torch.stack(hs, dim=1)

    def _scan_parallel(self, a: Tensor, b: Tensor) -> Tensor:
        """Hillis-Steele associative scan of the affine recurrence (O(log T) depth).

        Composition of affine maps f_t(h)=a_t·h+b_t: combine prev→cur as
        a←a_cur·a_prev, b←a_cur·b_prev+b_cur. Inclusive scan yields H_t (== _scan_loop)."""
        T = a.shape[1]
        a_s, b_s = a, b
        offset = 1
        while offset < T:
            a_prev = self._shift(a_s, offset, 1.0)   # identity a = 1
            b_prev = self._shift(b_s, offset, 0.0)   # identity b = 0
            b_s = a_s * b_prev + b_s
            a_s = a_s * a_prev
            offset *= 2
        return b_s

    def _scan(self, dA: Tensor, dBu: Tensor, h0: Tensor, mode: str) -> Tensor:
        b = self._fold_state(dA, dBu, h0)
        if mode == "parallel":
            return self._scan_parallel(dA, b)
        return self._scan_loop(dA, b)

    # ---------------------------------------------------------------- forward
    def forward(self, x: Tensor, state: Optional[Tensor] = None,
                scan_mode: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        """Run the block over a [B, T, D] sequence.

        x : [B, T, d_model]; state: carried SSM state [B, d_inner, d_state] or None.
        scan_mode: 'loop' (ONNX/streaming) or 'parallel' (fast training); defaults to
        self.scan_mode. Both compute the identical recurrence.

        Returns (out[B,T,D], new_state[B,d_inner,d_state]).
        """
        if x.dim() != 3:
            raise ValueError(f"GSSM expects [B, T, D]; got {tuple(x.shape)}")
        B, T, _ = x.shape

        uz = self.in_proj(x)                                  # [B,T,2E]
        u, z = uz.split(self.d_inner, dim=-1)                 # each [B,T,E]
        if self.u_act == "silu":
            u = F.silu(u)
        gate = F.silu(z)                                      # [B,T,E]

        # Selective params from u.
        x_dbl = self.x_proj(u)                                # [B,T,dt_rank+2N]
        dt, Bm, Cm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))                  # [B,T,E]
        A = -torch.exp(self.A_log)                            # [E,N]

        dA = torch.exp(delta.unsqueeze(-1) * A)               # exp(Δ⊙A)  [B,T,E,N]
        dBu = (delta * u).unsqueeze(-1) * Bm.unsqueeze(2)     # (Δ·u)⊗B   [B,T,E,N]

        h0 = (self.initial_state(B, x.device, x.dtype) if state is None
              else state.to(dtype=x.dtype))
        H = self._scan(dA, dBu, h0, scan_mode or self.scan_mode)   # [B,T,E,N]
        y = (H * Cm.unsqueeze(2)).sum(dim=-1)                 # [B,T,E]
        y = y + self.D * u                                    # skip connection

        out = y * gate                                        # gated fusion
        out = self._apply_out_proj(out)                       # [B,T,D]
        out = out + x                                         # residual
        return out, H[:, -1]

    # convenience alias used by the streaming layer in later phases
    def step(self, x_t: Tensor, state: Optional[Tensor] = None
             ) -> Tuple[Tensor, Tensor]:
        """Single-frame streaming step. x_t: [B, d_model] or [B, 1, d_model]."""
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        out, new_state = self.forward(x_t, state)
        return out[:, 0], new_state

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, d_inner={self.d_inner}, "
                f"d_state={self.d_state}, dt_rank={self.dt_rank}, "
                f"tie_projections={self.tie_projections}, params={self.parameter_count()}")
