# Phase 1 — Implementation Notes (G-SSM block)

## Goal
Implement the causal Gated SSM block that replaces Silero's LSTMCell, with explicit
cross-chunk state carry and ONNX-exportable recurrent inference.

- **Changing:** temporal core LSTM → selective G-SSM. **Fixed:** d_model=128 interface,
  causal/streaming contract, opset-17 export. **Improve:** params (< LSTM 131,584), dynamic
  memory via input-dependent Δ. **Cannot regress:** streaming==offline equivalence (<1e-5), export.

## Design (`clearvad/model/gssm.py`)
- `in_proj`: D→2E single matmul → (u, z). Gate g = SiLU(z). Faithful: no SiLU on u (gate
  supplies nonlinearity); optional `u_act='silu'`.
- Selective SSM: `x_proj`(u)→(Δ_lowrank, B, C); `dt_proj`→Δ=softplus(...) [B,T,E];
  A=−exp(A_log) diagonal-negative (stable); ZOH-style discretization
  `Ā=exp(Δ⊙A)`, `dBu=(Δ·u)⊗B`; explicit T-loop recurrence; `y=ΣC⊙h + D⊙u`.
- Gate fusion `y*g` → `out_proj`(E→D) → residual `+x`.
- `forward(x[B,T,D], state[B,E,N]|None) → (out[B,T,D], new_state[B,E,N])`; `step()` for T=1.
- Mamba-style Δ-bias init (softplus(bias)∈[1e-3,1e-1]); A_log=log(1..N) per channel; D=ones.

## ONNX strategy
The recurrence is an explicit Python `for t in range(T)` loop. At a **fixed T** (export-time
constant) torch.onnx.export **unrolls** it into static ops (Mul/Exp/Add/ReduceSum/Softplus),
so **no Scan/Loop op is needed** and opset-17 suffices. State is an explicit graph input/output,
so streaming carry works in ONNX. Exported and ORT-validated at T=1 (streaming) and T=16 (chunk).

## Parameter budget — honest note
The spec's "< 100K params at d_inner=256" **soft target is arithmetically infeasible** with
faithful dense projections:

| component (bias-free) | params |
|-----------------------|--------|
| in_proj 128→512 | 65,536 |
| out_proj 256→128 | 32,768 |
| x_proj 256→40 | 10,240 |
| dt_proj 8→256 (+bias) | 2,304 |
| A_log 256×16 | 4,096 |
| D 256 | 256 |
| **faithful total** | **115,200** |

in+out projections alone are 98,304 — the whole 100K budget. So:
- **Faithful default = 115,200 params < LSTM's 131,584 ⇒ HARD gate PASS.**
- `tie_projections=True` ties out_proj to in_proj's u-half (−32,768) → **82,432 < 100K**, the
  SOFT target. Default OFF (faithful) to preserve expressiveness for distillation; available as
  an ablation / lite knob. We'll A/B faithful-vs-tied accuracy during DFKD (Phase 3) and at
  ablation time (`reports/ablation_gssm_vs_lstm.md`).

## Tests (`tests/test_gssm.py`)
shape · state I/O · **streaming==offline (float64, <1e-5)** · zero==None state · gradient flow ·
**causality (no future leakage)** · **ONNX export+ORT match @ T∈{1,16}** · params<LSTM · tied<100K.
