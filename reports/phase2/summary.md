# Phase 2 — Summary (full ClearVAD model)

**Status: PHASE 2 PASSED** (server run 2026-06-23; `test_{frontend,encoder,gssm,full_model}` → all green).

## Acceptance gates

| Gate | Result | Status |
|------|--------|--------|
| All tests pass | frontend + encoder + full_model + gssm all green | ✅ |
| Total params < 300K | **220,292** (frontend 66,048 · encoder 38,915 · gssm 115,200 · head 129) | ✅ |
| Smaller than Silero (309,633 ONNX params) | 220,292 < 309,633 | ✅ |
| ONNX export opset 17 | full graph (conv+BN+unrolled G-SSM+head) exports, ORT-match <1e-4 | ✅ |
| Streaming equivalence (chunked==offline) | rigorous (forced dA≈1), <1e-5 | ✅ |
| Model size on disk (<2MB FP32 / <1MB INT8) | implied ~0.88MB FP32 by param count | ⏳ measured in Phase 5 |

## Verified state behavior (with diagnostics)
- `test_gssm_initial_state_affects_output`: state→output `y` diff **1.09** (state reaches SSM output).
- `test_full_model_state_carry`: `new_state` diff **5.0** (state plumbed through recurrence);
  full-model `prob` diff 1 ULP on the **untrained** model (channel-constant SSM contribution is
  washed out by the random head + residual; resolves after training).
- `streaming==offline` now forces dA≈1 so it genuinely exercises cross-chunk carry.

## Carried to Phase 3
- The untrained head does not yet connect the G-SSM temporal output to the decision — **DFKD must
  forge this link**; watch that the head learns to read the SSM output (not just the residual/skip).
- G-SSM memory ∝ 1/activation-magnitude (Phase 2 notes): monitor learned Δ; LayerNorm/scale before
  the G-SSM is an ablation candidate if the model can't hold speech/silence context.
