# Phase 1 — Summary (G-SSM block)

**Status: PHASE 1 PASSED** (server run 2026-06-23, `pytest tests/test_gssm.py` → 10 passed).

## Acceptance gates

| Gate | Result | Status |
|------|--------|--------|
| All tests pass | 10 passed (8 core + 2 ONNX) | ✅ |
| ONNX export @ T=1 (streaming) & T=16 (chunk) | both export + ORT-match within 1e-4 | ✅ |
| Streaming equivalence (chunked==offline <1e-5) | float64, exact | ✅ |
| Param count < LSTM equivalent (131,584) | faithful **115,200** | ✅ |
| (Soft) < 100K @ d_inner=256 | only via tying → **82,432** | ✅ via `tie_projections=True` |

## Notes carried forward
- **Faithful default = 115,200 params** (< LSTM). The spec's "<100K @ d_inner=256" is
  arithmetically infeasible with dense in+out projections (98,304 alone); `tie_projections=True`
  reaches 82,432. We A/B faithful-vs-tied accuracy in Phase 3 / ablation.
- **ONNX exporter deprecation:** torch 2.12 defaults to the dynamo exporter; we used the legacy
  exporter (`dynamo=False`) which works but is deprecated (removal planned). **Phase 5 action:**
  validate the dynamo/`torch.export` path yields an equivalent, fast opset-17 graph.
- G-SSM streaming exactness + causality verified → safe to build the full streaming model on top.
