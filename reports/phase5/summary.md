# Phase 5 — Summary (INT8 + single ONNX binary)

**Status: PHASE 5 PASSED** (server run 2026-06-23). All gates green.

## The deployment win (vs Silero v5)
| metric | Silero v5 | ClearVAD INT8 | result |
|--------|-----------|---------------|--------|
| ONNX size | 1.29 MB | **0.286 MB** | **4.5× smaller** |
| INT8 deployable | ❌ fails (`ConvInteger NOT_IMPLEMENTED`) | ✅ runs + state carry | **capability Silero lacks** |
| F1 (independent set) | 0.838 | 0.829 (FP32 0.832) | parity; **0.31 pp** INT8 drop |
| CPU latency / chunk | ~0.077 ms | 0.170 ms (p99 0.175) | slower (claim dropped); **188× real-time** |

## Acceptance gates
| Gate | Result |
|------|--------|
| INT8 size < 1.5 MB | ✅ 0.286 MB |
| INT8 2–3× smaller than Silero | ✅ **4.5×** |
| FP32→INT8 F1 degradation < 2 pp | ✅ 0.31 pp |
| INT8 runs + streaming state carry | ✅ true |

## How
- FP32 export (opset-17, loop-scan unrolls T_enc=3, legacy exporter).
- INT8 static quant, **QDQ format** (ORT x64 perf), Conv/Gemm/MatMul only — **G-SSM recurrence
  stays FP32** (the Silero `ConvInteger` lesson). BN folded via `quant_pre_process`; MinMax
  calibration on 1000 streamed synthetic chunks (states carried as in deployment).
- Validated: FP32 numeric ≡ torch (<1e-3), streaming state carry, determinism. INT8: runs,
  carries state, 0.31 pp F1 drop.

## Artifacts
`dist/clearvad_base.onnx` (FP32), `dist/clearvad_fp16.onnx`, `dist/clearvad_lite.onnx` (INT8, 0.286 MB).

## Next
- Phase 6 — hysteresis post-processor → tune FAR/endpoint to a clean operating point.
- Phase 7 — FastAPI serving (onnxruntime-only, no torch).
- Multi-teacher (Silero + Pyannote) — the accuracy-beating track.
