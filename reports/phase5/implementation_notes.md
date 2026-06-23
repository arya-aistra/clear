# Phase 5 — Implementation Notes (INT8 + single ONNX binary)

## Goal
Deployable artifacts: FP32 → FP16 → INT8 single ONNX binary, validated + benchmarked. The
headline win Silero can't match: **~3× smaller AND INT8 that actually runs**.

## Modules (`clearvad/export/`)
| File | What |
|------|------|
| `to_onnx.py` | FP32 export of the streaming graph `(chunk[B,576], state[B,256,16]) → (prob, new_state)`; G-SSM 'loop' scan unrolls T_enc=3 → opset-17, no Scan/Loop; legacy exporter preferred |
| `quantize.py` | FP16 (onnxconverter-common) + **INT8 static** (quantize_static); calibration streamed through the FP32 model (states carried as in deployment) |
| `validate_onnx.py` | `OrtVADRunner` (deployment streaming) + numeric / state-carry / determinism checks |
| `benchmark_onnx.py` | single-thread CPU latency (mean/p50/p90/p99), throughput, RTF, size, RSS |

## The INT8 stability decision (the Silero lesson)
Quantize **only `Conv`/`Gemm`/`MatMul`** (frontend conv, encoder convs, head conv, G-SSM
projections). The G-SSM recurrence ops (`Exp`/`Mul`/`Add`/`ReduceSum`/`Softplus`/`Sigmoid`)
stay **FP32** — these are exactly what broke Silero's naive INT8 (`ConvInteger NOT_IMPLEMENTED`
in Phase 0). `quant_pre_process` folds BatchNorm into convs first. QOperator format, per-channel
weights, MinMax calibration on 1000 streamed synthetic chunks.

## Pipeline (`scripts/export_onnx.py`)
load checkpoint → export fp32/fp16/int8 → validate each → **F1 degradation FP32 vs INT8 on the
independent eval set** → benchmark each → gate check → `reports/phase5/export_report.json` +
`dist/clearvad_{base,fp16,lite}.onnx`.

## Acceptance gates (checked in the report)
- INT8 binary < 1.5 MB
- INT8 **2-3× smaller than Silero** (1.29 MB)
- FP32→INT8 F1 degradation < 2 pp
- INT8 runs + streaming state carry holds in ONNX

## Tests
- `test_onnx_export.py`: FP32 export + numeric equivalence (<1e-3) + state carry + determinism.
- `test_onnx_quantized.py`: INT8 quantizes, runs, carries state, smaller than FP32.

## Run (server)
```bash
pytest tests/test_onnx_export.py tests/test_onnx_quantized.py -v
python scripts/export_onnx.py --checkpoint checkpoints/stage2_final.pt
```
