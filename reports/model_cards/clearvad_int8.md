# Model Card — ClearVAD INT8 (`clearvad_lite.onnx`)

## Overview
Compact, CPU-deployable Voice Activity Detector. Causal **Gated-SSM** temporal core distilled
(label-free) from Silero VAD v5, INT8 static-quantized to a single ONNX binary.

## Spec
| | |
|--|--|
| Input | 512-sample chunk @ 16 kHz (32 ms) + 64-sample left context = 576 samples |
| Interface | `(chunk[B,576], state[B,256,16]) → (prob[B,1], new_state[B,256,16])` |
| Output | speech probability ∈ (0,1) per 32 ms chunk; state carried externally |
| Params | 220,292 | 
| **Binary size** | **0.286 MB** (INT8) · 0.88 MB FP32 |
| **CPU latency** | **0.170 ms/chunk** (p99 0.175), single thread, ONNX Runtime; RTF 0.0053 (188× real-time) |
| Opset | 17 |

## vs Silero VAD v5
- **4.5× smaller** ONNX binary (0.286 MB vs 1.29 MB).
- **INT8-deployable** — Silero's INT8 fails to run (`ConvInteger NOT_IMPLEMENTED`); ClearVAD runs.
- **Accuracy parity** on an independent eval set (F1 0.829 vs 0.838; FP32→INT8 drop 0.31 pp).
- **Faster onset** (~66 ms vs ~108 ms).
- Slower raw per-chunk latency than Silero (0.17 ms vs 0.077 ms) — still sub-0.2 ms / 188× RT.

## Training
Data-free of human labels: distilled from Silero (teacher) on real unlabeled LibriSpeech
(dev-clean) + synthetic edge cases; class-balanced; 2-stage (soft → +boundary/smoothness).

## Quantization
INT8 static (QDQ), Conv/Gemm/MatMul only; G-SSM recurrence kept FP32 for stability. Calibrated
on 1000 streamed synthetic chunks.

## Limitations
- Accuracy is bounded by the Silero teacher (parity, not better) — accuracy-beating requires the
  multi-teacher track (Silero + Pyannote).
- Endpoint/FAR operating point set by the Phase 6 post-processor; raw model is slightly eager.
- Validated on LibriSpeech-derived English; broader-domain eval pending.
