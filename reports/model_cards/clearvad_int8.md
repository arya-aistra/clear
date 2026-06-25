# Model Card — ClearVAD INT8 (`clearvad_lite.onnx`)

## Overview
Compact, CPU-deployable Voice Activity Detector. Causal **Gated-SSM** temporal core, trained with
**no human labels** (Silero distillation + supervised training on constructed real-speech+real-
noise clips). INT8 static-quantized to a single ONNX binary. **Matches and exceeds Silero VAD v5
on accuracy** while being 4.5× smaller and INT8-deployable where Silero's INT8 cannot run.

## Spec
| | |
|--|--|
| Input | 512-sample chunk @ 16 kHz (32 ms) + 64-sample left context = 576 samples |
| Interface | `(chunk[B,576], state[B,256,16]) → (prob[B,1], new_state[B,256,16])` |
| Output | speech probability ∈ (0,1) per 32 ms chunk; state carried externally |
| Params | 220,292 |
| **Binary size** | **0.286 MB** (INT8) · 0.88 MB (FP32) |
| **CPU latency** | **0.170 ms/chunk** (p99 0.18), single thread, ONNX Runtime; 188× real-time |
| Opset | 17 (no Scan/Loop; unrolled SSM) |

## Accuracy vs Silero VAD v5 (independent construction-labeled evals)
| | Silero | ClearVAD FP32 | ClearVAD INT8 |
|--|--|--|--|
| Clean F1 / AUC | 0.838 / 0.836 | **0.923 / 0.957** | **0.892** |
| Noisy 0–12 dB F1 / AUC | 0.848 / 0.842 | **0.919 / 0.856** | **0.882** |
| Miss rate (noisy) | 0.229 | **0.104** | — |
| Onset / endpoint latency | 108 / 47 ms | **16–23 / 13–30 ms** | — |

INT8 beats Silero on **both** clean and noisy conditions. FP32→INT8 drop ≈ 3–4 pp.

## vs Silero — summary
- **Beats Silero on accuracy** (F1/AUC) clean and under real-world noise, with ~½ the miss rate.
- **Detects short silences Silero misses** (50 ms: 66% vs 50%; 100 ms: 84% vs 65%, noisy eval).
- **4.5× smaller**, **INT8-deployable** (Silero's INT8 fails: `ConvInteger NOT_IMPLEMENTED`).
- **Faster onset/endpoint**; **slower** raw per-chunk CPU latency (0.17 vs 0.077 ms; both ≫ real-time).

## Training data (no human VAD labels)
Real speech: LibriSpeech (train-clean-100). Real noise: MUSAN (noise+music), mixed into speech at
0–20 dB SNR. Frame labels from **construction** (we place speech/silence segments). Optional
Silero soft-label distillation.

## Quantization
INT8 static (QDQ, per-channel), Conv/Gemm/MatMul only; **G-SSM recurrence kept FP32** for
stability. Calibrated on 1000 synthetic chunks (wider activation coverage than constructed →
lower INT8 error).

## Limitations
- Eval labels are segment-level (convention-aligned); AUC / miss-rate / short-silence-on-true-gaps
  are the convention-independent wins.
- Train + noisy-eval both use MUSAN → "robust to this noise family"; held-out-noise (DEMAND) and a
  human-labeled benchmark (AVA-Speech) are the next validations.
- FAR under noise slightly above Silero at the default threshold (tunable; ROC dominates).
- Validated on English (LibriSpeech) + MUSAN; broader domains pending.
