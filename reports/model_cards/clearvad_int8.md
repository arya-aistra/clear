# Model Card — ClearVAD-CfC INT8 (`clearvad_lite.onnx`)

## Overview
Compact, CPU/INT8-deployable Voice Activity Detector with a **novel closed-form continuous-time
(CfC) temporal core** (Hasani et al. 2022) — an architecture essentially untouched for VAD. Trained
with **no human VAD labels**: frame-accurate targets come from **forced alignment** (torchaudio
MMS_FA) of LibriSpeech transcripts, with real-noise and real room-impulse-response augmentation.
INT8 static-quantized to a single ONNX binary.

**Honest positioning:** on clean frame-accurate audio ClearVAD **matches** Silero VAD v5 (F1 tied,
AUROC within 0.4 pt) and **wins** on false-alarm rate, endpoint latency, size, and INT8-deployability;
it is near-parity under unseen noise. We do **not** claim accuracy superiority over Silero (Silero
edges AUROC and is modestly more noise-robust thanks to its far larger training corpus).

## Spec
| | |
|--|--|
| Input | 512-sample chunk @ 16 kHz (32 ms) + 64-sample left context = 576 samples |
| Interface | `(chunk[B,576], state[B,2,128]) → (prob[B,1], new_state[B,2,128])` |
| Output | speech probability ∈ (0,1) per 32 ms chunk; state carried externally |
| Temporal core | 2-layer CfC (closed-form continuous-time), hidden 128 |
| Params | 302,980 (< Silero 309,633) |
| **Binary size** | **0.457 MB INT8** · 0.646 MB FP16 · 1.236 MB FP32 |
| **CPU latency** | **0.161 ms/chunk INT8** (p99 0.19), single thread, ONNX Runtime; ~200× real-time |
| Opset | 17 (no Scan/Loop; CfC recurrence unrolled over T_enc=3) |

## Accuracy vs Silero VAD v5 — frame-accurate eval (forced alignment, identical labels both models)
| | Silero | ClearVAD FP32 | ClearVAD INT8 |
|--|--|--|--|
| Clean F1 / AUROC | 0.958 / 0.972 | **0.958** (tie) / 0.968 | 0.945 / — |
| Noisy (held-out DEMAND) F1 / AUROC | 0.960 / 0.970 | 0.943 / 0.947 | — |
| False-alarm rate (clean) | 0.224 | **0.130** | 0.131 |
| Miss rate (clean) | **0.016** | 0.042 | 0.067 |
| Onset / endpoint latency (clean) | 28 / 112 ms | 23 / **44 ms** | — |

FP32→INT8 F1 drop ≈ **1.29 pp** (0.958 → 0.945). All four export gates pass (INT8 < 1.5 MB,
≥2× smaller than Silero, < 2 pp degradation, INT8 runs + carries state).

## vs Silero — summary (honest)
- **Matches** Silero on clean frame-accurate accuracy (F1 tied 0.958; AUROC 0.968 vs 0.972).
- **Near-parity** under unseen (DEMAND) noise (AUROC 0.947 vs 0.970).
- **Wins:** false-alarm rate (0.130 vs 0.224), endpoint latency (44 vs 112 ms), short-pause detection,
  size, and **INT8-deployability** (Silero's INT8 fails: `ConvInteger NOT_IMPLEMENTED`).
- **Novel architecture:** the CfC core beat the prior selective-SSM (G-SSM) core 0.947 vs 0.915 with
  identical frontend/encoder/head — a controlled ablation.
- **Does NOT beat Silero on AUROC/F1**; Silero is modestly more noise-robust (larger training corpus).

## Training data (no human VAD labels)
Speech: LibriSpeech train-clean-360 (~20 h sampled). Frame-accurate labels via **MMS_FA forced
alignment** (word spans + 100 ms min-silence smoothing + 40 ms edge padding). Real noise: MUSAN +
ESC-50 mixed at 0–20 dB SNR. Augmentation: real RIRs (OpenSLR SLR28), mu-law codec, gain. Pure
supervised on the forced-alignment true labels (no teacher in the final model).

## Quantization
INT8 static (QDQ, per-channel), Conv/Gemm/MatMul. Calibrated on 1000 synthetic chunks (wider
activation coverage than in-distribution → lower INT8 error). INT8 recurrence quantization costs
1.29 pp F1; mixed-precision was not required.

## Limitations
- Silero leads AUROC on both clean and noisy; ClearVAD matches on clean and is near-parity on noisy.
  No accuracy-superiority claim.
- Frame-accurate labels come from forced alignment (MMS_FA); a fully human-labeled benchmark
  (AVA-Speech) is the next external validation.
- Validated on English (LibriSpeech) with MUSAN/ESC-50/urban + DEMAND (held-out) noise; broader
  languages/domains pending (would require the multi-corpus data program).
- Closing the residual noise gap further needs a step-change in training data scale (sharded
  streaming pipeline, 50–100 h+), with parity the likely outcome (diminishing returns observed).
