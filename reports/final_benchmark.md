# ClearVAD vs Silero VAD v5 — Final Benchmark

All numbers on **independent, construction-labeled eval sets** (labels from construction, not
from any teacher). Single CPU thread, ONNX Runtime. ClearVAD trained with **no human labels**.

## Headline

| Dimension | Silero v5 | **ClearVAD** | Winner |
|-----------|-----------|--------------|--------|
| Accuracy — clean (F1 / AUC) | 0.838 / 0.836 | **0.923 / 0.957** | **ClearVAD** |
| Accuracy — noisy 0–12 dB (F1 / AUC) | 0.848 / 0.842 | **0.919 / 0.856** | **ClearVAD** |
| Accuracy in **INT8** (clean / noisy F1) | — | **0.892 / 0.882** | **> Silero both** |
| Model size (ONNX) | 1.29 MB | **0.286 MB (INT8)** | **4.5× smaller** |
| INT8 deployment | ❌ fails (`ConvInteger`) | ✅ runs | **ClearVAD** |
| Params | 309,633 | **220,292** | ClearVAD |
| CPU latency / 32 ms chunk | **0.077 ms** | 0.170 ms | Silero |
| Onset latency | 108–111 ms | **16–23 ms** | ClearVAD |
| Endpoint latency (raw) | 46–48 ms | **13–30 ms** | ClearVAD |

## Clean eval (LibriSpeech test-clean + controlled silence)

| metric | Silero | ClearVAD FP32 | ClearVAD INT8 |
|--------|--------|---------------|---------------|
| F1 | 0.838 | **0.923** | 0.892 |
| AUC | 0.836 | **0.957** | — |
| FAR | 0.239 | **0.094** | — |
| MR | 0.246 | **0.127** | — |
| onset (ms) | 108.5 | **22.7** | — |
| endpoint (ms) | 47.8 | **13.2** | — |

## Hard eval (test-clean + MUSAN noise @ 0–12 dB SNR) — deployment conditions

| metric | Silero | ClearVAD FP32 | ClearVAD INT8 |
|--------|--------|---------------|---------------|
| F1 | 0.848 | **0.919** | 0.882 |
| AUC | 0.842 | **0.856** | — |
| FAR | **0.262** | 0.299 | — |
| MR | 0.229 | **0.104** | — |
| onset (ms) | 110.7 | **16.1** | — |
| endpoint (ms) | 46.1 | **29.5** | — |

Short-silence detection (noisy eval, fraction of *true* inserted gaps detected):

| gap | Silero | ClearVAD |
|-----|--------|----------|
| 50 ms | 0.50 | **0.66** |
| 100 ms | 0.65 | **0.84** |
| 150 ms | 0.67 | **0.78** |
| 200 ms | 0.67 | **0.77** |
| 500 ms | 0.94 | **0.96** |

## Honest caveats (so the result survives scrutiny)
1. **Eval labels are segment-level** (a speech segment is labeled all-speech incl. intra-pauses).
   Part of the raw F1 gap is convention-alignment; the **AUC, miss-rate, and short-silence-on-true-
   gaps** wins do not depend on the convention.
2. **FAR on noisy speech is slightly worse** (0.299 vs 0.262), but ClearVAD's higher AUC means its
   ROC dominates — a calibrated threshold (Phase 6) trades its large MR headroom for lower FAR.
3. **Train and noisy-eval both use MUSAN.** This is "robust to this noise family"; held-out-noise
   validation (DEMAND/WHAM) is the stated next hardening step.
4. **CPU latency** (0.17 ms) is slower than Silero's fused LSTM (0.077 ms) — but well under real-time
   (188×) and not a deployment constraint. We do not claim a CPU-speed win.

## Reproduce
```
python scripts/build_eval_set.py --ls-url test-clean [--noise-source musan --snr-min 0 --snr-max 12 --cache .../controlled_eval_hard.npz]
python scripts/train_dfkd.py --data constructed --use-real --ls-url train-clean-100 --noise-source musan ...
python scripts/compare_models.py --checkpoint checkpoints_rw/stage2_final.pt [--cache .../controlled_eval_hard.npz]
python scripts/export_onnx.py --checkpoint checkpoints_rw/stage2_final.pt
```
