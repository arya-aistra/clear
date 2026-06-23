# Phase 6 — Summary (post-processor + calibration)

**Status: infrastructure COMPLETE; endpoint/FAR gate NOT a clean win (model-limited).**

## Built & tested
- `HysteresisSmoother` (asymmetric thresholds, min_silence/pad end-holdoff, min_speech cleaner);
  streaming `step` canonical, chunked==offline verified. 7/7 tests pass.
- `calibrate` (per-mode grid search) → wrote `configs/postprocess/{low_latency,balanced,high_precision}.yaml`.

## Honest head-to-head (independent eval set, post-processed)
| comparison | endpoint | FAR | MR |
|------------|----------|-----|----|
| ClearVAD low_latency vs **Silero default** | 59.1 ms vs 116.8 ms ✅ faster | 0.289 vs 0.461 ✅ lower | 0.285 vs 0.202 |
| ClearVAD vs Silero **EQUAL profile** | 59.1 ms vs **35.2 ms** ❌ slower | 0.289 vs **0.180** ❌ higher | — |

**Verdict:** the "vs default" advantage is a config difference (Silero's default is conservative).
At equal post-processing **Silero wins endpoint + FAR** — ClearVAD's raw probabilities are
**sticky / less sharp** (the SSM bridges silence gaps). Endpoint and FAR are therefore **not**
clean architecture wins.

## Root cause (unifies 3 metrics)
The distilled student's probabilities are **less sharp than the teacher's**. One fact explains:
- higher FAR (0.29 vs 0.18) — trigger-happy in gaps,
- lingering endpoint (sticky),
- lower AUC (0.785 vs 0.836).
This is the distillation ceiling. Post-processing cannot fix a blurry raw signal — it needs a
**model improvement**.

## Clean, equal-footing wins that DO hold (the shippable deployment-SOTA story)
- **Size 4.5× smaller** (0.286 MB INT8 vs 1.29 MB) ✅
- **INT8-deployable** (Silero's INT8 fails) ✅
- **Onset latency faster** (66 ms vs 108 ms, equal raw threshold) ✅
- **Accuracy parity** (F1 0.83) ✅

## Next (the lever that converts endpoint + FAR + AUC + accuracy at once)
Sharper student probabilities via the **multi-teacher track** (Silero + Pyannote) + anti-stickiness
levers (↑boundary-loss weight, LayerNorm-before-G-SSM from the Phase 2 finding, more
transition/silence data). A crisper teacher → crisper student → lower FAR, faster endpoint,
higher AUC, and the accuracy beat — all from one track.
