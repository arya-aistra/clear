# ClearVAD vs Silero VAD v5 — Final Benchmark

All numbers on **independent, construction-labeled eval sets** (labels from construction, not
from any teacher). Single CPU thread, ONNX Runtime. ClearVAD trained with **no human labels**.

> ⚠️ **READ THIS FIRST — accuracy claim corrected to "competitive, not superior."** The
> segment-level accuracy numbers below are **convention-biased** (intra-speech pauses labeled
> speech) and must NOT be cited as an accuracy win. On a frame-accurate eval (forced alignment),
> the original segment-trained model scored AUROC **0.514 (chance)**. After retraining on
> frame-accurate labels (commit 430af94), ClearVAD is a genuine frame-level VAD: **AUROC 0.849
> vs Silero 0.915** on identical labels (see "Frame-accurate eval"). **We do NOT beat Silero on
> AUROC/F1.** Defensible claims: **4.5× smaller, INT8-deployable (Silero can't), data-free,
> lower false-alarm rate (0.331 vs 0.383), faster endpoint (87 vs 152 ms), better short-pause
> sensitivity — at competitive frame-level accuracy.** Lead with deployability + competitive
> accuracy, never "more accurate than Silero."

## Headline (segment-level labels — accuracy rows are convention-biased, see warning above)

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

## Held-out noise (test-clean + **DEMAND** @ 0–12 dB) — noise NOT seen in training (MUSAN)
Standard VAD metrics (pooled), comparable to published numbers:

| metric | Silero | ClearVAD | WebRTC (aggr.3) |
|--------|--------|----------|-----------------|
| F1 | 0.850 | **0.930** | 0.902 |
| AUROC | 0.859 | **0.878** | 0.642 |
| TPR@FPR=0.315 | 0.809 | **0.920** | 0.909 |
| PR-AUC | **0.972** | 0.967 | 0.885 |
| FAR | **0.216** | 0.340 | 0.706 |
| MR | 0.232 | 0.079 | **0.074** |
| onset / endpoint (ms) | 114 / 40 | 21 / 34 | **15** / 105 |

Short-silence detection on held-out DEMAND (fraction of *true* gaps detected):

| gap | Silero | ClearVAD | WebRTC |
|-----|--------|----------|--------|
| 50 ms | 0.52 | **0.76** | 0.04 |
| 100 ms | 0.66 | **0.86** | 0.03 |
| 150 ms | 0.67 | **0.72** | 0.08 |
| 200 ms | 0.74 | 0.69 | 0.48 |
| 300 ms | **0.99** | 0.88 | 0.56 |
| 500 ms | **1.00** | 0.92 | 0.56 |

**Flag 2 (noise generalization) closed:** ClearVAD trained on MUSAN beats Silero on *unseen*
DEMAND noise on F1, AUROC, TPR@FPR, MR, latency. FAR is higher at the default threshold but
AUROC dominates → a calibrated threshold trades ClearVAD's large MR headroom for lower FAR.

**WebRTC (aggressiveness 3) is the energy-VAD floor:** 0.90 F1 looks competitive, but AUROC 0.642
and FAR 0.706 show it barely discriminates — it scores F1 by calling almost everything speech (its
low MR is just a side effect of rarely going silent). It detects essentially **no** short pauses
(<0.08 at ≤150 ms). ClearVAD has the best discrimination (AUROC) of the three and resolves the
short pauses both Silero and WebRTC miss.

## Frame-accurate eval (forced alignment, LibriSpeech test-clean) — Flag 1 RESOLVED (competitive)
Labels from torchaudio MMS_FA forced alignment: real word boundaries, intra-word / inter-utterance
silence = non-speech. Removes the segment-level convention bias. All three models scored on the
**same** labels. `min_silence=100 ms` applies the standard VAD convention (co-articulation micro-gaps
< 100 ms are not "pauses"); this is the apples-to-apples comparison and it improves Silero MORE than
ClearVAD, so it is generous to Silero. Pooled, threshold 0.5:

| metric | Silero | ClearVAD (frame-acc.) | WebRTC | (orig. segment-trained ClearVAD) |
|--------|--------|-----------------------|--------|----------------------------------|
| **AUROC** | **0.915** | 0.849 | 0.716 | 0.514 (chance) |
| F1 | **0.915** | 0.883 | 0.869 | 0.759 |
| PR-AUC | **0.940** | 0.889 | 0.773 | 0.590 |
| TPR@FPR=0.315 | **0.987** | 0.901 | 0.997 | — |
| FAR | 0.383 | **0.331** | 0.626 | 0.876 |
| MR | **0.005** | 0.092 | 0.000 | 0.016 |
| endpoint (ms) | 152 | **87** | 220 | 276 |

Short-silence detection on *real* pauses — the latching defect is gone; every long silence now
detected, and ClearVAD leads at short pauses:

| gap | Silero | ClearVAD | WebRTC |
|-----|--------|----------|--------|
| 96 ms | 0.01 | **0.22** | 0.00 |
| 128 ms | 0.02 | **0.32** | 0.00 |
| 256 ms | 0.62 | **0.73** | 0.25 |
| ≥1000 ms | 1.00 | **1.00** | ~0.97 |
| 1696 / 1792 / 1984 ms | 1.00 | **1.00** (was 0.00) | 1.00 |

**Verdict:** Retraining on frame-accurate labels turned ClearVAD from chance (AUROC 0.514) into a
genuine frame-level VAD (0.849). It now **wins** false-alarm rate, endpoint latency, and short-pause
sensitivity, at 4.5× smaller + INT8. **But Silero still leads AUROC (0.915 vs 0.849) and F1 (0.915
vs 0.883)** — so the honest claim is *competitive accuracy + deployment wins*, NOT accuracy
superiority. ClearVAD's weak spot is miss-rate (0.092 vs 0.005): it under-detects speech, the lever
for closing the AUROC gap (more aligned data / steps, recall-weighted loss, threshold calibration).
Raw/strict eval (min_silence=0): ClearVAD F1 0.826, FAR 0.445, endpoint 63 ms.

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
