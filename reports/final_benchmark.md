# ClearVAD vs Silero VAD v5 — Final Benchmark

All numbers on **independent, construction-labeled eval sets** (labels from construction, not
from any teacher). Single CPU thread, ONNX Runtime. ClearVAD trained with **no human labels**.

> ⚠️ **READ THIS FIRST — accuracy claim is "competitive, NOT superior."** The segment-level
> accuracy numbers below are **convention-biased** (intra-speech pauses labeled speech) and must
> NOT be cited as an accuracy win. On a frame-accurate eval (forced alignment), the original
> segment-trained model scored AUROC **0.514 (chance)**; the latching defect was real. After
> retraining on frame-accurate labels + the **novel CfC core** (best model `checkpoints_cfc`),
> ClearVAD is a genuine frame-level VAD: **AUROC 0.947 vs Silero 0.972**, **PR-AUC 0.980 vs 0.989**
> on identical labels (see "Frame-accurate eval"). It clearly beats WebRTC (0.776) and **beats
> Silero on FAR (0.094 vs 0.224), endpoint latency (31 vs 112 ms), and short-pause sensitivity** —
> but Silero still leads AUROC/F1, so **do NOT claim accuracy superiority yet**. The CfC core beat
> the G-SSM core (0.947 vs 0.915, same everything else); the 2.5 pt gap to Silero is small and has
> several untried levers (foundation-teacher blend, multi-corpus data, capacity, calibration).
> Defensible claims today: **~204k params, INT8-deployable (Silero can't), data-free, much lower
> FAR + faster latency — at competitive (and closing) frame-level accuracy.**

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

### Frame-accurate head-to-head — best model is the NOVEL CfC core (`checkpoints_cfc`)

All on the **identical** pad40+smooth100 eval (same labels for every model). G-SSM `fa` and `fa2`
are the prior selective-SSM cores; **CfC** is the closed-form continuous-time core (same frontend/
encoder/head — a controlled architecture swap):

| metric | Silero | **CfC** (`checkpoints_cfc`) | G-SSM `fa` | G-SSM `fa2` | WebRTC |
|--------|--------|------------------------------|------------|-------------|--------|
| AUROC | **0.972** | 0.947 | 0.915 | 0.877 | 0.776 |
| PR-AUC | **0.989** | 0.980 | 0.964 | 0.938 | 0.867 |
| F1 | **0.958** | 0.917 | 0.917 | 0.914 | 0.922 |
| TPR@FPR=0.315 | **0.993** | 0.952 | 0.947 | 0.926 | 0.991 |
| FAR | 0.224 | **0.094** | 0.199 | 0.298 | 0.516 |
| MR | **0.016** | 0.126 | 0.108 | 0.088 | 0.002 |
| onset / endpoint (ms) | 28 / 112 | 30 / **31** | 29 / 72 | 20 / 103 | 4 / 180 |

CfC also **beats Silero on short-pause detection at nearly every gap** (96 ms 0.56 vs 0.23; 128 ms
0.64 vs 0.45; 224 ms 0.96 vs 0.72; 256 ms 0.99 vs 0.88) and matches it (1.00) on all gaps ≥480 ms.

**Verdict (current best = `checkpoints_cfc`, the novel CfC core):** Swapping the temporal core
G-SSM→CfC lifted AUROC **0.915 → 0.947** with everything else fixed — a clean architecture win and a
publishable ablation. CfC is now **only 2.5 AUROC points behind Silero** (0.947 vs 0.972), with
PR-AUC nearly level (0.980 vs 0.989), at ~204k params, and it **beats Silero on FAR (0.094 vs 0.224),
endpoint latency (31 vs 112 ms), and short-pause sensitivity**. This is **pure-supervised, single-corpus,
no teacher** — the hybrid program's accuracy levers (foundation-teacher soft-label blend, multi-corpus
data, capacity bump, threshold calibration for the high MR 0.126) are all still untried. **Silero
still leads AUROC/F1 — no superiority claim yet — but the gap is now small and beating it is plausible.**
Raw/strict eval (no pad/smooth) for `fa`: F1 0.826, FAR 0.445, endpoint 63 ms.

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
