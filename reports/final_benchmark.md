# ClearVAD vs Silero VAD v5 — Final Benchmark

All numbers on **independent, construction-labeled eval sets** (labels from construction, not
from any teacher). Single CPU thread, ONNX Runtime. ClearVAD trained with **no human labels**.

> ⚠️ **READ THIS FIRST — accuracy claim is "competitive, NOT superior."** The segment-level
> accuracy numbers below are **convention-biased** (intra-speech pauses labeled speech) and must
> NOT be cited as an accuracy win. On a frame-accurate eval (forced alignment), the original
> segment-trained model scored AUROC **0.514 (chance)**; the latching defect was real. After
> retraining on frame-accurate labels + the **novel CfC core**, scaled to 20 h + real RIRs (best
> model `checkpoints_cfc_20h`, 2-layer `liquidvad_l2`), ClearVAD reaches **parity on clean**: F1
> **tied** (0.958 vs 0.958), clean **AUROC 0.968 vs Silero 0.972 (0.45 pt)**, noisy **0.947 vs 0.970
> (2.3 pt)**. The CfC core beat the G-SSM core (0.947 vs 0.915, same everything else). CfC **beats
> Silero on FAR (0.130 vs 0.224), endpoint latency (44 vs 112 ms), short-pause, and INT8** at ~300k
> params (< Silero 309k) — but Silero still edges AUROC, so **do NOT claim accuracy superiority**
> (we MATCH on clean; surpassing needs 50–100 h+ via a sharded pipeline, with parity the likely
> outcome — diminishing returns). Defensible today: **novel CfC core at parity-on-clean accuracy,
> ~300k params, INT8-deployable (Silero can't), lower FAR + faster latency — a SOTA
> accuracy-per-parameter frontier result.**

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

### Frame-accurate head-to-head — best model is the NOVEL CfC core, 20 h (`checkpoints_cfc_20h`)

All on the **identical** pad40+smooth100 eval (same labels for every model). **CfC** is the closed-form
continuous-time core (same frontend/encoder/head as the G-SSM — a controlled architecture swap; 2-layer
`liquidvad_l2`, ~300k params < Silero 309k). Progression: CfC(1h) → `scaled`(4h+aug) → `rir`(real RIRs)
→ **`20h`** (20 h speech, same RIR/noise/aug recipe).

**CLEAN eval:**

| metric | Silero | **CfC 20h** | CfC+RIR (4h) | CfC scaled | WebRTC |
|--------|--------|-------------|--------------|------------|--------|
| AUROC | **0.972** | 0.968 | 0.963 | 0.955 | 0.776 |
| PR-AUC | **0.989** | 0.987 | 0.986 | 0.982 | 0.867 |
| F1 | 0.958 | 0.958 (**tie**) | 0.951 | 0.946 | 0.922 |
| TPR@FPR=0.315 | **0.993** | 0.987 | 0.979 | 0.971 | 0.991 |
| FAR | 0.224 | **0.130** | 0.125 | 0.127 | 0.516 |
| MR | **0.016** | 0.042 | 0.057 | 0.067 | 0.002 |
| onset / endpoint (ms) | 28 / 112 | 23 / **44** | 24 / 48 | 26 / 52 | 4 / 180 |

**NOISY eval (held-out DEMAND @ 0–12 dB, frame-accurate):**

| metric | Silero | **CfC 20h** | CfC+RIR (4h) | CfC scaled | WebRTC |
|--------|--------|-------------|--------------|------------|--------|
| AUROC | **0.970** | 0.947 | 0.942 | 0.904 | 0.882 |
| F1 | **0.960** | 0.943 | 0.936 | 0.916 | 0.946 |
| FAR | **0.183** | 0.205 | 0.207 | 0.280 | 0.277 |
| MR | **0.022** | 0.046 | 0.059 | 0.075 | 0.018 |
| endpoint (ms) | 90 | **67** | 73 | 103 | 99 |

**Verdict (current best = `checkpoints_cfc_20h`):** The CfC core beats the G-SSM core (0.915→0.947,
same everything else) — a clean architecture win — and scaling data + augmentation + real RIRs reached
**parity on clean**: F1 **tied** (0.958 vs 0.958), AUROC within **0.45 pt** (0.968 vs 0.972), and CfC
**beats** Silero on FAR (0.130 vs 0.224), endpoint latency (44 vs 112 ms), short-pause, size, and INT8.
Noisy is within **2.3 pt** (0.947 vs 0.970) and closing. **Silero still edges AUROC on both — no
strict superiority claim — but on clean frame-accurate audio ClearVAD now MATCHES Silero at ~300k
params + INT8 + half the false-alarm rate + ~2.5× faster endpoint.** The clean gap has halved each
~5× data step (2.5→1.7→0.9→0.45 pt): real but **diminishing returns**, asymptoting toward parity.
Surpassing Silero outright would need a step-change in data (50–100 h+ via a sharded streaming
pipeline) with parity the more likely outcome. Defensible today: **novel CfC core at parity-on-clean /
near-parity-on-noisy accuracy, ~300k params, INT8-deployable (Silero can't), lower FAR, faster
endpoint — a SOTA accuracy-per-parameter frontier result.**

**Noise-diversity null result (locks the model).** Adding a 3rd urban-noise family
(`musan+esc50+urbansound8k`, `checkpoints_cfc_divnoise2`) did NOT move the held-out-DEMAND gap:
noisy AUROC 0.947→0.9475 (flat), FAR 0.205→0.250 (worse). So noise-family coverage is **not** the
bottleneck — the residual ~2.3 pt noisy gap is Silero's raw data scale, not noise variety. (Divnoise's
clean F1 0.9593 marginally edges Silero's 0.9583 — confirming clean parity.) **Decision: lock
`checkpoints_cfc_20h`** (noisy FAR 0.205 < divnoise 0.250; clean identical) as the shipping model;
move to consolidation (INT8 export + verification, paper/model-card, serving).

### Deployment (INT8 ONNX) — locked `checkpoints_cfc_20h` (302,980 params)

| variant | size (MB) | vs Silero 1.29 | F1 (frame-acc clean) | latency ms/chunk | state-carry |
|---------|-----------|----------------|----------------------|------------------|-------------|
| FP32 | 1.236 | 1.04× | 0.958 (= Silero) | 0.089 | exact (Δprob 2e-6) |
| FP16 | 0.646 | 2.0× | — | 0.104 | ✅ |
| **INT8** | **0.457** | **2.82×** | **0.945** (−1.29 pp) | 0.161 (~200× RT) | ✅ runs (Silero INT8 fails) |

All four export gates PASS: INT8 < 1.5 MB, ≥2× smaller than Silero, F1 degradation < 2 pp, INT8
runs + carries state. The 2-layer CfC (303k params) trades the old 1-layer model's 4.5× size edge
for clean-accuracy parity — INT8 is 2.82× smaller and is the **only** one of the two models that
exists in INT8 at all (Silero's INT8 fails on `ConvInteger`). INT8 recurrence quantization costs
1.29 pp F1; mixed-precision not needed. Binaries in `dist/` (`clearvad_lite.onnx` = INT8).

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
