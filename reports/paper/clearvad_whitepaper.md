# ClearVAD: A Compact, INT8-Deployable Continuous-Time (CfC) Voice Activity Detector Matching Silero with No Human Labels

**Technical Report / White Paper — draft for arXiv + company release**

## Abstract

We present **ClearVAD**, a ~303K-parameter streaming voice activity detector whose temporal core is
a **closed-form continuous-time (CfC) recurrence** (Hasani et al. 2022) — an architecture, to our
knowledge, not previously applied to VAD — in place of the LSTM used by Silero VAD v5. ClearVAD is
trained with **no human-annotated VAD labels**: frame-accurate targets are obtained by **forced
alignment** (torchaudio MMS_FA) of read-speech transcripts, augmented with real noise (MUSAN, ESC-50)
and real room impulse responses (OpenSLR SLR28). On a **frame-accurate** evaluation (forced-aligned
labels, scored identically for every model), ClearVAD **matches** Silero on clean speech (F1 0.958
vs 0.958, AUROC 0.968 vs 0.972) and is near-parity under **unseen** (DEMAND) noise (AUROC 0.947 vs
0.970), while being **smaller** (0.457 MB INT8 vs 1.29 MB), **INT8-deployable** (Silero's INT8 fails
to run), with **roughly half the false-alarm rate** (0.130 vs 0.224) and **~2.5× faster endpoint
latency** (44 ms vs 112 ms). We make two methodological contributions beyond the model: (1) we show
that the common **segment-level** VAD eval convention (intra-speech pauses labeled as speech) can
**inflate accuracy by ~40 AUROC points** for a model trained to that convention, and provide a
forced-alignment eval that removes the bias; and (2) a **controlled architecture ablation** (CfC vs a
selective state-space / G-SSM core, identical front-end) in which CfC wins (AUROC 0.947 vs 0.915). We
explicitly **do not claim accuracy superiority** over Silero; the contribution is matching a mature
production VAD at a fraction of the deployment cost with a novel, tiny, continuous-time core.

## 1. Introduction

Voice activity detection (VAD) gates nearly every speech pipeline (ASR, diarization, telephony,
voice agents). Silero VAD v5 is the de-facto open CPU VAD: accurate, ~1.3 MB, LSTM-based. Two
practical gaps motivate ClearVAD: (1) Silero's INT8 quantization **fails to run** on ONNX Runtime
(`ConvInteger NOT_IMPLEMENTED`), blocking the smallest edge deployments; (2) we ask whether a
compact, **continuous-time** core trained **without human labels** can match Silero while being
smaller and INT8-deployable.

Contributions:
1. **A CfC (closed-form continuous-time) VAD core** — a tiny streaming recurrence, ONNX-exportable
   (no Scan/Loop, no ODE solver) and INT8-quantizable — apparently the first use of CfC for VAD.
2. **A label-free, frame-accurate training recipe** using forced alignment + real-noise/RIR
   augmentation, with a study of how accuracy scales with training-data hours.
3. **A frame-accurate evaluation methodology** that exposes and removes the segment-level
   convention bias common in VAD reporting.
4. **A controlled CfC-vs-SSM ablation** and an honest near-parity result vs Silero.

## 2. Architecture

Input is a 512-sample (32 ms @ 16 kHz) chunk plus 64 samples of left context (576 total).

- **Front-end:** STFT-style `Conv1d(1→258, k=256, s=128)` → 129 magnitude features (mirrors Silero;
  learnable). **Encoder:** 4 depthwise-separable conv blocks (129→128→64→64→128, ~38.9K params).
  These are shared across architectures so the temporal-core comparison is controlled.
- **CfC temporal core (novel for VAD):** a 2-layer closed-form continuous-time cell. Each step
  interpolates two input-dependent candidate states via a time-gated sigmoid,
  `h_t = ff1(x_t,h_{t-1})·(1−σ(t_a·Δt+t_b)) + ff2(x_t,h_{t-1})·σ(t_a·Δt+t_b)` — elementwise +
  matmuls only, so it streams (carry `h`) and exports cleanly. Hidden 128, ~198K params.
- **Head:** `Conv1d(128→1)` → per-chunk logit → sigmoid.

**State** (`h`, [B,2,128]) is carried across chunks; the encoder runs per-chunk on the 576-window,
so chunked streaming equals offline processing. The CfC recurrence over the T_enc=3 sub-frames
unrolls into a static opset-17 graph (no Scan/Loop). Total **302,980** params (< Silero's 309,633).

A selective state-space (Mamba-style **G-SSM**) core with the *same* front-end/encoder/head is
retained as an ablation baseline (§5).

## 3. Training without human labels

**Frame-accurate labels via forced alignment.** We align LibriSpeech transcripts to audio with
torchaudio's MMS_FA aligner, yielding word time-spans; inter-word / leading / trailing silence is
labeled non-speech. A 100 ms min-silence smoothing (co-articulation gaps are not pauses) and 40 ms
word-edge padding (MMS_FA trims onsets) give clean per-frame targets — **no human VAD annotation**.

**Construction + augmentation.** Aligned speech is interleaved with real silence/noise at known
boundaries and mixed with real noise (MUSAN, ESC-50) at 0–20 dB SNR; clips are further augmented
with **real room impulse responses** (OpenSLR SLR28), mu-law codec distortion, and gain. The final
model is trained **purely supervised** on the forced-alignment labels (no teacher).

**Data scaling.** Accuracy improves monotonically with training hours and augmentation diversity;
the clean-eval gap to Silero closed 2.5 → 1.7 → 0.9 → 0.45 AUROC points as speech scaled 1 h → 20 h,
with diminishing returns approaching parity (using ~0.4 % of available LibriSpeech).

## 4. Evaluation methodology: segment-level vs frame-accurate

Many VAD evals label a "speech segment" as all-speech, including intra-utterance pauses. A model
trained to that convention scores its own bias back: our segment-trained model reached F1 0.92 /
AUROC 0.96 on segment-level labels but **AUROC 0.514 (chance)** on forced-aligned frame-accurate
labels — it had learned speech-*region* detection, not frame-level VAD, and failed to release on
real multi-second pauses. We therefore evaluate **only** on frame-accurate forced-aligned labels,
scored identically for every model (Silero, ClearVAD, WebRTC). This is a cautionary, reusable
finding for VAD reporting.

## 5. Deployment: INT8 ONNX

A single ONNX binary exports at opset-17 (CfC unrolled). INT8 **static** quantization (QDQ,
per-channel) yields **0.457 MB** (2.82× smaller than Silero's 1.29 MB), runs on ONNX Runtime CPU
with streaming state carry — where Silero's INT8 fails — at **0.161 ms / 32 ms chunk** (~200×
real-time). FP32→INT8 F1 drop is **1.29 pp** (0.958 → 0.945); mixed-precision was not required.

## 6. Results

Frame-accurate eval (forced alignment; identical labels for all models). Full detail in
`reports/final_benchmark.md`.

| | Silero v5 | **ClearVAD (CfC)** | WebRTC |
|--|--|--|--|
| Clean F1 / AUROC | 0.958 / 0.972 | **0.958** / 0.968 | 0.922 / 0.776 |
| Noisy (held-out DEMAND) F1 / AUROC | 0.960 / 0.970 | 0.943 / 0.947 | 0.946 / 0.882 |
| False-alarm rate (clean) | 0.224 | **0.130** | 0.516 |
| Onset / endpoint (clean, ms) | 28 / 112 | 23 / **44** | 4 / 180 |
| Params / INT8 size | 309,633 / ❌ INT8 | 302,980 / **0.457 MB** | — |

**Architecture ablation:** swapping the temporal core G-SSM→CfC (identical front-end/encoder/head)
lifted clean AUROC **0.915 → 0.947** — CfC is the stronger compact VAD core. ClearVAD **matches**
Silero on clean (F1 tied), is near-parity on noisy, and **wins** false-alarm rate, endpoint latency,
size, and INT8-deployability; Silero retains a small AUROC edge and a raw per-chunk CPU-latency
advantage (0.077 vs 0.161 ms; both far under real-time).

## 7. Limitations & honest caveats

- **No accuracy-superiority claim.** Silero leads AUROC on both clean (0.972 vs 0.968) and noisy
  (0.970 vs 0.947); ClearVAD matches on clean and is near-parity on noisy. Silero is modestly more
  noise-robust, attributable to its far larger (thousand-hour) training corpus.
- **Labels are forced-alignment-derived**, not fully human; a human-labeled benchmark (AVA-Speech)
  is the next external validation.
- **Data budget.** Trained on ~20 h of English read speech + 3 noise families; closing the residual
  noise gap further would need a step-change in data scale (sharded streaming, 50–100 h+), with
  parity the likely outcome (diminishing returns observed).
- **CPU per-chunk latency** is higher than Silero's fused LSTM (both ≫ real-time).

## 8. Conclusion

A tiny continuous-time (CfC) VAD, trained with **no human labels** on forced-alignment targets,
**matches Silero VAD v5** on clean frame-accurate accuracy and is **near-parity under unseen noise**,
while being smaller, **INT8-deployable** where Silero is not, with lower false-alarm rate and faster
endpoint latency. The frame-accurate evaluation (which exposes segment-level convention bias) and
the CfC-vs-SSM ablation are reusable contributions for compact, label-free VAD.

---
*Reproducibility: full code, configs, eval harness, and per-phase reports in the repository.
All seeds fixed; results in `reports/`. Locked model: `checkpoints_cfc_20h` (config `liquidvad_l2`).*
