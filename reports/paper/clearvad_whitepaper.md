# ClearVAD: A Compact, INT8-Deployable Gated-SSM Voice Activity Detector that Matches and Exceeds Silero with No Human Labels

**Technical Report / White Paper — draft for arXiv + company release**

## Abstract

We present **ClearVAD**, a 220K-parameter streaming voice activity detector built on a causal
**Gated State-Space Model (G-SSM)** temporal core in place of the LSTM used by Silero VAD v5.
ClearVAD is trained with **no human-annotated VAD labels** — using a combination of (i) knowledge
distillation from Silero and (ii) supervised training on *programmatically constructed* clips
(real speech + real noise interleaved at known boundaries). On independent construction-labeled
evaluations ClearVAD **exceeds Silero on accuracy** on both clean speech (F1 0.92 vs 0.84, AUC
0.96 vs 0.84) and real-world noisy speech at 0–12 dB SNR (F1 0.92 vs 0.85, AUC 0.86 vs 0.84, with
less than half the miss rate), while being **4.5× smaller** (0.29 MB INT8 vs 1.29 MB), **INT8-
deployable** (Silero's INT8 fails to run), and lower-latency on onset/endpoint. The model runs at
0.17 ms per 32 ms chunk on a single CPU thread (188× real-time). We also report a **negative
result**: distilling from stronger teachers (Pyannote, NeMo MarbleNet) failed to transfer, which
we trace to a data-construction issue rather than teacher quality — an instructive finding for
label-free VAD distillation.

## 1. Introduction

Voice activity detection (VAD) gates nearly every speech pipeline (ASR, diarization, telephony,
voice agents). Silero VAD v5 is the de-facto open CPU VAD: accurate, ~1.3 MB, LSTM-based. Two
practical gaps motivate ClearVAD: (1) Silero's INT8 quantization **fails to run** on ONNX Runtime
(`ConvInteger NOT_IMPLEMENTED`), blocking the smallest edge deployments; (2) endpoint/short-silence
behavior is governed by a fixed post-processing pipeline. We ask whether a compact **selective
SSM** core, trained **without human labels**, can match or beat Silero while being substantially
smaller and INT8-deployable.

Contributions:
1. **A causal Gated-SSM VAD core** (selective state-space, input-dependent timescale) that is
   ONNX-exportable via an explicit-loop / parallel-scan duality and INT8-quantizable.
2. **A label-free training recipe**: distillation from Silero + supervised training on
   *constructed* real-speech-plus-real-noise clips whose frame labels come from construction.
3. **Empirical results** exceeding Silero on clean and noisy accuracy at 4.5× smaller size and
   working INT8.
4. **A negative result** on multi-teacher distillation and its root-cause analysis.

## 2. Architecture

Input is a 512-sample (32 ms @ 16 kHz) chunk plus 64 samples of left context (576 total).

- **Front-end:** fixed-STFT-style `Conv1d(1→258, k=256, s=128)` → magnitude → 129 features
  (mirrors Silero's measured front-end; learnable, optionally warm-started from Silero).
- **Encoder:** 4 depthwise-separable conv blocks, channel schedule 129→128→64→64→128 (lighter
  than Silero's plain convs; ~38.9K params).
- **G-SSM temporal core (novel):** a Mamba-style *selective* SSM wrapped in a SiLU gate.
  Input-dependent Δ (timescale), B, C; diagonal-negative A = −exp(A_log) for stability;
  recurrence `h_t = exp(Δ_t⊙A)·h_{t-1} + (Δ_t·u_t)⊗B_t`, `y_t = (h_t⊙C_t)Σ + D·u_t`; gated
  fusion `y·SiLU(z)`; residual. 115K params (< the 132K LSTM it replaces).
- **Head:** `Conv1d(128→1)` → per-chunk logit → sigmoid.

**State** (the SSM hidden `h`, [B,256,16]) is carried across chunks; the encoder runs per-chunk
on the 576-window (like Silero), so chunked streaming is **exactly equal** to offline processing.

**Train/deploy duality.** The SSM recurrence is implemented as a log-depth **associative scan**
for fast training and as an **explicit unrolled loop** for export — mathematically identical
(verified to <1e-5), the loop yielding a clean opset-17 ONNX graph with no Scan/Loop op. Total
220,292 params.

## 3. Training without human labels

**Distillation (clean).** Synthetic + real unlabeled speech is labeled by Silero; ClearVAD is
trained to match (temperature-scaled KL + boundary-weighted BCE + temporal smoothness), class-
balanced for the speech minority. Two stages (soft → +boundary/smoothness).

**Constructed supervision (the accuracy unlock).** We build clips by interleaving real speech
(LibriSpeech) with real silence/noise (MUSAN) at **known boundaries**, and mixing noise into
speech at 0–20 dB SNR. Because we place the segments, we have **exact frame labels with no human
annotation**, including realistic *within-clip* speech↔silence transitions. Supervised training on
these labels lets ClearVAD exceed the Silero teacher.

**Negative result (multi-teacher).** Distilling from stronger teachers (Pyannote segmentation,
NeMo Frame-VAD MarbleNet) — single and ensembled with Silero — consistently *failed*: ~0.93
training agreement but near-random held-out AUC. Root cause was **not** teacher quality (NeMo was
cleanly time-aligned with Silero, corr 0.88) but the **training data**: continuous-real-speech +
synthetic clips gave the strong teachers degenerate "real-vs-synthetic" labels (they don't fire on
synthetic), so the student never learned within-audio boundaries. The fix was the constructed data
above, where labels are exact regardless of teacher. We report this because it is a likely pitfall
for others attempting label-free VAD distillation.

## 4. Deployment: INT8 ONNX

A single ONNX binary exports at opset-17 (loop-unrolled SSM). INT8 **static** quantization
(QDQ, per-channel) quantizes only Conv/Gemm/MatMul; the **SSM recurrence stays FP32** — the
precise reason Silero's naive INT8 fails (control-flow + recurrence ops). Result: **0.286 MB**
(4.5× smaller than Silero's 1.29 MB), runs on ONNX Runtime CPU with streaming state carry, at
**0.170 ms / 32 ms chunk** (188× real-time), FP32→INT8 accuracy drop ≈ 3–4 pp (INT8 still exceeds
Silero on both clean and noisy evals).

## 5. Results

See `reports/final_benchmark.md`. Summary on independent construction-labeled evals:

| | Silero v5 | ClearVAD (FP32) | ClearVAD (INT8) |
|--|--|--|--|
| Clean F1 / AUC | 0.838 / 0.836 | **0.923 / 0.957** | 0.892 |
| Noisy (0–12 dB) F1 / AUC | 0.848 / 0.842 | **0.919 / 0.856** | 0.882 |
| Size | 1.29 MB | 0.88 MB | **0.286 MB** |
| Onset / endpoint (ms) | 108 / 47 | **17–23 / 13–30** | — |

ClearVAD wins accuracy (F1, AUC, miss-rate), short-silence detection, size, and onset/endpoint
latency. Silero retains a raw per-chunk CPU-latency advantage (0.077 ms vs 0.170 ms); both are far
under real-time.

## 6. Limitations & honest caveats

- **Segment-level eval labels**: speech segments are labeled all-speech (incl. intra-pauses), so
  part of the F1 gap is convention-alignment; the AUC / miss-rate / short-silence-on-true-gaps
  wins are convention-independent.
- **Noise generalization**: training and noisy eval both use MUSAN — "robust to this noise family";
  held-out-noise (DEMAND/WHAM) and a human-labeled benchmark (AVA-Speech) are the next validations.
- **FAR under noise** is marginally worse than Silero at the default threshold (tunable; ClearVAD's
  ROC dominates).
- **CPU latency** is higher than Silero's fused LSTM.

## 7. Conclusion

A compact Gated-SSM VAD, trained with no human labels, **matches and exceeds Silero VAD v5 on
accuracy** in both clean and realistic-noise conditions, while being **4.5× smaller and INT8-
deployable** where Silero is not. The data-construction insight (and the multi-teacher negative
result) are reusable lessons for label-free VAD.

---
*Reproducibility: full code, configs, eval harness, and per-phase reports in the repository.
All seeds fixed; results in `reports/`.*
