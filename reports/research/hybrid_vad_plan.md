# Hybrid VAD research program — tiny novel core distilled from a foundation teacher

**Goal (honest):** a genuinely novel, *tiny*, streaming VAD that is **SOTA on accuracy-per-parameter
/ per-watt** and *competitive on absolute accuracy* with the leaders (Silero, Pyannote, NeMo,
FireRed, TEN), with WebRTC as the floor. Beating everyone on **absolute** AUROC is a stretch goal,
not a promise — the realistic, defensible contribution is the Pareto frontier (accuracy at a fraction
of size/latency/power, INT8-deployable where others fail).

## Why this shape (and not "a brand-new architecture beats everyone")
The VAD accuracy frontier is set by **data + self-supervised representations**, not by novel layers
(Pyannote/academic SOTA ride WavLM/wav2vec2). A novel layer trained on scrapeable data loses to a
boring CNN that distills a foundation model. So the novelty that *pays* is **efficiency**: a tiny
core that inherits a strong teacher's accuracy. The novel contribution is the **system**
(foundation-teacher → tiny continuous-time student, multi-corpus), not a layer in isolation.

## Architecture
- **Frontend + encoder + head:** reuse ClearVAD's proven STFT frontend + depthwise-separable conv
  encoder + mean head. Unchanged → controlled comparison.
- **Temporal core (NEW): CfC** (Closed-form Continuous-time cell, Hasani et al. 2022). Continuous-time
  RNN whose ODE solution is closed-form (interpolate two candidates via a time-gated sigmoid). Tiny,
  streaming (carry hidden state), ONNX-friendly (elementwise + matmuls, no scan, no ODE solver).
  Untouched for VAD. We A/B it against the existing G-SSM core (same everything else) — a clean
  ablation and a paper result either way.
- Keep both cores behind a model factory (`arch: gssm | cfc`) so the trainer/eval are shared.

## Teacher (foundation-grade, multi-teacher, anchored)
Soft-label distillation blended with our **forced-alignment true labels** (the anchor that prevents
any single teacher's bias/misalignment from dominating — this is what fixes the earlier pyannote
failure). Teachers, in order of integration:
1. Silero (have it, 0.972 AUROC on our frame-acc eval) — first blend.
2. NeMo **Frame-VAD MarbleNet** (per-20ms speech prob, ONNX) — clean frame-level.
3. Pyannote **segmentation-3.0** (strong; handle frame-rate/powerset alignment carefully).
Ensemble = mean of frame-aligned teacher probs; `true_weight` blends with alignment ground truth.

## Data (phased by coverage-per-effort; avoid paid/fragile first)
- **Phase-1 (free, reliable, alignable):** LibriSpeech + Libri-Light subset (read; forced-align),
  AMI/ICSI (meetings, far-field; provided annotations), CommonVoice (multilingual), TED-LIUM.
  Noise: MUSAN, DEMAND, WHAM!/WHAMR!, DNS-Challenge, ESC-50. Far-field: convolve with RIRs (SLR28).
- **Phase-2 (free but fragile / gold eval):** AVA-Speech (gold VAD labels — the benchmark; some train),
  VoxConverse. **Paid (maybe skip):** Switchboard/Fisher, DIHARD (LDC). AudioSet (YouTube-fragile).
- **Labels:** read speech → forced alignment (have it). Conversational → provided VAD/diarization.
  Noise-only → 0. Far-field → RIR-convolved clean speech keeps labels.

## Evaluation (to claim anything vs the field)
- **AVA-Speech** is the standard (report TPR@FPR + AUROC). Add our frame-accurate LibriSpeech set,
  a noisy held-out set, and a conversational set (AMI/VoxConverse test).
- Competitors on the SAME sets: Silero, Pyannote, NeMo, WebRTC (TEN/FireRed if installable).
- Always compare checkpoints on the SAME eval cache; change ONE variable per run (lesson from fa2).

## Phases
- **P0 (now):** CfC core + model factory; train on EXISTING aligned constructed data + Silero blend;
  eval on the pad40 frame-acc set vs G-SSM `fa` (0.915). Gate: CfC ≥ G-SSM, else iterate the core.
- **P1:** multi-corpus data pipeline (LibriLight/AMI/CommonVoice + RIR + richer noise) + label harmonization.
- **P2:** multi-teacher ensemble (add NeMo Frame-VAD, pyannote-seg) anchored by true labels.
- **P3:** AVA-Speech + multi-domain eval vs the full field. INT8 export. Efficiency-frontier claims.
- **P4:** paper (accuracy-per-param frontier + CfC-vs-SSM ablation + multi-corpus robustness).

## Honest risks
- Absolute-accuracy SOTA is not guaranteed; the per-param frontier is the bankable win.
- Paid corpora (LDC) and YouTube-sourced sets (AudioSet/AVA) are cost/decay risks → Phase-2.
- CfC may not beat G-SSM; if so we keep G-SSM and the program still stands (data+teacher carry it).
- A student distilled from a teacher is bounded *toward* the teacher; true labels are the only route
  to *exceed* it locally.
