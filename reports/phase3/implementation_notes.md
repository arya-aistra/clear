# Phase 3 — Implementation Notes (Data-Free Knowledge Distillation)

## Goal
Train ClearVAD to match Silero's per-chunk outputs using only synthetic audio labeled by
the teacher — no external dataset. Targets: teacher-student agreement >90%, crisp boundaries.

## Pipeline
```
SyntheticAudioGenerator → (clips) → SileroTeacher.label (audio_forward) → soft probs
                                          ↓
ClearVADModel.forward_sequence(windows) → student logits → DFKDLoss(student, teacher) → AdamW
```

## Modules
| File | What |
|------|------|
| `distill/synthetic_gen.py` | 4 categories (noise / speech-like / transitions / hard-neg); numpy, seedable |
| `distill/teacher.py` | Silero soft labels via `audio_forward`; builds 576 student windows (64 ctx + 512), aligned chunk-for-chunk with teacher 512-chunks |
| `distill/losses.py` | `DFKDLoss = λ1·KL_T + λ2·boundary-BCE + λ3·smooth-L1`; `transition_weight`, `agreement_rate` |
| `distill/trainer.py` | `DataPool` (refreshable, teacher-labeled) + `DFKDTrainer` (2 stages, OneCycleLR, logging, ckpt, agreement + real-audio transfer eval) |
| `model/head.py`, `clearvad_model.py` | head exposes pre-sigmoid **logit**; model gains `forward_sequence([B,K,576])→[B,K]` (one G-SSM pass over concatenated chunk features) |

## Key design decisions
1. **Efficient training forward.** `forward_sequence` runs the frontend/encoder per-chunk,
   concatenates features, runs the G-SSM **once** over the full sequence (== streaming by the
   Phase 2 equivalence), then the head per chunk. Avoids a Python per-chunk loop at train time.
2. **Logit-space distillation.** Head reduces per-frame logits → one logit *before* sigmoid, so
   temperature-scaled KD is exact. (Slightly different from Silero's sigmoid-then-mean, but we
   distill to the teacher's final prob anyway.)
3. **Teacher labels via `audio_forward`** (fast batched pass) — labels the whole pool cheaply;
   pool refreshed every `refresh_every` steps for diversity. Teacher on CPU, student on GPU.
4. **DataPool stores raw audio + probs** (not pre-built windows) → windows built per batch
   (cheap), halving memory.

## Honest-reframe additions (beyond spec)
- **`scripts/generate_synthetic.py` measures teacher activation per category** and emits a
  verdict (does speech/transition fire Silero? does noise stay silent?). This is the **DFKD
  viability gate** — if synthetic speech-like audio doesn't fire Silero, DFKD collapses to
  all-silence and we must enrich synthesis or use unlabeled real audio. Surfaced BEFORE training.
- **`DataPool.refresh` logs `teacher_speech_frac`** every refresh — live DFKD health signal.
- **`DFKDTrainer.evaluate_transfer_real`** measures teacher-student agreement on REAL audio
  (the defensible version of Claim 2: matches Silero on real audio, not just synthetic).

## ⚠️ Risks to watch on the server run
- **Synthetic activation** (the gate). Run `generate_synthetic.py` FIRST. If `speech`/`transition`
  don't fire (≥0.5 clips-with-speech), stop and enrich synthesis before the full train.
- **Train speed.** The explicit-loop G-SSM is the accepted slow path; `forward_sequence` over
  K·T_enc = 64·3 = 192 steps × batch 256 is the cost. Reduce `batch_size`/`chunks_per_sample`
  if OOM or too slow. Smoke test uses tiny settings.
- **SSM memory (Phase 2 finding).** Watch agreement on transition-heavy stage 2; if the model
  can't hold speech/silence context, the LayerNorm-before-G-SSM ablation is the lever.

## Tests
- `test_synthetic_gen.py` (numpy): shapes, range, length, reproducibility, categories.
- `test_dfkd_loss.py` (torch): finite, soft→0 when matched, boundary weighting fires, KL≥0,
  gradient flow, stage-1 isolation, agreement_rate.
- `smoke_test_training.py` (silero, slow): 60-step run, loss must decrease.

## Run order (server)
```bash
python scripts/generate_synthetic.py --clips-per-cat 64        # VIABILITY GATE first
pytest tests/test_synthetic_gen.py tests/test_dfkd_loss.py -v   # unit tests
pytest tests/smoke_test_training.py -v -s                       # smoke (loss decreases)
python scripts/train_dfkd.py --real-audio <real_speech.wav>     # full DFKD (long)
```
