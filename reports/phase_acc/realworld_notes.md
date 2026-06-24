# Real-World Robustness Track — Implementation Notes

## Why
The accuracy beat (F1 0.92 / AUC 0.96 vs Silero 0.84) was proven on **clean LibriSpeech +
synthetic noise**. Real deployment is **call-centre / babble / traffic / construction / music
+ overlapping speech**. To claim real-world superiority we train AND eval on realistic hard
data — using REAL noise, mixed at low SNR, with labels still from construction (no human labels).

## What was built
| Component | What |
|-----------|------|
| `distill/real_noise.py` | `RealNoiseSource`: MUSAN (OpenSLR `musan.tar.gz`, extracts noise+music) or a local dir; `.sample(n, rng)` random noise slice |
| `distill/constructed_data.py` | `construct_clip` now takes `noise_source`: real noise as non-speech backgrounds **and mixed into speech at 0–20 dB SNR** (hard speech-in-noise, labeled speech). `ConstructedDataPool(noise_source=...)` |
| `evaluation/eval_dataset.py` | `build_eval_set(noise_source=..., snr_range=...)` → HARD noisy eval |
| `scripts/build_eval_set.py` | `--noise-source musan/local --snr-min/--snr-max` → builds the hard eval + re-measures the Silero bar on it |
| `scripts/train_dfkd.py` | `--noise-source musan/local --noise-dir --noise-buffer-seconds` for constructed training |

Labels stay ground-truth: we mix *known* speech (LibriSpeech) with *known* noise (MUSAN) at
*known* times, so speech-vs-non-speech is exact even though the audio is now genuinely hard.

## Run order (server)
```bash
pytest tests/test_real_noise.py tests/test_constructed_data.py -v

# 1. Build the HARD real-world eval (speech + MUSAN @ low SNR). Downloads MUSAN (~11GB) once.
python scripts/build_eval_set.py --ls-url test-clean --noise-source musan \
  --snr-min 0 --snr-max 12 --cache data/eval/controlled_eval_hard.npz \
  --out reports/phase8/silero_bar_hard.json

# 2. Train supervised on REAL-NOISE constructed data, more volume (train-clean-100, 100h)
python scripts/train_dfkd.py --data constructed --aux-teacher none \
  --model-config configs/model/clearvad_base.yaml \
  --stage1 configs/train/supervised_stage1.yaml --stage2 configs/train/supervised_stage2.yaml \
  --use-real --ls-url train-clean-100 --buffer-seconds 36000 \
  --noise-source musan --noise-buffer-seconds 3600 \
  --out-dir checkpoints_rw 2>&1 | tee reports/phase_acc/train_rw.log

# 3. Head-to-head on the HARD eval
python scripts/compare_models.py --checkpoint checkpoints_rw/stage2_final.pt \
  --model-config configs/model/clearvad_base.yaml --cache data/eval/controlled_eval_hard.npz \
  --out reports/phase8/comparison_hard.json
```

## Notes / knobs
- MUSAN is ~11 GB (downloads once from OpenSLR — the host LibriSpeech came from). Use
  `--noise-dir <path>` to point at pre-downloaded / local noise and skip the download.
- `train-clean-100` is ~6 GB (100 h vs the 5 h dev-clean we used) — bigger volume.
- SNR 0–12 dB is "hard"; raise for milder conditions. For babble/commotion specifically, add a
  babble noise dir via `--noise-dir`.
- The win to verify: ClearVAD AUC/F1 + short-silence on the HARD eval still beating Silero
  (Silero degrades under noise; a noise-trained ClearVAD should hold up better).
