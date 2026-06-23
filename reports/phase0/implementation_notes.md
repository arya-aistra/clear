# Phase 0 — Implementation Notes

## What was built (code only; execution pending on the GPU server)

| Component | File | Purpose |
|-----------|------|---------|
| Project scaffold | `pyproject.toml`, `requirements*.txt`, `LICENSE`, `.gitignore` | Packaging; lean inference deps vs full train extras |
| Frozen constants | `clearvad/__init__.py` | `SAMPLE_RATE=16000`, `CHUNK_SAMPLES=512`, `CHUNK_MS=32`, `CONTEXT_SAMPLES=64`, `TOTAL_INPUT_SAMPLES=576` |
| Teacher wrapper | `clearvad/model/silero_compat.py` | `SileroVAD`: `forward`, `reset_states`, `streaming_step`, `get_state`, `describe`, `probabilities` |
| Metrics (frozen) | `clearvad/evaluation/metrics.py` | F1/FAR/MR/AUC + onset/endpoint latency + segment extraction |
| Audio utils | `clearvad/utils/audio.py` | load/resample/chunk, RMS-normalize, SNR mixing, frame↔ms |
| Config/seed | `clearvad/utils/config.py` | YAML load/save, global seeding (py/numpy/torch) |
| Logging/IO | `clearvad/utils/logging_utils.py` | logger, JSON/CSV writers |
| Eval protocol | `configs/eval/benchmark.yaml` | FROZEN chunk size, threshold grid, post-proc defaults, latency settings |
| Env logger | `scripts/log_environment.py` | → `environment.yaml` (versions, CUDA, ORT providers) |
| Dissection | `scripts/dissect_silero.py` | → `reports/silero_dissection.{md,json}` |
| Baseline bench | `scripts/benchmark_silero.py` | → `reports/phase0_baseline.json` |
| Failure probes | `scripts/silero_failures.py` | → `reports/silero_failures.{md,json}`, `threshold_sweep.csv` |
| Tests | `tests/test_metrics.py`, `test_audio.py`, `test_silero_wrapper.py`, `conftest.py` | pure + teacher-gated |

## Key design decisions

1. **Scripts introspect, never assume.** The dissection reads ground-truth shapes from the
   JIT `named_parameters()` and the ONNX graph (`onnx.shape_inference`) rather than hardcoding
   the spec's `[512,128]` LSTM / `[1,1,128]` state numbers. The spec values appear only in a
   §6 "reconciliation" table to be checked against measured values. This is deliberate: Silero
   v5's exact ONNX I/O has varied across point releases, and a dissection that assumes is useless.

2. **Two latency numbers, both honest.** `benchmark_silero.py` reports (a) *end-to-end* streaming
   latency (context concat + ORT — what production pays) and (b) *pure-ORT* `session.run` latency
   (inference only). The headline baseline is (a); (b) isolates the model. Both are single-thread
   `CPUExecutionProvider`. We will compare ClearVAD on the *same* two axes.

3. **Generic ORT feed builder.** Because Silero v5's audio input length (512 vs 576 with context)
   and state I/O names differ by release, the benchmark/failure runners auto-detect the accepted
   audio length and classify inputs (audio/state/sr) by name+dtype+shape. Robust to version drift.

4. **Failure probes self-validate.** `silero_failures.py` checks whether the teacher actually
   activated (max prob ≥ 0.5) on the speech stimulus. If a *synthetic* stimulus fails to fire the
   teacher, it warns and recommends `--speech-wav`. We never report a "Silero miss" that is really
   a non-firing probe. Real speech is strongly recommended for trustworthy endpoint/short-silence numbers.

5. **Lean inference deps.** `pyproject` base deps are numpy + onnxruntime + pyyaml only. torch and
   the silero teacher live in extras/`requirements.txt`. The eventual pip package + Docker image
   never need PyTorch. `silero_compat` lazy-imports torch so importing the module in a torch-free
   env doesn't break.

## Open items to reconcile after the server run

- Confirm measured LSTM `weight_ih`/`weight_hh` shapes vs spec `[512,128]`.
- Confirm carried state shape (spec says `h_n`/`c_n` `[1,1,128]`; v5 ONNX often uses a single
  combined `state [2,B,128]`). The dissection §2 reports the truth.
- Confirm Silero v5 ONNX audio input length (512 vs 576) — drives the pure-ORT bench.
- Confirm `audio_forward` exists on the loaded JIT backend (drives the streaming-equivalence test;
  test skips gracefully if absent).
