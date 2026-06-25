# ClearVAD

> A compact, CPU-deployable **SSM-based Voice Activity Detector** distilled from Silero VAD v5.
> Single ONNX binary, no GPU at inference, pip-installable, FastAPI-servable.

ClearVAD replaces Silero's LSTM temporal core with a compact **causal Gated SSM (G-SSM)**
block, trains the student via **data-free knowledge distillation** from Silero v5, and ships
as an INT8-quantized, structurally-pruned ONNX model targeting sub-millisecond CPU inference.

**Status:** Model complete — **beats Silero VAD v5 on accuracy** (clean *and* real-world noise),
trained with **no human labels**, shipping as a **0.29 MB INT8 ONNX** binary. Serving (FastAPI)
next. See `reports/final_benchmark.md` and `reports/paper/clearvad_whitepaper.md`.

## Results vs Silero VAD v5 (independent, construction-labeled; no human labels in training)

| | Silero v5 | **ClearVAD** |
|--|--|--|
| Clean F1 / AUC | 0.838 / 0.836 | **0.923 / 0.957** |
| Noisy 0–12 dB F1 / AUC | 0.848 / 0.842 | **0.919 / 0.856** |
| INT8 F1 (clean / noisy) | ❌ INT8 fails to run | **0.892 / 0.882** |
| Size (ONNX) | 1.29 MB | **0.286 MB INT8 (4.5×)** |
| Onset / endpoint latency | 108 / 47 ms | **16–23 / 13–30 ms** |
| CPU latency / chunk | 0.077 ms | 0.170 ms (188× real-time) |

ClearVAD wins accuracy, miss-rate, short-silence detection, size, INT8-deployability, and
onset/endpoint latency. Honest caveats (segment-level eval labels, MUSAN-noise train/eval,
tunable FAR) are documented in `reports/final_benchmark.md`.

---

## Install

### Inference only (lean — no PyTorch)
```bash
pip install clearvad           # depends only on numpy + onnxruntime + pyyaml
```

### Full training / distillation / export (GPU server)
```bash
pip install -r requirements.txt          # torch, silero-vad teacher, onnx tooling
pip install -e ".[train,serve,dev]"      # editable install with all extras
```

> **ONNX Runtime note:** ClearVAD's latency claims are **CPU-only**. Use the `onnxruntime`
> (CPU) build, *not* `onnxruntime-gpu`. The benchmark scripts force single-thread
> `CPUExecutionProvider` regardless, but a CPU-only build avoids ambiguity.

---

## Phase 0 — run order (remote GPU/Jupyter server)

```bash
# 0.1 record the exact environment
python scripts/log_environment.py

# 0.3 dissect Silero's architecture (shapes, ops, params)  -> reports/silero_dissection.{md,json}
python scripts/dissect_silero.py

# 0.4 benchmark Silero as the CPU baseline                 -> reports/phase0_baseline.json
python scripts/benchmark_silero.py --measure-chunks 10000 --warmup 200

# 0.5 characterize failure modes                           -> reports/silero_failures.{md,json}, threshold_sweep.csv
python scripts/silero_failures.py --speech-wav <real_clean_speech.wav>   # real speech strongly recommended

# 0.7 tests
pytest -q                       # pure tests always run; silero tests run if teacher loads
pytest -q -m requires_silero    # just the teacher-wrapper tests
```

Paste the terminal output back and we reconcile any shape mismatches against the spec
before starting Phase 1.

---

## Evaluation / test audio

The frozen evaluation protocol lives in [`configs/eval/benchmark.yaml`](configs/eval/benchmark.yaml).
Two ground-truth modes are supported:

1. **`manual`** — your own labeled audio. Put mono WAVs in `data/eval/manual/audio/` and a
   matching `data/eval/manual/labels/<name>.json` per file:
   ```json
   { "speech": [[0.50, 2.30], [3.10, 5.80]] }    // speech intervals in SECONDS
   ```
   A self-recorded 20–30 minute set across quiet/noisy conditions is ideal.

2. **`silero_pseudolabel`** — drop real speech (e.g. LibriSpeech `test-clean`) into
   `data/eval/librispeech/`. Silero labels it; this measures *transfer/agreement* of the
   distilled student to the teacher on **real** audio (the honest version of Claim 2).
   It does **not** measure absolute accuracy — the labels are the teacher.

Synthetic transition/short-silence probes are generated on the fly (no download).

---

## Repository layout

```
clearvad/
  model/        frontend, encoder, gssm (novel core), head, full model, silero shim
  distill/      synthetic generator, teacher wrapper, DFKD losses + trainer
  postprocess/  hysteresis smoother + threshold calibration
  export/       ONNX export, INT8 quantization, validation, CPU benchmark
  streaming/    chunker, SSM state manager, endpoint detector
  serve/        FastAPI app + schemas + middleware
  evaluation/   frozen metrics, benchmark runner, model comparison
  utils/        audio, config, logging
configs/        all YAML (model / train / eval / export / postprocess)
scripts/        runnable entrypoints
tests/          unit, shape, streaming-equivalence, export validation
reports/        per-phase artifacts (dissection, baselines, benchmarks)
```

Phases land incrementally — only Phase 0 modules exist today.

## License
MIT. ClearVAD is distilled from Silero VAD v5 (MIT). See [LICENSE](LICENSE).
