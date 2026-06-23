# Phase 0 — Summary

**Goal:** Characterize Silero VAD v5 (architecture, CPU baseline, failure modes) and freeze a
reproducible evaluation protocol before building anything.

**Hypothesis:** Silero v5 has reproducible weaknesses (endpoint delay, short-silence misses, INT8
instability) that a distilled SSM student can target; baseline CPU latency ≈ 0.3 ms/chunk.

## Acceptance gates — ALL PASSED (server run 2026-06-23)

| Gate | How it's satisfied | Status |
|------|--------------------|--------|
| Dissection document complete | `reports/silero_dissection.md` (shapes/ops/params measured) | ✅ PASS |
| Baseline CPU latency measured & logged | `reports/phase0_baseline.json` (0.0774 ms e2e / 0.0538 ms pure-ORT) | ✅ PASS |
| Failure modes reproducible | `reports/silero_failures.md` (+ `findings.md` reconciliation) | ✅ PASS |
| Evaluation protocol fixed in config | `configs/eval/benchmark.yaml` (frozen, committed) | ✅ PASS |
| Wrapper tests pass | `pytest -q` → 22 passed, 1 skipped (state-carry, backend hides state) | ✅ PASS |
| `reports/phase0_baseline.json` saved | written by benchmark script | ✅ PASS |

**Status: PHASE 0 PASSED.** All gates met. See `findings.md` for the spec reconciliation and the
narrative-shifting results (Silero raw is already fast/crisp; endpoint & short-silence "failures"
are post-processing artifacts; CPU-speed-beating is off the table; size/INT8/data-free survive).

## Commands to run on the server

```bash
pip install -r requirements-dev.txt
python scripts/log_environment.py
python scripts/dissect_silero.py
python scripts/benchmark_silero.py --measure-chunks 10000 --warmup 200
python scripts/silero_failures.py --speech-wav <real_clean_speech.wav>
pytest -q
```

## What Phase 0 feeds forward

- **Dissection** → Phase 1/2 target shapes (param budget < LSTM's; state dims to replace).
- **Baseline JSON** → the number every later phase must beat (latency, size, F1/FAR/MR).
- **Failure modes** → justify each ClearVAD design choice (G-SSM Δ for short silences, hysteresis
  for endpoint delay, FP32-state INT8 for quant stability).
- **Frozen eval config** → guarantees apples-to-apples comparison through Phase 8.
