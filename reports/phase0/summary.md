# Phase 0 — Summary

**Goal:** Characterize Silero VAD v5 (architecture, CPU baseline, failure modes) and freeze a
reproducible evaluation protocol before building anything.

**Hypothesis:** Silero v5 has reproducible weaknesses (endpoint delay, short-silence misses, INT8
instability) that a distilled SSM student can target; baseline CPU latency ≈ 0.3 ms/chunk.

## Acceptance gates

| Gate | How it's satisfied | Status |
|------|--------------------|--------|
| Dissection document complete | `scripts/dissect_silero.py` → `reports/silero_dissection.md` | ⏳ pending server run |
| Baseline CPU latency measured & logged | `scripts/benchmark_silero.py` → `reports/phase0_baseline.json` | ⏳ pending server run |
| Failure modes reproducible | `scripts/silero_failures.py` → `reports/silero_failures.md` | ⏳ pending server run |
| Evaluation protocol fixed in config | `configs/eval/benchmark.yaml` (frozen, committed) | ✅ done |
| Wrapper tests pass | `pytest -q` (pure + `requires_silero`) | ⏳ pending server run |
| `reports/phase0_baseline.json` saved | written by benchmark script | ⏳ pending server run |

**Code-side status:** all Phase 0 modules/scripts/tests written and byte-compile cleanly.
Numbers are produced on the server. This file's table gets its ✅/❌ filled from pasted output.

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
