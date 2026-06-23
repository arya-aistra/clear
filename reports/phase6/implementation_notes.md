# Phase 6 — Implementation Notes (post-processor + calibration)

## Goal
Configurable hysteresis post-processor + calibration to set ClearVAD's operating point so FAR
and endpoint latency match-or-beat Silero's default pipeline (the two "worse" cells in the
Phase 5 head-to-head). Honest framing: an operating-point/post-processing win (portable to
Silero), benchmarked fairly.

## Modules
| File | What |
|------|------|
| `postprocess/smoother.py` | `HysteresisSmoother`: asymmetric onset/offset + min_silence/pad end-holdoff + min_speech segment cleaner. Streaming `step` (causal) is canonical; `process` runs it over an array → chunked==offline by construction. `segments_seconds` for the file API. |
| `postprocess/calibrate.py` | grid-search per mode (balanced / high_precision / low_latency) minimizing a mode cost over labeled eval data → best {onset, offset, min_silence, pad}. |

## Operating-point levers (and how they move endpoint latency)
- **offset_threshold** ↑ (toward onset) → releases sooner → faster endpoint (less hysteresis).
- **min_silence_ms** ↓ → commits end sooner → faster endpoint.
- **speech_pad_ms** ↓ → no tail extension → faster endpoint (trades against clipping).
- **min_speech_ms** → drops short false-speech segments → lower FAR.

Endpoint latency ≈ raw-decay + (min_silence + pad) frames. `low_latency` = high offset + small
min_silence + zero pad. `high_precision` = larger hysteresis gap + pad (suppress FAR).

## Calibration & head-to-head (`scripts/calibrate_postproc.py`)
Runs ClearVAD (INT8) over the cached independent eval set, calibrates all 3 modes → writes
`configs/postprocess/{low_latency,balanced,high_precision}.yaml`, then compares **ClearVAD
low_latency vs Silero default** (FAR/MR/endpoint). Also reports **Silero under the same
low_latency profile** — full honesty that the profile is portable; the win is that Silero's
*default* pipeline doesn't use it.

## Acceptance gate
- Endpoint latency (ClearVAD low_latency) < Silero default by a meaningful margin; FAR not worse.

## Tests (`tests/test_smoother.py`, pure numpy)
hysteresis holds through short dip · ends after min_silence · onset/offset asymmetry ·
**streaming==chunked** · min_speech drop · pad extends end · calibrate runs.

## Run
```bash
pytest tests/test_smoother.py -v
python scripts/calibrate_postproc.py --onnx dist/clearvad_lite.onnx
```
