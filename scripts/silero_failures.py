#!/usr/bin/env python
"""Phase 0.5 — characterize Silero VAD v5's documented failure modes.

Probes (all controlled stimuli with known ground truth):
  1. Endpoint delay   : speech -> silence; measure ms from true speech end to the
                        model's probability dropping below threshold.
  2. Short silence    : speech | silence(Δ) | speech for Δ in {50,100,150,200,300} ms;
                        does the model's probability dip below threshold during the gap?
  3. INT8 quantization: dynamic-quantize the Silero ONNX and measure output drift vs FP32.
  4. Threshold sweep  : FAR/MR over a labeled concatenation, thresholds 0.1..0.9.

Stimuli use REAL speech if --speech-wav is given (strongly recommended); otherwise a
synthetic speech-like generator is used and the script WARNS if the teacher fails to
activate on it (so a non-firing probe is never mistaken for a model "failure").

Writes:
  reports/silero_failures.md / .json
  reports/silero_threshold_sweep.csv

Run:
    python scripts/silero_failures.py --speech-wav path/to/clean_speech.wav
    python scripts/silero_failures.py            # synthetic fallback
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad import CHUNK_MS, CHUNK_SAMPLES, SAMPLE_RATE  # noqa: E402
from clearvad.evaluation.metrics import binary_frame_metrics, probs_to_labels  # noqa: E402
from clearvad.model.silero_compat import SileroVAD  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_csv, write_json  # noqa: E402
from scripts.benchmark_silero import _classify, _np_dtype, _resolve_shape  # noqa: E402
from scripts.dissect_silero import find_silero_onnx_path  # noqa: E402

LOG = get_logger("failures")
THR = 0.5


# --------------------------------------------------------------- stimulus builders
def synth_speechlike(dur_s: float, f0: float = 120.0, seed: int = 0) -> np.ndarray:
    """Voiced-like signal: harmonic stack + formant bandpass + syllabic AM + noise.

    Intended to trigger Silero's speech response. NOT real speech — if it fails to
    activate the teacher, the caller is warned and should pass --speech-wav.
    """
    rng = np.random.default_rng(seed)
    n = int(dur_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    sig = np.zeros(n, dtype=np.float64)
    # slight F0 jitter/drift
    f0_track = f0 * (1.0 + 0.05 * np.sin(2 * np.pi * 3.0 * t) + 0.02 * rng.standard_normal(n).cumsum() / n)
    phase = 2 * np.pi * np.cumsum(f0_track) / SAMPLE_RATE
    for k in range(1, 25):
        sig += (1.0 / k) * np.sin(k * phase)
    # formant-ish shaping via simple bandpass around 500/1500/2500 Hz
    try:
        from scipy.signal import butter, lfilter

        shaped = np.zeros_like(sig)
        for fc, g in ((500, 1.0), (1500, 0.7), (2500, 0.4)):
            lo, hi = max(fc - 200, 50) / (SAMPLE_RATE / 2), min(fc + 200, 7900) / (SAMPLE_RATE / 2)
            b, a = butter(2, [lo, hi], btype="band")
            shaped += g * lfilter(b, a, sig)
        sig = shaped
    except Exception:  # noqa: BLE001
        pass
    # syllabic amplitude modulation (~4 Hz) + noise floor
    am = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    sig = sig * (0.4 + 0.6 * am)
    sig += 0.01 * rng.standard_normal(n)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.6
    return sig.astype(np.float32)


def load_speech(speech_wav: Optional[str], dur_s: float, seed: int = 0) -> np.ndarray:
    if speech_wav:
        from clearvad.utils.audio import load_audio

        wav = load_audio(speech_wav, SAMPLE_RATE)
        need = int(dur_s * SAMPLE_RATE)
        if len(wav) < need:
            reps = int(np.ceil(need / max(len(wav), 1)))
            wav = np.tile(wav, reps)
        return wav[:need].astype(np.float32)
    return synth_speechlike(dur_s, seed=seed)


def silence(dur_ms: float) -> np.ndarray:
    return np.zeros(int(dur_ms / 1000.0 * SAMPLE_RATE), dtype=np.float32)


# --------------------------------------------------------------- probes
def probe_endpoint_delay(vad: SileroVAD, speech: np.ndarray,
                         sil_ms: float = 1500.0) -> Dict[str, Any]:
    stim = np.concatenate([speech, silence(sil_ms)])
    probs = vad.probabilities(stim, reset=True)
    true_end_frame = len(speech) // CHUNK_SAMPLES  # first silence frame
    activated = bool(probs[:true_end_frame].max() >= THR) if true_end_frame else False
    # first frame at/after true end where prob < THR
    release = None
    for f in range(true_end_frame, len(probs)):
        if probs[f] < THR:
            release = f
            break
    delay_ms = (release - true_end_frame) * CHUNK_MS if release is not None else float("nan")
    return {
        "true_speech_end_frame": int(true_end_frame),
        "model_release_frame": (int(release) if release is not None else None),
        "endpoint_delay_ms": (round(delay_ms, 2) if release is not None else None),
        "teacher_activated_on_speech": activated,
        "max_prob_during_speech": round(float(probs[:true_end_frame].max()), 4) if true_end_frame else None,
    }


def probe_short_silence(vad: SileroVAD, speech_a: np.ndarray, speech_b: np.ndarray,
                        gaps_ms: List[float]) -> List[Dict[str, Any]]:
    rows = []
    for g in gaps_ms:
        stim = np.concatenate([speech_a, silence(g), speech_b])
        probs = vad.probabilities(stim, reset=True)
        start = len(speech_a) // CHUNK_SAMPLES
        gap_frames = max(int(round(g / CHUNK_MS)), 0)
        end = start + gap_frames
        gap_probs = probs[start:end] if end > start else np.array([1.0])
        min_gap = float(gap_probs.min()) if gap_probs.size else float("nan")
        detected = bool(min_gap < THR)
        rows.append({
            "gap_ms": g,
            "gap_frames": gap_frames,
            "min_prob_in_gap": round(min_gap, 4),
            "silence_detected": detected,
        })
    return rows


def _make_ort_runner(onnx_path: str):
    """Return (step_fn, reset_fn) streaming a 512-sample chunk -> prob, carrying state."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    in_meta, out_meta = sess.get_inputs(), sess.get_outputs()
    roles = {m.name: _classify(m) for m in in_meta}
    state_in = next((m for m in in_meta if roles[m.name] == "state"), None)
    out_names = [o.name for o in out_meta]
    state_out_idx = next((i for i, n in enumerate(out_names) if "state" in n.lower()), None)
    if state_out_idx is None and len(out_meta) > 1:
        state_out_idx = len(out_meta) - 1
    prob_out_idx = next((i for i, n in enumerate(out_names) if "output" in n.lower()), 0)

    def build_feeds(audio: np.ndarray, state_val):
        feeds = {}
        for m in in_meta:
            r = roles[m.name]
            dt = _np_dtype(m.type)
            if r == "sr":
                feeds[m.name] = np.array(SAMPLE_RATE, dtype=dt)
            elif r == "state":
                feeds[m.name] = (state_val if state_val is not None
                                 else np.zeros(_resolve_shape(m.shape, "state", len(audio)), dtype=dt))
            else:
                feeds[m.name] = audio.reshape(_resolve_shape(m.shape, "audio", len(audio))).astype(dt)
        return feeds

    # find accepted audio length
    audio_len = CHUNK_SAMPLES
    for cand in (CHUNK_SAMPLES, CHUNK_SAMPLES + 64):
        try:
            sess.run(None, build_feeds(np.zeros(cand, np.float32), None))
            audio_len = cand
            break
        except Exception:  # noqa: BLE001
            continue

    state = {"val": None}

    def reset():
        state["val"] = None

    def step(chunk: np.ndarray) -> float:
        a = chunk[:audio_len] if len(chunk) >= audio_len else np.pad(chunk, (0, audio_len - len(chunk)))
        feeds = build_feeds(a.astype(np.float32), state["val"])
        if state_in is not None and state["val"] is not None:
            feeds[state_in.name] = state["val"].astype(_np_dtype(state_in.type))
        outs = sess.run(None, feeds)
        if state_out_idx is not None:
            state["val"] = outs[state_out_idx]
        return float(np.asarray(outs[prob_out_idx]).reshape(-1)[0])

    return step, reset, audio_len


def probe_int8_quant(onnx_path: str, test_chunks: List[np.ndarray]) -> Dict[str, Any]:
    """Dynamic-quantize the Silero ONNX and measure output drift vs FP32."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"onnxruntime.quantization unavailable: {exc!r}"}

    tmp = Path(tempfile.gettempdir()) / "silero_int8_dynamic.onnx"
    try:
        quantize_dynamic(onnx_path, str(tmp), weight_type=QuantType.QInt8)
    except Exception as exc:  # noqa: BLE001
        return {"status": "quantize_failed", "error": repr(exc)}

    try:
        fp_step, fp_reset, _ = _make_ort_runner(onnx_path)
        q_step, q_reset, _ = _make_ort_runner(str(tmp))
    except Exception as exc:  # noqa: BLE001
        return {"status": "int8_load_failed", "error": repr(exc),
                "note": "Silero INT8 often fails to load/run — this IS a documented failure mode."}

    fp_reset(); q_reset()
    diffs, fp_probs, q_probs = [], [], []
    for c in test_chunks:
        pf = fp_step(c)
        pq = q_step(c)
        fp_probs.append(pf); q_probs.append(pq)
        diffs.append(abs(pf - pq))
    fp_arr, q_arr = np.array(fp_probs), np.array(q_probs)
    agree = float(np.mean(probs_to_labels(fp_arr, THR) == probs_to_labels(q_arr, THR)))
    return {
        "status": "ok",
        "mean_abs_prob_diff": round(float(np.mean(diffs)), 5),
        "max_abs_prob_diff": round(float(np.max(diffs)), 5),
        "agreement_at_0.5": round(agree, 4),
        "int8_size_mb": round(tmp.stat().st_size / 1e6, 4) if tmp.exists() else None,
        "note": "Dynamic quant is the simplest attempt; static (Phase 5) needs calibration.",
    }


def probe_threshold_sweep(vad: SileroVAD, speech: np.ndarray,
                          thresholds: List[float]) -> List[Dict[str, Any]]:
    """Labeled concat [speech|silence|speech|silence]; sweep thresholds -> FAR/MR."""
    seg = speech
    sil = silence(800)
    stim = np.concatenate([seg, sil, seg, sil])
    probs = vad.probabilities(stim, reset=True)
    n = len(probs)
    # ground truth per chunk
    sps = len(seg) // CHUNK_SAMPLES
    sis = len(sil) // CHUNK_SAMPLES
    label = np.zeros(n, dtype=bool)
    cursor = 0
    for is_speech, length in [(True, sps), (False, sis), (True, sps), (False, sis)]:
        end = min(cursor + length, n)
        label[cursor:end] = is_speech
        cursor = end
    rows = []
    for thr in thresholds:
        pred = probs_to_labels(probs[: len(label)], thr)
        m = binary_frame_metrics(pred, label)
        rows.append({"threshold": thr, "far": m["far"], "mr": m["mr"],
                     "f1": m["f1"], "precision": m["precision"], "recall": m["recall"]})
    return rows


# --------------------------------------------------------------- report
def render_md(d: Dict[str, Any]) -> str:
    L = ["# Silero VAD v5 — Failure-Mode Characterization\n",
         "> Auto-generated by `scripts/silero_failures.py`. Stimuli have known ground "
         "truth. Source: **{}**.\n".format(d.get("speech_source"))]
    if not d.get("teacher_activated", True):
        L.append("> ⚠️ **WARNING**: the teacher did NOT strongly activate on the speech "
                 "stimulus (max prob < 0.5). Re-run with `--speech-wav <real_speech>` "
                 "for trustworthy probes.\n")

    ep = d["endpoint_delay"]
    L.append("## 1. Endpoint delay\n")
    L.append(f"- Endpoint delay: **{ep.get('endpoint_delay_ms')} ms** "
             f"(true end frame {ep.get('true_speech_end_frame')}, "
             f"model release frame {ep.get('model_release_frame')})")
    L.append(f"- Max prob during speech: {ep.get('max_prob_during_speech')}\n")

    L.append("## 2. Short-silence detection\n")
    L.append("| gap (ms) | gap frames | min prob in gap | silence detected? |")
    L.append("|----------|-----------|-----------------|-------------------|")
    for r in d["short_silence"]:
        L.append(f"| {r['gap_ms']} | {r['gap_frames']} | {r['min_prob_in_gap']} "
                 f"| {'YES' if r['silence_detected'] else 'NO (MISS)'} |")
    L.append("")

    L.append("## 3. INT8 quantization stability\n")
    q = d["int8_quant"]
    L.append(f"- status: `{q.get('status')}`")
    for k in ("mean_abs_prob_diff", "max_abs_prob_diff", "agreement_at_0.5",
              "int8_size_mb", "error", "note"):
        if k in q:
            L.append(f"- {k}: `{q[k]}`")
    L.append("")

    L.append("## 4. Threshold sensitivity (FAR / MR sweep)\n")
    L.append("| threshold | FAR | MR | F1 |")
    L.append("|-----------|-----|----|----|")
    for r in d["threshold_sweep"]:
        L.append(f"| {r['threshold']} | {r['far']} | {r['mr']} | {r['f1']} |")
    L.append("\n_Full curve in reports/silero_threshold_sweep.csv_\n")

    L.append("## Implications for ClearVAD\n")
    L.append("- **Endpoint delay** → motivates the asymmetric-hysteresis post-processor "
             "(Phase 6) + boundary-weighted distillation loss (Phase 3).")
    L.append("- **Short-silence misses** → motivates the input-dependent timescale (Δ) in "
             "the G-SSM core (Phase 1) so memory can shorten dynamically.")
    L.append("- **INT8 instability** → motivates leaving SSM state ops in FP32 and using "
             "static (calibrated) quantization on Conv/Gemm only (Phase 5).")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-wav", default=None, help="real clean speech wav (recommended)")
    ap.add_argument("--out-md", default="reports/silero_failures.md")
    ap.add_argument("--out-json", default="reports/silero_failures.json")
    ap.add_argument("--out-csv", default="reports/silero_threshold_sweep.csv")
    args = ap.parse_args()

    vad = SileroVAD(onnx=False)
    speech = load_speech(args.speech_wav, dur_s=1.5, seed=0)
    speech_b = load_speech(args.speech_wav, dur_s=1.5, seed=1)

    LOG.info("Probe 1: endpoint delay")
    endpoint = probe_endpoint_delay(vad, speech)
    LOG.info("Probe 2: short silence")
    short_sil = probe_short_silence(vad, speech, speech_b, [50, 100, 150, 200, 300])
    LOG.info("Probe 4: threshold sweep")
    sweep = probe_threshold_sweep(vad, speech, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    LOG.info("Probe 3: INT8 quantization")
    onnx_path = find_silero_onnx_path()
    rng = np.random.default_rng(0)
    test_chunks = ([speech[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
                    for i in range(len(speech) // CHUNK_SAMPLES)]
                   + [rng.standard_normal(CHUNK_SAMPLES).astype(np.float32) * 0.05 for _ in range(50)])
    int8 = (probe_int8_quant(onnx_path, test_chunks) if onnx_path
            else {"status": "error", "error": "silero onnx not found"})

    result = {
        "speech_source": args.speech_wav or "synthetic_speechlike",
        "teacher_activated": endpoint.get("teacher_activated_on_speech", False),
        "endpoint_delay": endpoint,
        "short_silence": short_sil,
        "int8_quant": int8,
        "threshold_sweep": sweep,
    }
    write_json(result, args.out_json)
    write_csv(sweep, args.out_csv)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_md(result), encoding="utf-8")
    LOG.info("Wrote %s, %s, %s", args.out_md, args.out_json, args.out_csv)
    if not result["teacher_activated"]:
        LOG.warning("Teacher did not activate on the speech stimulus — pass --speech-wav.")


if __name__ == "__main__":
    main()
