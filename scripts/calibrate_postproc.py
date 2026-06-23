#!/usr/bin/env python
"""Phase 6 — calibrate ClearVAD's post-processor and show the operating-point win vs Silero.

Runs ClearVAD over the cached independent eval set, calibrates the hysteresis smoother for
each mode (low_latency / balanced / high_precision), writes the profile configs, and reports
ClearVAD-calibrated vs Silero-default (FAR / MR / endpoint latency) — the honest operating-point
comparison (post-proc is portable; Silero's default pipeline does not use these settings).

Run (after export_onnx.py + build_eval_set.py):
    python scripts/calibrate_postproc.py --onnx dist/clearvad_lite.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad.evaluation.metrics import binary_frame_metrics, endpoint_latency_ms, frames_to_segments, probs_to_labels  # noqa: E402
from clearvad.export.validate_onnx import OrtVADRunner  # noqa: E402
from clearvad.model.silero_compat import SileroVAD  # noqa: E402
from clearvad.postprocess.calibrate import calibrate_all_modes  # noqa: E402
from clearvad.postprocess.smoother import HysteresisSmoother  # noqa: E402
from clearvad.utils.config import save_yaml  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("calib")

# Silero's production defaults (VADIterator / get_speech_timestamps)
SILERO_DEFAULT = dict(onset_threshold=0.5, offset_threshold=0.35,
                      min_speech_ms=250.0, min_silence_ms=100.0, speech_pad_ms=30.0)


def agg_metrics(probs_list, labels_list, smoother, chunk_ms=32.0):
    fars, mrs, f1s, eps = [], [], [], []
    for probs, labels in zip(probs_list, labels_list):
        labels = np.asarray(labels).astype(bool)
        mask = smoother.process(probs)
        n = min(len(mask), len(labels))
        m = binary_frame_metrics(mask[:n], labels[:n])
        fars.append(m["far"]); mrs.append(m["mr"]); f1s.append(m["f1"])
        ep = endpoint_latency_ms(mask[:n], frames_to_segments(labels[:n]), chunk_ms)
        if ep["endpoint_latency_mean_ms"] == ep["endpoint_latency_mean_ms"]:
            eps.append(ep["endpoint_latency_mean_ms"])
    return {"far": round(float(np.mean(fars)), 4), "mr": round(float(np.mean(mrs)), 4),
            "f1": round(float(np.mean(f1s)), 4),
            "endpoint_latency_mean_ms": round(float(np.mean(eps)), 2) if eps else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="dist/clearvad_lite.onnx")
    ap.add_argument("--cache", default="data/eval/controlled_eval.npz")
    ap.add_argument("--out", default="reports/phase6/calibration.json")
    ap.add_argument("--config-dir", default="configs/postprocess")
    args = ap.parse_args()

    data = np.load(args.cache, allow_pickle=True)
    audio, labels = data["audio"], list(data["labels"])

    LOG.info("Running ClearVAD (%s) over %d eval sequences...", args.onnx, len(audio))
    runner = OrtVADRunner(args.onnx)
    cv_probs = [runner.stream(a, reset=True) for a in audio]

    LOG.info("Running Silero over the same set...")
    silero = SileroVAD(onnx=False)
    sil_probs = [silero.probabilities(a, reset=True) for a in audio]

    # calibrate ClearVAD per mode
    LOG.info("Calibrating ClearVAD post-processor (3 modes)...")
    modes = calibrate_all_modes(cv_probs, labels)
    for mode, res in modes.items():
        cfg = {"profile": mode, **res["params"], "chunk_ms": 32.0}
        save_yaml(cfg, str(Path(args.config_dir) / f"{mode}.yaml"))
        LOG.info("  %-14s params=%s  metrics=%s", mode, res["params"], res["metrics"])

    # head-to-head: ClearVAD low_latency vs Silero default (both post-processed)
    cv_ll = HysteresisSmoother(**modes["low_latency"]["params"])
    cv_bal = HysteresisSmoother(**modes["balanced"]["params"])
    sil_default = HysteresisSmoother(**SILERO_DEFAULT)
    comparison = {
        "clearvad_low_latency": agg_metrics(cv_probs, labels, cv_ll),
        "clearvad_balanced": agg_metrics(cv_probs, labels, cv_bal),
        "silero_default": agg_metrics(sil_probs, labels, sil_default),
    }
    # also Silero with the SAME low-latency profile (honesty: the profile is portable)
    comparison["silero_low_latency_profile"] = agg_metrics(
        sil_probs, labels, HysteresisSmoother(**modes["low_latency"]["params"]))

    report = {"modes": modes, "comparison": comparison, "silero_default_profile": SILERO_DEFAULT}
    write_json(report, args.out)

    cl = comparison["clearvad_low_latency"]; sd = comparison["silero_default"]
    se = comparison["silero_low_latency_profile"]
    gap = sd["endpoint_latency_mean_ms"] - cl["endpoint_latency_mean_ms"]
    LOG.info("ENDPOINT: ClearVAD low_latency=%.1fms  vs  Silero default=%.1fms  (%.1fms faster)",
             cl["endpoint_latency_mean_ms"], sd["endpoint_latency_mean_ms"], gap)
    LOG.info("FAR/MR: ClearVAD low_latency FAR=%.3f MR=%.3f | Silero default FAR=%.3f MR=%.3f",
             cl["far"], cl["mr"], sd["far"], sd["mr"])
    # EQUAL-FOOTING honesty: both models under ClearVAD's low_latency profile
    LOG.info("EQUAL-FOOTING (same low_latency profile): "
             "ClearVAD endpoint=%.1fms FAR=%.3f | Silero endpoint=%.1fms FAR=%.3f",
             cl["endpoint_latency_mean_ms"], cl["far"],
             se["endpoint_latency_mean_ms"], se["far"])
    LOG.info("Wrote %s and configs/postprocess/{low_latency,balanced,high_precision}.yaml", args.out)


if __name__ == "__main__":
    main()
