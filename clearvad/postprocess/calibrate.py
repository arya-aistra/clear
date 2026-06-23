"""Threshold calibration: grid-search the hysteresis smoother for a target operating mode.

Minimizes a mode-specific cost over a labeled eval set:
  * balanced       : 0.5·FAR + 0.5·MR
  * high_precision : 0.8·FAR + 0.2·MR  (suppress false alarms)
  * low_latency    : endpoint_ms/100 + 0.5·(FAR+MR)  (fast endpoint, kept reasonable)

Returns the best {onset, offset, min_silence_ms, speech_pad_ms} + its metrics, ready to write
to a configs/postprocess/*.yaml.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from clearvad import CHUNK_MS
from clearvad.evaluation.metrics import binary_frame_metrics, endpoint_latency_ms, frames_to_segments
from clearvad.postprocess.smoother import HysteresisSmoother

# One broad grid for ALL modes; the mode-specific COST (not the grid) decides the operating
# point. (A larger min_silence bridges silence gaps -> raises FAR, so high_precision must be
# free to choose min_silence=0 — hence a shared grid rather than per-mode restricted grids.)
GRID = dict(onset=[0.4, 0.5, 0.6, 0.7, 0.8], min_silence_ms=[0, 20, 40, 60, 100], pad=[0, 30])
HYST_GAP = [0.0, 0.1, 0.15, 0.2]  # onset - offset
MODES = ("balanced", "high_precision", "low_latency")


def evaluate_params(probs_list, labels_list, params: Dict, chunk_ms: float = CHUNK_MS) -> Dict:
    sm = HysteresisSmoother(chunk_ms=chunk_ms, **params)
    fars, mrs, f1s, eps = [], [], [], []
    for probs, labels in zip(probs_list, labels_list):
        labels = np.asarray(labels).astype(bool)
        mask = sm.process(probs)
        n = min(len(mask), len(labels))
        m = binary_frame_metrics(mask[:n], labels[:n])
        fars.append(m["far"]); mrs.append(m["mr"]); f1s.append(m["f1"])
        ep = endpoint_latency_ms(mask[:n], frames_to_segments(labels[:n]), chunk_ms)
        v = ep["endpoint_latency_mean_ms"]
        if v == v:  # not NaN
            eps.append(v)
    return {
        "far": round(float(np.mean(fars)), 4),
        "mr": round(float(np.mean(mrs)), 4),
        "f1": round(float(np.mean(f1s)), 4),
        "endpoint_latency_mean_ms": round(float(np.mean(eps)), 2) if eps else float("nan"),
    }


def _cost(mode: str, m: Dict) -> float:
    far, mr = m["far"], m["mr"]
    ep = m["endpoint_latency_mean_ms"]
    ep = 1e3 if ep != ep else ep  # NaN → large
    if mode == "high_precision":
        return 0.8 * far + 0.2 * mr
    if mode == "low_latency":
        return ep / 100.0 + 0.5 * (far + mr)
    return 0.5 * far + 0.5 * mr  # balanced


def calibrate(probs_list: Sequence, labels_list: Sequence, mode: str = "balanced",
              chunk_ms: float = CHUNK_MS) -> Dict:
    """Grid-search the smoother for `mode`. Returns best params + metrics + cost."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    best = None
    for onset in GRID["onset"]:
        for gap in HYST_GAP:
            offset = round(max(0.05, onset - gap), 3)
            for min_sil in GRID["min_silence_ms"]:
                for pad in GRID["pad"]:
                    params = dict(onset_threshold=onset, offset_threshold=offset,
                                  min_silence_ms=min_sil, speech_pad_ms=pad,
                                  min_speech_ms=100.0)
                    m = evaluate_params(probs_list, labels_list, params, chunk_ms)
                    c = _cost(mode, m)
                    if best is None or c < best["cost"]:
                        best = {"mode": mode, "params": params, "metrics": m,
                                "cost": round(c, 5)}
    return best


def calibrate_all_modes(probs_list, labels_list, chunk_ms: float = CHUNK_MS) -> Dict[str, Dict]:
    return {mode: calibrate(probs_list, labels_list, mode, chunk_ms) for mode in MODES}
