"""Frozen VAD metric definitions.

These definitions are FIXED for the whole project. Every model (Silero baseline,
ClearVAD FP32/FP16/INT8, WebRTC) is scored with this exact code so comparisons are
fair (GSD benchmark discipline). All frame-level metrics operate at 32 ms resolution.

Definitions
-----------
F1     : harmonic mean of precision & recall on the *speech* class.
FAR    : False Alarm Rate = FP / (FP + TN)  — fraction of non-speech frames flagged speech.
MR     : Miss Rate        = FN / (FN + TP)  — fraction of speech frames missed.
AUC    : threshold-free ranking quality (Mann-Whitney U / probabilistic interpretation).
Onset latency    : ms from true speech onset to first predicted speech frame.
Endpoint latency : ms from true speech offset to predicted offset (speech->silence).
RTF    : real-time factor = compute_time / audio_duration (lower is better; <1 = realtime).

Latency conventions: positive = the model reacts *late* (the usual case). Negative =
the model reacts *early* (e.g. predicts the endpoint before the true offset). We report
raw signed values and never clip, so endpoint improvements are auditable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from clearvad import CHUNK_MS

Segment = Tuple[int, int]  # [start_frame, end_frame)  half-open


# --------------------------------------------------------------------------- core
def probs_to_labels(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold probabilities into a boolean speech mask."""
    return np.asarray(probs, dtype=np.float32) >= float(threshold)


def binary_frame_metrics(pred: np.ndarray, label: np.ndarray,
                         eps: float = 1e-12) -> Dict[str, float]:
    """Compute confusion-matrix-derived frame metrics. Inputs are boolean masks."""
    pred = np.asarray(pred).astype(bool).reshape(-1)
    label = np.asarray(label).astype(bool).reshape(-1)
    if pred.shape != label.shape:
        raise ValueError(f"pred/label length mismatch: {pred.shape} vs {label.shape}")

    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    fn = int(np.sum(~pred & label))
    tn = int(np.sum(~pred & ~label))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)          # = 1 - MR
    f1 = 2 * precision * recall / (precision + recall + eps)
    far = fp / (fp + tn + eps)
    mr = fn / (fn + tp + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "far": round(far, 6),
        "mr": round(mr, 6),
        "accuracy": round(accuracy, 6),
    }


def roc_auc(probs: np.ndarray, label: np.ndarray) -> float:
    """Threshold-free AUC via the Mann-Whitney U statistic (no sklearn dependency)."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    label = np.asarray(label).astype(bool).reshape(-1)
    pos = probs[label]
    neg = probs[~label]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    # average ranks for ties
    _assign_tie_ranks(probs, ranks)
    sum_ranks_pos = ranks[label].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


# --------------------------------------------------------------------- segments
def frames_to_segments(mask: np.ndarray) -> List[Segment]:
    """Convert a boolean frame mask to a list of half-open [start, end) speech segments."""
    mask = np.asarray(mask).astype(bool).reshape(-1)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


# --------------------------------------------------------------------- latencies
def onset_latency_ms(pred_mask: np.ndarray, true_segments: Sequence[Segment],
                     chunk_ms: float = CHUNK_MS) -> Dict[str, float]:
    """Per-true-segment onset latency: first predicted-speech frame at/after true onset.

    Search window for each segment runs from its onset up to its own offset (the model
    must fire *during* the segment to count as detected). Returns mean/median over
    detected segments plus a detection rate.
    """
    pred_mask = np.asarray(pred_mask).astype(bool).reshape(-1)
    latencies: List[float] = []
    detected = 0
    for (s, e) in true_segments:
        window = pred_mask[s:e]
        hits = np.where(window)[0]
        if hits.size:
            detected += 1
            latencies.append(hits[0] * chunk_ms)
    return _latency_summary(latencies, detected, len(true_segments), "onset")


def endpoint_latency_ms(pred_mask: np.ndarray, true_segments: Sequence[Segment],
                        chunk_ms: float = CHUNK_MS,
                        search_frames: Optional[int] = None) -> Dict[str, float]:
    """Per-true-segment endpoint latency: time from true offset to predicted offset.

    For each true segment ending at frame ``e``, find the first frame >= e where the
    prediction is silence (speech has ended). ``search_frames`` bounds how far past the
    offset we look (default: until the next true onset or end of signal). Signed:
    positive = model lingers (late endpoint); negative = model ends early.
    """
    pred_mask = np.asarray(pred_mask).astype(bool).reshape(-1)
    n = len(pred_mask)
    latencies: List[float] = []
    detected = 0
    segs = list(true_segments)
    for idx, (s, e) in enumerate(segs):
        # bound the search at the next segment's onset (don't bleed into next utterance)
        next_onset = segs[idx + 1][0] if idx + 1 < len(segs) else n
        limit = min(next_onset, e + search_frames) if search_frames else next_onset
        limit = max(limit, e)
        # first silence frame at/after the true offset
        sil = None
        for f in range(min(e, n), min(limit, n)):
            if not pred_mask[f]:
                sil = f
                break
        if sil is None:
            # model never released within window -> treat as undetected endpoint
            continue
        detected += 1
        # offset is between frame e-1 (last true speech) and e; predicted release at `sil`
        latencies.append((sil - e) * chunk_ms)
    return _latency_summary(latencies, detected, len(segs), "endpoint")


def _latency_summary(latencies: List[float], detected: int, total: int,
                     name: str) -> Dict[str, float]:
    if latencies:
        arr = np.asarray(latencies, dtype=np.float64)
        out = {
            f"{name}_latency_mean_ms": round(float(arr.mean()), 3),
            f"{name}_latency_median_ms": round(float(np.median(arr)), 3),
            f"{name}_latency_p90_ms": round(float(np.percentile(arr, 90)), 3),
        }
    else:
        out = {
            f"{name}_latency_mean_ms": float("nan"),
            f"{name}_latency_median_ms": float("nan"),
            f"{name}_latency_p90_ms": float("nan"),
        }
    out[f"{name}_detection_rate"] = round(detected / total, 4) if total else float("nan")
    out[f"{name}_n_segments"] = total
    return out


# --------------------------------------------------------------------- aggregate
def summarize(probs: np.ndarray, label_mask: np.ndarray, threshold: float = 0.5,
              chunk_ms: float = CHUNK_MS) -> Dict[str, float]:
    """One-call summary: frame metrics + AUC + onset/endpoint latency at a threshold."""
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    label_mask = np.asarray(label_mask).astype(bool).reshape(-1)
    pred = probs_to_labels(probs, threshold)
    true_segments = frames_to_segments(label_mask)
    out: Dict[str, float] = {"threshold": threshold}
    out.update(binary_frame_metrics(pred, label_mask))
    out["auc"] = round(roc_auc(probs, label_mask), 6)
    out.update(onset_latency_ms(pred, true_segments, chunk_ms))
    out.update(endpoint_latency_ms(pred, true_segments, chunk_ms))
    return out
