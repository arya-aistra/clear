"""Evaluation: frozen metric definitions, benchmark + comparison runners."""

from clearvad.evaluation.metrics import (  # noqa: F401
    binary_frame_metrics,
    endpoint_latency_ms,
    frames_to_segments,
    onset_latency_ms,
    probs_to_labels,
    roc_auc,
    summarize,
)

__all__ = [
    "binary_frame_metrics",
    "endpoint_latency_ms",
    "frames_to_segments",
    "onset_latency_ms",
    "probs_to_labels",
    "roc_auc",
    "summarize",
]
