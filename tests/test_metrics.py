"""Pure-numpy tests for the frozen metric definitions (run anywhere)."""

import numpy as np

from clearvad.evaluation.metrics import (
    binary_frame_metrics,
    endpoint_latency_ms,
    frames_to_segments,
    onset_latency_ms,
    probs_to_labels,
    roc_auc,
    summarize,
)


def test_frames_to_segments_basic():
    mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert frames_to_segments(mask) == [(1, 3), (5, 6)]


def test_frames_to_segments_edges():
    assert frames_to_segments(np.array([1, 1], dtype=bool)) == [(0, 2)]
    assert frames_to_segments(np.array([0, 0], dtype=bool)) == []
    assert frames_to_segments(np.array([], dtype=bool)) == []


def test_binary_frame_metrics_known():
    # pred=label for speech 2, plus 1 FP, 1 FN
    label = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    pred = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
    m = binary_frame_metrics(pred, label)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 2
    # recall = 2/3, precision = 2/3, f1 = 2/3
    assert abs(m["recall"] - 2 / 3) < 1e-4
    assert abs(m["precision"] - 2 / 3) < 1e-4
    assert abs(m["f1"] - 2 / 3) < 1e-4
    # FAR = FP/(FP+TN) = 1/3 ; MR = FN/(FN+TP) = 1/3
    assert abs(m["far"] - 1 / 3) < 1e-4
    assert abs(m["mr"] - 1 / 3) < 1e-4


def test_roc_auc_perfect_and_reversed():
    label = np.array([0, 0, 1, 1], dtype=bool)
    assert abs(roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), label) - 1.0) < 1e-9
    assert abs(roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), label) - 0.0) < 1e-9
    # ties -> 0.5
    assert abs(roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), label) - 0.5) < 1e-9


def test_probs_to_labels():
    p = np.array([0.49, 0.5, 0.51])
    np.testing.assert_array_equal(probs_to_labels(p, 0.5), np.array([False, True, True]))


def test_onset_latency():
    # true speech in frames [2,6); model fires at frame 4 -> 2 frames late
    pred = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    seg = [(2, 6)]
    out = onset_latency_ms(pred, seg, chunk_ms=32.0)
    assert abs(out["onset_latency_mean_ms"] - 2 * 32.0) < 1e-6
    assert out["onset_detection_rate"] == 1.0


def test_endpoint_latency_late_release():
    # true speech [0,3); model keeps firing until frame 5 -> releases at 5 (2 late)
    pred = np.array([1, 1, 1, 1, 1, 0, 0], dtype=bool)
    seg = [(0, 3)]
    out = endpoint_latency_ms(pred, seg, chunk_ms=32.0)
    assert abs(out["endpoint_latency_mean_ms"] - 2 * 32.0) < 1e-6


def test_summarize_keys():
    rng = np.random.default_rng(0)
    label = np.array([0] * 10 + [1] * 10 + [0] * 10, dtype=bool)
    probs = np.where(label, 0.8, 0.2).astype(np.float32) + rng.normal(0, 0.01, label.shape).astype(np.float32)
    s = summarize(probs, label, threshold=0.5)
    for k in ("f1", "far", "mr", "auc", "onset_latency_mean_ms", "endpoint_latency_mean_ms"):
        assert k in s
    assert s["f1"] > 0.9
