"""Phase 6 tests — hysteresis smoother (pure numpy; runs anywhere)."""

import numpy as np

from clearvad.postprocess.calibrate import calibrate, evaluate_params
from clearvad.postprocess.smoother import HysteresisSmoother, frames_for_ms


def test_hysteresis_holds_through_dip():
    # prob dips below offset briefly but not long enough to end (min_silence holdoff)
    sm = HysteresisSmoother(onset_threshold=0.5, offset_threshold=0.35,
                            min_speech_ms=0, min_silence_ms=100, speech_pad_ms=0, chunk_ms=32)
    probs = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9])  # 2-frame dip (~64ms < 100ms holdoff)
    mask = sm.process(probs)
    assert mask.tolist() == [True, True, True, True, True, True]


def test_ends_after_min_silence():
    sm = HysteresisSmoother(onset_threshold=0.5, offset_threshold=0.35,
                            min_speech_ms=0, min_silence_ms=32, speech_pad_ms=0, chunk_ms=32)
    probs = np.array([0.9, 0.1, 0.1, 0.1])  # 1-frame holdoff -> ends on the 2nd low frame
    mask = sm.process(probs)
    assert mask[0] and not mask[-1]


def test_onset_offset_asymmetry():
    # value between offset and onset should NOT start speech, but sustains it once started
    sm = HysteresisSmoother(onset_threshold=0.6, offset_threshold=0.3,
                            min_speech_ms=0, min_silence_ms=0, speech_pad_ms=0)
    assert sm.step(0.45).is_speech is False      # below onset -> no start
    assert sm.step(0.7).is_speech is True        # crosses onset -> start
    assert sm.step(0.45).is_speech is True        # above offset -> sustain
    assert sm.step(0.2).is_speech is False        # below offset -> end (min_silence=0)


def test_streaming_equals_chunked():
    """Feeding probs all at once == feeding in chunks while carrying state."""
    rng = np.random.default_rng(0)
    probs = rng.random(200).astype(np.float32)
    sm = HysteresisSmoother()
    full = sm.process(probs, reset=True)

    sm.reset()
    parts = []
    for i in range(0, len(probs), 37):  # arbitrary chunk boundaries
        parts.extend(sm.step(float(p)).is_speech for p in probs[i:i + 37])
    assert np.array_equal(full, np.array(parts, dtype=bool))


def test_segments_drops_short_speech():
    sm = HysteresisSmoother(onset_threshold=0.5, offset_threshold=0.5,
                            min_speech_ms=100, min_silence_ms=0, speech_pad_ms=0, chunk_ms=32)
    # one 1-frame blip (32ms < 100ms) then a long speech run
    probs = np.array([0.9, 0.1, 0.1] + [0.9] * 6)
    segs = sm.segments(probs)
    assert all((e - s) >= frames_for_ms(100, 32) for (s, e) in segs)
    assert len(segs) == 1


def test_speech_pad_extends_end():
    short = HysteresisSmoother(onset_threshold=0.5, offset_threshold=0.5,
                               min_speech_ms=0, min_silence_ms=32, speech_pad_ms=0, chunk_ms=32)
    padded = HysteresisSmoother(onset_threshold=0.5, offset_threshold=0.5,
                                min_speech_ms=0, min_silence_ms=32, speech_pad_ms=96, chunk_ms=32)
    probs = np.array([0.9, 0.9] + [0.1] * 8)
    assert int(padded.process(probs).sum()) > int(short.process(probs).sum())


def test_calibrate_runs():
    rng = np.random.default_rng(1)
    # 4 sequences: speech block then silence block
    probs_list, labels_list = [], []
    for _ in range(4):
        lab = np.array([1] * 20 + [0] * 20, dtype=bool)
        p = np.where(lab, 0.85, 0.15).astype(np.float32) + rng.normal(0, 0.05, lab.shape).astype(np.float32)
        probs_list.append(p.clip(0, 1)); labels_list.append(lab)
    best = calibrate(probs_list, labels_list, mode="balanced")
    assert "params" in best and "metrics" in best
    assert 0.0 <= best["metrics"]["f1"] <= 1.0
