"""Pure tests for audio chunking / resampling / mixing (run anywhere)."""

import numpy as np
import pytest

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE
from clearvad.utils.audio import (
    frame_to_ms,
    iter_chunks,
    mix_at_snr,
    ms_to_frame,
    num_chunks,
    rms_normalize,
)


def test_iter_chunks_drop_last():
    wav = np.zeros(CHUNK_SAMPLES * 3 + 10, dtype=np.float32)
    chunks = list(iter_chunks(wav, CHUNK_SAMPLES, drop_last=True))
    assert len(chunks) == 3
    assert all(c.shape == (CHUNK_SAMPLES,) for c in chunks)


def test_iter_chunks_pad_last():
    wav = np.ones(CHUNK_SAMPLES + 10, dtype=np.float32)
    chunks = list(iter_chunks(wav, CHUNK_SAMPLES, drop_last=False))
    assert len(chunks) == 2
    assert chunks[1].shape == (CHUNK_SAMPLES,)
    assert np.all(chunks[1][10:] == 0)  # zero-padded tail


def test_num_chunks():
    assert num_chunks(CHUNK_SAMPLES * 5 + 1) == 5


def test_frame_time_roundtrip():
    for f in range(0, 100):
        assert ms_to_frame(frame_to_ms(f)) == f


def test_rms_normalize_hits_target():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    out = rms_normalize(wav, target_dbfs=-20.0)
    rms = np.sqrt(np.mean(out ** 2))
    dbfs = 20 * np.log10(rms)
    assert abs(dbfs - (-20.0)) < 0.5


def test_mix_at_snr_achieves_snr():
    rng = np.random.default_rng(0)
    speech = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    noise = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    mix, scaled = mix_at_snr(speech, noise, snr_db=10.0)
    sp = np.mean(speech ** 2)
    npw = np.mean(scaled ** 2)
    measured = 10 * np.log10(sp / npw)
    assert abs(measured - 10.0) < 0.5


@pytest.mark.requires_torch
def test_resample_length_ratio():
    pytest.importorskip("scipy")
    from clearvad.utils.audio import resample

    wav = np.zeros(8000, dtype=np.float32)
    out = resample(wav, 8000, 16000)
    assert abs(len(out) - 16000) <= 2
