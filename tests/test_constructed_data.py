"""Tests for the constructed ground-truth-labeled training data (accuracy track)."""

import numpy as np
import pytest

from clearvad import CHUNK_SAMPLES
from clearvad.distill.synthetic_gen import SyntheticAudioGenerator


def test_construct_clip_has_speech_and_silence():
    from clearvad.distill.constructed_data import construct_clip
    gen = SyntheticAudioGenerator()
    rng = np.random.default_rng(0)
    n = 64 * CHUNK_SAMPLES
    audio, labels = construct_clip(rng.standard_normal(16000 * 60).astype(np.float32),
                                   n, rng, gen)
    K = n // CHUNK_SAMPLES
    assert audio.shape[0] == K * CHUNK_SAMPLES
    assert labels.shape == (K,)
    # a constructed clip must contain BOTH speech and silence (the whole point)
    assert labels.max() == 1.0 and labels.min() == 0.0
    assert 0.1 < labels.mean() < 0.95


def test_construct_clip_deterministic():
    from clearvad.distill.constructed_data import construct_clip
    gen = SyntheticAudioGenerator()
    buf = np.random.default_rng(1).standard_normal(16000 * 30).astype(np.float32)
    a1, l1 = construct_clip(buf, 16 * CHUNK_SAMPLES, np.random.default_rng(7), gen)
    a2, l2 = construct_clip(buf, 16 * CHUNK_SAMPLES, np.random.default_rng(7), gen)
    np.testing.assert_array_equal(l1, l2)


@pytest.mark.requires_torch
def test_constructed_pool_shapes():
    pytest.importorskip("torch")
    import torch
    from clearvad.distill.constructed_data import ConstructedDataPool

    class _Src:
        buffer = np.random.default_rng(0).standard_normal(16000 * 120).astype(np.float32)

    pool = ConstructedDataPool(_Src(), SyntheticAudioGenerator(), pool_size=8, clip_chunks=16,
                               teacher=None, true_weight=1.0, label_smooth=0.05)
    info = pool.refresh(seed=0)
    assert "speech_frac" in info
    windows, target = pool.sample(4, np.random.default_rng(0), device="cpu")
    assert windows.shape == (4, 16, 576)
    assert target.shape == (4, 16)
    # label smoothing keeps targets inside (0,1)
    assert float(target.min()) >= 0.0 and float(target.max()) <= 1.0
    hw, hl = pool.holdout(n=6)
    assert hw.shape == (6, 16, 576) and hl.shape == (6, 16)
