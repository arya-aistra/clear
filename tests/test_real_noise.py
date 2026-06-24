"""Tests for the real-noise source + speech-in-noise construction (real-world track)."""

import numpy as np

from clearvad import CHUNK_SAMPLES
from clearvad.distill.synthetic_gen import SyntheticAudioGenerator


class _FakeNoise:
    """Stand-in RealNoiseSource (avoids the 11GB MUSAN download in unit tests)."""
    def __init__(self, seconds=30):
        self.buffer = np.random.default_rng(0).standard_normal(16000 * seconds).astype(np.float32)

    def sample(self, n, rng):
        L = len(self.buffer)
        if L >= n:
            s = int(rng.integers(0, L - n + 1))
            return self.buffer[s:s + n].copy()
        reps = int(np.ceil(n / max(L, 1)))          # tile when buffer shorter than request
        return np.tile(self.buffer, reps)[:n].copy()


def test_construct_clip_with_real_noise_has_both_classes():
    from clearvad.distill.constructed_data import construct_clip
    gen = SyntheticAudioGenerator()
    rng = np.random.default_rng(0)
    buf = np.random.default_rng(1).standard_normal(16000 * 60).astype(np.float32)
    audio, labels = construct_clip(buf, 64 * CHUNK_SAMPLES, rng, gen, noise_source=_FakeNoise())
    assert labels.max() == 1.0 and labels.min() == 0.0      # speech + non-speech present
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 1.05


def test_real_noise_sample_shape_and_tiling():
    fn = _FakeNoise(seconds=1)         # buffer shorter than the request -> must tile
    rng = np.random.default_rng(0)
    out = fn.sample(16000 * 3, rng)
    assert out.shape == (16000 * 3,)


def test_constructed_pool_with_noise_source():
    import pytest
    pytest.importorskip("torch")
    from clearvad.distill.constructed_data import ConstructedDataPool

    class _Src:
        buffer = np.random.default_rng(0).standard_normal(16000 * 120).astype(np.float32)

    pool = ConstructedDataPool(_Src(), SyntheticAudioGenerator(), pool_size=6, clip_chunks=16,
                               noise_source=_FakeNoise(), true_weight=1.0)
    pool.refresh(seed=0)
    windows, target = pool.sample(3, np.random.default_rng(0), device="cpu")
    assert windows.shape == (3, 16, 576) and target.shape == (3, 16)
