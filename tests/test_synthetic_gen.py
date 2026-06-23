"""Phase 3 tests for the synthetic audio generator (numpy-only, run anywhere)."""

import numpy as np

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE
from clearvad.distill.synthetic_gen import CATEGORIES, DEFAULT_WEIGHTS, SyntheticAudioGenerator


def _gen():
    return SyntheticAudioGenerator(SAMPLE_RATE)


def test_generate_shapes_and_range():
    g = _gen()
    rng = np.random.default_rng(0)
    n = 16000
    for cat in CATEGORIES:
        x = g.generate(cat, n, rng)
        assert x.shape == (n,)
        assert x.dtype == np.float32
        assert np.all(np.abs(x) <= 1.0 + 1e-5), f"{cat} out of range"
        assert np.isfinite(x).all()


def test_generate_exact_length_pad_truncate():
    g = _gen()
    rng = np.random.default_rng(1)
    for n in (512, 5000, 32768):
        assert g.generate("transition", n, rng).shape == (n,)


def test_reproducibility():
    g = _gen()
    a, ca = g.generate_batch(8, 8000, seed=42)
    b, cb = g.generate_batch(8, 8000, seed=42)
    np.testing.assert_array_equal(a, b)
    assert ca == cb


def test_batch_categories_valid():
    g = _gen()
    audio, cats = g.generate_batch(32, 8000, seed=7)
    assert audio.shape == (32, 8000)
    assert all(c in CATEGORIES for c in cats)


def test_noise_vs_speech_differ():
    g = _gen()
    rng = np.random.default_rng(3)
    noise = g.generate("noise", 16000, rng)
    speech = g.generate("speech", 16000, rng)
    # speech-like should have stronger low-frequency periodicity than white-ish noise;
    # at minimum they must not be identical
    assert not np.allclose(noise, speech)


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-6


def test_clip_chunks_alignment():
    g = _gen()
    rng = np.random.default_rng(5)
    K = 64
    x = g.generate("transition", K * CHUNK_SAMPLES, rng)
    assert len(x) == K * CHUNK_SAMPLES
