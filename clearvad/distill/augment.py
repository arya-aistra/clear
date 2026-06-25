"""Cheap, dataset-free training augmentation for VAD robustness.

The MUSAN-only model failed to reject unseen noise; part of the fix is exposing the model to
the *transformations* real audio undergoes — reverberation, level changes, codec distortion —
not just more noise corpora. These are applied to constructed training clips (labels unchanged);
they need no extra datasets (synthetic RIR, analytic mu-law). Eval is left untouched.

All numpy, FFT-based convolution so per-clip reverb is fast even on a 2048-clip pool.
"""

from __future__ import annotations

import numpy as np

from clearvad import SAMPLE_RATE


def _fftconvolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    nf = 1 << int(np.ceil(np.log2(max(n, 1))))
    y = np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(h, nf), nf)[:len(x)]
    return y.astype(np.float32)


def synthetic_rir(rng: np.random.Generator, rt60_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A cheap room impulse response: exponentially decaying gaussian noise (RT60-controlled)."""
    length = max(int(rt60_s * sr), 8)
    t = np.arange(length, dtype=np.float32)
    decay = np.exp(-6.908 * t / max(rt60_s * sr, 1.0))   # -60 dB over rt60
    rir = rng.standard_normal(length).astype(np.float32) * decay
    rir[0] = 1.0                                          # direct path
    n = float(np.sqrt(np.sum(rir * rir)) + 1e-9)
    return rir / n


def apply_reverb(audio: np.ndarray, rng: np.random.Generator, rt60_range=(0.1, 0.6),
                 sr: int = SAMPLE_RATE) -> np.ndarray:
    rir = synthetic_rir(rng, float(rng.uniform(*rt60_range)), sr)
    return _fftconvolve_same(audio, rir)


def apply_gain(audio: np.ndarray, rng: np.random.Generator, db_range=(-8.0, 6.0)) -> np.ndarray:
    return (audio * float(10.0 ** (rng.uniform(*db_range) / 20.0))).astype(np.float32)


def apply_mulaw(audio: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """mu-law companding round-trip — telephony/codec distortion (8-bit), realistic for call-centre."""
    x = np.clip(audio, -1.0, 1.0)
    comp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((comp + 1) / 2 * mu) / mu * 2 - 1                  # quantize to ~8-bit
    return (np.sign(q) * (1.0 / mu) * ((1 + mu) ** np.abs(q) - 1)).astype(np.float32)


def augment_clip(audio: np.ndarray, rng: np.random.Generator, p_reverb: float = 0.5,
                 p_codec: float = 0.3, p_gain: float = 0.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Randomly apply reverb / codec / gain to a clip. Labels are unaffected (time-aligned ops)."""
    out = audio
    if rng.random() < p_reverb:
        out = apply_reverb(out, rng, sr=sr)
    if rng.random() < p_codec:
        out = apply_mulaw(out)
    if rng.random() < p_gain:
        out = apply_gain(out, rng)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)
