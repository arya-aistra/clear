"""Synthetic audio generator for data-free distillation (numpy, 16 kHz).

Four categories (per the GSD Phase 3 spec). All clips are mono float32 in [-1, 1]. The
TEACHER (Silero) provides the labels, so these signals only need to be *diverse* and to
elicit varied teacher responses — especially crisp speech↔silence transitions.

  A  noise      — white / pink / brown / colored (AC/fan/traffic-like) noise textures
  B  speech     — voiced harmonic excitation + formant shaping + syllabic AM ("speech-like")
  C  transition — silence↔speech-like segments (trains onset/endpoint behavior)
  D  hardneg    — music-like harmonic stacks, short bursts (click/breath), TV-like speech+bg

IMPORTANT (honest-reframe risk): category B/C/D must make Silero actually fire as "speech",
or the student just learns "all silence". `scripts/generate_synthetic.py` and the smoke test
MEASURE teacher activation on these and report it; if activation is low we revisit synthesis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from clearvad import SAMPLE_RATE

CATEGORIES = ("noise", "speech", "transition", "hardneg")
DEFAULT_WEIGHTS = {"noise": 0.30, "speech": 0.30, "transition": 0.25, "hardneg": 0.15}


def _bandpass(x: np.ndarray, lo_hz: float, hi_hz: float, sr: int, order: int = 2) -> np.ndarray:
    try:
        from scipy.signal import butter, lfilter

        nyq = sr / 2
        lo, hi = max(lo_hz, 20) / nyq, min(hi_hz, nyq - 100) / nyq
        b, a = butter(order, [lo, hi], btype="band")
        return lfilter(b, a, x).astype(np.float32)
    except Exception:  # noqa: BLE001 - scipy missing or filter edge case
        return x.astype(np.float32)


def _lowpass(x: np.ndarray, hz: float, sr: int, order: int = 2) -> np.ndarray:
    try:
        from scipy.signal import butter, lfilter

        b, a = butter(order, min(hz, sr / 2 - 100) / (sr / 2), btype="low")
        return lfilter(b, a, x).astype(np.float32)
    except Exception:  # noqa: BLE001
        return x.astype(np.float32)


def _norm(x: np.ndarray, peak: float = 0.7) -> np.ndarray:
    m = float(np.max(np.abs(x))) + 1e-9
    return (x / m * peak).astype(np.float32)


class SyntheticAudioGenerator:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sr = sample_rate

    # ---------------------------------------------------------------- textures
    def white(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(n).astype(np.float32)

    def pink(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Voss-McCartney-ish via 1/f filtering in freq domain
        w = rng.standard_normal(n)
        f = np.fft.rfftfreq(n, d=1.0 / self.sr)
        f[0] = f[1] if len(f) > 1 else 1.0
        spec = np.fft.rfft(w) / np.sqrt(f)
        return np.fft.irfft(spec, n=n).astype(np.float32)

    def brown(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x = np.cumsum(rng.standard_normal(n))
        return (x - x.mean()).astype(np.float32)

    def colored_env(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """AC/fan/traffic-like: low-frequency-dominant filtered noise + slow AM."""
        base = rng.standard_normal(n)
        cutoff = rng.uniform(200, 1200)
        x = _lowpass(base, cutoff, self.sr)
        t = np.arange(n) / self.sr
        am = 0.7 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.2, 2.0) * t)
        return (x * am).astype(np.float32)

    def noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        kind = rng.choice(["white", "pink", "brown", "colored"])
        x = {"white": self.white, "pink": self.pink,
             "brown": self.brown, "colored": self.colored_env}[kind](n, rng)
        return _norm(x, rng.uniform(0.2, 0.7))

    # ---------------------------------------------------------------- speech-like
    def speech_envelope(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Voiced excitation (harmonic stack, F0 80–300 Hz w/ jitter) + formants + syllabic AM."""
        t = np.arange(n) / self.sr
        f0 = rng.uniform(90, 240)
        # F0 drift + jitter
        drift = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))
        jitter = 1.0 + 0.01 * np.cumsum(rng.standard_normal(n)) / max(n, 1)
        phase = 2 * np.pi * np.cumsum(drift * jitter) / self.sr
        sig = np.zeros(n)
        n_harm = int(rng.integers(15, 35))
        for k in range(1, n_harm + 1):
            sig += (1.0 / k) * np.sin(k * phase)
        # formant shaping (3 random formants in typical vowel ranges)
        f1 = rng.uniform(300, 800); f2 = rng.uniform(900, 2200); f3 = rng.uniform(2400, 3400)
        shaped = (1.0 * _bandpass(sig, f1 - 150, f1 + 150, self.sr)
                  + 0.7 * _bandpass(sig, f2 - 200, f2 + 200, self.sr)
                  + 0.4 * _bandpass(sig, f3 - 250, f3 + 250, self.sr))
        # syllabic amplitude modulation (3–7 Hz) + voicing gaps
        syl = rng.uniform(3, 7)
        am = 0.5 * (1.0 + np.sin(2 * np.pi * syl * t + rng.uniform(0, 6.28)))
        am = np.clip(am, 0.05, 1.0)
        shaped = shaped * (0.3 + 0.7 * am)
        shaped += 0.02 * rng.standard_normal(n)  # breath noise
        return _norm(shaped, rng.uniform(0.4, 0.8))

    # ---------------------------------------------------------------- transitions
    def transition(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Concatenate silence / speech-like segments with sharp boundaries."""
        out = np.zeros(n, dtype=np.float32)
        pos = 0
        while pos < n:
            is_speech = rng.random() < 0.55
            seg_ms = rng.uniform(80, 700) if is_speech else rng.uniform(40, 400)
            seg = int(seg_ms / 1000 * self.sr)
            seg = min(seg, n - pos)
            if is_speech and seg > 0:
                out[pos:pos + seg] = self.speech_envelope(seg, rng)
            pos += seg
        return out

    # ---------------------------------------------------------------- hard negatives
    def music(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Steady harmonic stack (fixed pitch set) + rhythmic AM — music-like, not speech."""
        t = np.arange(n) / self.sr
        root = rng.uniform(110, 330)
        sig = np.zeros(n)
        for mult in (1.0, 1.5, 2.0, 2.5, 3.0):  # chord-ish, near-stationary pitch
            sig += rng.uniform(0.4, 1.0) * np.sin(2 * np.pi * root * mult * t)
        beat = rng.uniform(1.5, 4.0)  # rhythmic pulse
        env = 0.5 * (1 + np.sign(np.sin(2 * np.pi * beat * t)) * 0.6)
        return _norm(sig * env, rng.uniform(0.3, 0.7))

    def burst(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Short transient bursts: clicks / breaths / throat-clear (mostly silence)."""
        out = np.zeros(n, dtype=np.float32)
        n_bursts = int(rng.integers(1, 5))
        for _ in range(n_bursts):
            dur = int(rng.uniform(0.01, 0.12) * self.sr)
            start = int(rng.uniform(0, max(n - dur, 1)))
            b = rng.standard_normal(dur)
            b = _lowpass(b, rng.uniform(500, 4000), self.sr) * np.hanning(dur)
            out[start:start + dur] += b.astype(np.float32)
        return _norm(out, rng.uniform(0.2, 0.6))

    def tv(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """TV-like: speech-envelope mixed with background noise at moderate SNR."""
        speech = self.speech_envelope(n, rng)
        bg = self.colored_env(n, rng)
        snr = rng.uniform(0, 15)
        sp = np.mean(speech ** 2) + 1e-9
        npw = np.mean(bg ** 2) + 1e-9
        scale = np.sqrt(sp / (10 ** (snr / 10)) / npw)
        return _norm(speech + scale * bg, 0.7)

    def hardneg(self, n: int, rng: np.random.Generator) -> np.ndarray:
        kind = rng.choice(["music", "burst", "tv"])
        return {"music": self.music, "burst": self.burst, "tv": self.tv}[kind](n, rng)

    # ---------------------------------------------------------------- assembly
    def generate(self, category: str, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        fn = {"noise": self.noise, "speech": self.speech_envelope,
              "transition": self.transition, "hardneg": self.hardneg}[category]
        x = fn(n_samples, rng)
        if len(x) < n_samples:
            x = np.pad(x, (0, n_samples - len(x)))
        return x[:n_samples].astype(np.float32)

    def generate_batch(
        self,
        batch_size: int,
        n_samples: int,
        seed: int,
        category_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Return (audio[batch, n_samples] float32, categories[batch])."""
        rng = np.random.default_rng(seed)
        w = category_weights or DEFAULT_WEIGHTS
        cats = list(w.keys())
        probs = np.array([w[c] for c in cats], dtype=np.float64)
        probs = probs / probs.sum()
        out = np.zeros((batch_size, n_samples), dtype=np.float32)
        chosen: List[str] = []
        for i in range(batch_size):
            c = cats[int(rng.choice(len(cats), p=probs))]
            out[i] = self.generate(c, n_samples, rng)
            chosen.append(c)
        return out, chosen
