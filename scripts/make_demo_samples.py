#!/usr/bin/env python
"""Build a couple of short 16 kHz demo wavs for the Benchmark Studio (zero-upload demo).

Prefers REAL speech (LibriSpeech if present): concatenates 2 short utterances with silence gaps so
the VAD visibly toggles speech/silence; makes one clean and one noise-mixed. Falls back to a
synthetic speech-like signal (formant-ish harmonic complex, syllable-rate envelope) if no speech is
found. Writes to clearvad/serving/static/samples/ (committed via a .gitignore exception).

    python scripts/make_demo_samples.py            # real if LibriSpeech present, else synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

SR = 16000
OUT = Path(__file__).resolve().parents[1] / "clearvad" / "serving" / "static" / "samples"


def _sil(sec):
    return np.zeros(int(sec * SR), np.float32)


def _find_flacs(root: str, n: int):
    p = Path(root)
    return sorted(p.rglob("*.flac"))[:n] if p.exists() else []


def _real(n_utts: int):
    import soundfile as sf
    from clearvad.utils.audio import load_audio
    flacs = _find_flacs("data/librispeech/LibriSpeech/test-clean", 8) or _find_flacs("data/librispeech", 8)
    if len(flacs) < 2:
        return None
    rng = np.random.default_rng(0)
    clips = []
    for f in flacs[:n_utts]:
        a = load_audio(f, SR)
        a = a[:int(6 * SR)]                       # cap each utterance to 6 s
        a = a / (np.max(np.abs(a)) + 1e-9) * 0.7
        clips.append(a.astype(np.float32))
    seq = _sil(0.4)
    for i, a in enumerate(clips):
        seq = np.concatenate([seq, a, _sil(0.6 if i < len(clips) - 1 else 0.4)])
    return seq.astype(np.float32)


def _synthetic():
    """A formant-ish, syllable-modulated harmonic complex with silence gaps (no dataset needed)."""
    rng = np.random.default_rng(1)
    def burst(dur, f0=130.0):
        t = np.arange(int(dur * SR)) / SR
        sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 12))
        for fc, bw in [(600, 120), (1200, 180), (2600, 250)]:        # crude formant emphasis
            sig += 0.6 * np.sin(2 * np.pi * fc * t) * np.exp(-((t % 0.18) * bw))
        env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t))                 # ~4 Hz syllable rate
        sig = sig * env + 0.01 * rng.standard_normal(len(t))
        return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.7).astype(np.float32)
    return np.concatenate([_sil(0.4), burst(2.0), _sil(0.6), burst(1.5), _sil(0.4)]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr-db", type=float, default=8.0, help="SNR for the noisy demo clip")
    args = ap.parse_args()
    import soundfile as sf
    OUT.mkdir(parents=True, exist_ok=True)

    base = _real(2)
    kind = "librispeech" if base is not None else "synthetic"
    if base is None:
        base = _synthetic()
    sf.write(OUT / "demo_clean.wav", base, SR)

    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(base)).astype(np.float32)
    sig_p = float(np.mean(base ** 2)) + 1e-9
    noise *= np.sqrt(sig_p / (10 ** (args.snr_db / 10)) / (np.mean(noise ** 2) + 1e-9))
    noisy = base + noise
    noisy = (noisy / (np.max(np.abs(noisy)) + 1e-9) * 0.9).astype(np.float32)
    sf.write(OUT / "demo_noisy.wav", noisy, SR)

    print(f"[{kind}] wrote {OUT/'demo_clean.wav'} and {OUT/'demo_noisy.wav'} "
          f"({len(base)/SR:.1f}s each)")
    if kind == "synthetic":
        print("NOTE: synthetic demo (no LibriSpeech found). For a real-speech demo, run this on the "
              "server where data/librispeech exists, then commit the wavs.")


if __name__ == "__main__":
    main()
