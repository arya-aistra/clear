#!/usr/bin/env python
"""NeMo MarbleNet teacher gate — validate before training, incl. TEMPORAL ALIGNMENT.

NeMo's ~20ms frames don't divide our 32ms chunk, so labels are resampled — the same class
of operation that silently broke Pyannote. This gate cross-correlates NeMo vs Silero per-frame
on a transition clip; best-lag must be 0. Also checks fires-on-speech / quiet-on-noise, and
logs NeMo's raw logit shape (so we confirm the orientation handling).

Run:
    python scripts/test_nemo.py --speech-wav harvard.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad import CHUNK_SAMPLES  # noqa: E402
from clearvad.distill.synthetic_gen import SyntheticAudioGenerator  # noqa: E402
from clearvad.distill.teacher import SileroTeacher  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("nemo-gate")


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 6):
    n = min(len(a), len(b))
    a0 = a[:n] - a[:n].mean()
    b0 = b[:n] - b[:n].mean()
    best = (0, -2.0)
    corr0 = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = a0[: n + lag], b0[-lag:]
        elif lag > 0:
            x, y = a0[lag:], b0[: n - lag]
        else:
            x, y = a0, b0
        if len(x) < 3:
            continue
        denom = float(np.std(x) * np.std(y))
        c = float(np.mean(x * y) / denom) if denom > 1e-9 else 0.0
        if lag == 0:
            corr0 = c
        if c > best[1]:
            best = (lag, c)
    return best[0], round(best[1], 3), round(corr0, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-wav", default="harvard.wav")
    ap.add_argument("--out", default="reports/phase_nemo/nemo_gate.json")
    args = ap.parse_args()

    import torch
    from clearvad.distill.nemo_teacher import NeMoMarbleTeacher

    LOG.info("Loading NeMo MarbleNet teacher...")
    nemo = NeMoMarbleTeacher()
    silero = SileroTeacher()
    gen = SyntheticAudioGenerator()

    clips = {}
    if Path(args.speech_wav).exists():
        from clearvad.utils.audio import load_audio
        w = load_audio(args.speech_wav)
        K = len(w) // CHUNK_SAMPLES
        clips["real_speech"] = w[: K * CHUNK_SAMPLES]
    rng = np.random.default_rng(0)
    clips["synth_noise"] = gen.generate("noise", 64 * CHUNK_SAMPLES, rng)
    clips["synth_transition"] = gen.generate("transition", 64 * CHUNK_SAMPLES, rng)

    report = {}
    for name, wav in clips.items():
        t = torch.from_numpy(wav).unsqueeze(0)
        sp = silero.label(t)[0].numpy()
        np_ = nemo.label(t)[0].numpy()
        K = min(len(sp), len(np_))
        lag, corr_best, corr0 = best_lag(np_[:K], sp[:K])
        report[name] = {
            "n_chunks": int(K),
            "silero_speech_frac": round(float((sp[:K] > 0.5).mean()), 4),
            "nemo_speech_frac": round(float((np_[:K] > 0.5).mean()), 4),
            "nemo_mean": round(float(np_[:K].mean()), 4),
            "nemo_max": round(float(np_[:K].max()), 4),
            "best_lag_vs_silero": int(lag),
            "corr_at_lag0": corr0,
        }
        LOG.info("%-16s K=%d nemo_speech=%.3f silero_speech=%.3f | best_lag=%d corr@lag0=%.2f",
                 name, K, report[name]["nemo_speech_frac"], report[name]["silero_speech_frac"],
                 lag, corr0)

    tr = report.get("synth_transition", {})
    verdict = {
        "nemo_fires_on_speech": report.get("real_speech", {}).get("nemo_speech_frac", 0) > 0.5,
        "nemo_quiet_on_noise": report.get("synth_noise", {}).get("nemo_speech_frac", 1) < 0.2,
        "temporally_aligned": (tr.get("best_lag_vs_silero", 99) == 0 and tr.get("corr_at_lag0", 0) > 0.5),
    }
    write_json({"per_clip": report, "verdict": verdict}, args.out)
    LOG.info("Verdict: %s", verdict)
    if not verdict["temporally_aligned"]:
        LOG.warning("NeMo labels TIME-MISALIGNED vs Silero — fix resampling before training.")


if __name__ == "__main__":
    main()
