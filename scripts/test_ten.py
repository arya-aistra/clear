#!/usr/bin/env python
"""TEN VAD teacher gate — validate BEFORE training, including TEMPORAL ALIGNMENT.

The Pyannote teacher passed an aggregate gate but its per-frame labels were time-misaligned,
which silently broke training. This gate adds a cross-correlation LAG check between the TEN
teacher and Silero on a transition clip: if the best-correlation lag != 0, the labels are
shifted and would poison distillation. Also checks TEN fires on speech / quiet on noise.

Run:
    python scripts/test_ten.py --speech-wav harvard.wav
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

LOG = get_logger("ten-gate")


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 6):
    """Return (best_lag, corr_at_best, corr_at_0) for per-frame series a vs b."""
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
    ap.add_argument("--hop-size", type=int, default=256)
    ap.add_argument("--out", default="reports/phase_ten/ten_gate.json")
    args = ap.parse_args()

    import torch
    from clearvad.distill.ten_teacher import TenVadTeacher

    LOG.info("Loading TEN VAD teacher (hop=%d)...", args.hop_size)
    ten = TenVadTeacher(hop_size=args.hop_size)
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
        tp = ten.label(t)[0].numpy()
        K = min(len(sp), len(tp))
        lag, corr_best, corr0 = best_lag(tp[:K], sp[:K])
        report[name] = {
            "n_chunks": int(K),
            "silero_speech_frac": round(float((sp[:K] > 0.5).mean()), 4),
            "ten_speech_frac": round(float((tp[:K] > 0.5).mean()), 4),
            "ten_mean": round(float(tp[:K].mean()), 4),
            "ten_max": round(float(tp[:K].max()), 4),
            "best_lag_vs_silero": int(lag),
            "corr_at_best_lag": corr_best,
            "corr_at_lag0": corr0,
        }
        LOG.info("%-16s K=%d ten_speech=%.3f silero_speech=%.3f | best_lag=%d corr@lag0=%.2f",
                 name, K, report[name]["ten_speech_frac"], report[name]["silero_speech_frac"],
                 lag, corr0)

    tr = report.get("synth_transition", {})
    verdict = {
        "ten_fires_on_speech": report.get("real_speech", {}).get("ten_speech_frac", 0) > 0.5,
        "ten_quiet_on_noise": report.get("synth_noise", {}).get("ten_speech_frac", 1) < 0.2,
        "temporally_aligned": (tr.get("best_lag_vs_silero", 99) == 0 and tr.get("corr_at_lag0", 0) > 0.5),
    }
    write_json({"per_clip": report, "verdict": verdict}, args.out)
    LOG.info("Verdict: %s", verdict)
    if not verdict["temporally_aligned"]:
        LOG.warning("TEN labels are TIME-MISALIGNED vs Silero (best_lag != 0 or weak corr). "
                    "Fix alignment before training — this is what silently broke Pyannote.")


if __name__ == "__main__":
    main()
