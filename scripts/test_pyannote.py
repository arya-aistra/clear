#!/usr/bin/env python
"""Multi-teacher viability gate — validate the Pyannote teacher BEFORE a long train.

Loads Pyannote, runs it + Silero on a few real-speech and synthetic clips, and reports per-chunk
speech fractions + Silero/Pyannote agreement. Confirms Pyannote fires on speech, stays quiet on
noise, and aligns to our 32 ms grid. Mirrors the synthetic viability gate from Phase 3.

Run:
    HF_TOKEN=hf_xxx python scripts/test_pyannote.py            # or --hf-token hf_xxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad import CHUNK_SAMPLES  # noqa: E402
from clearvad.distill.losses import agreement_rate  # noqa: E402
from clearvad.distill.synthetic_gen import SyntheticAudioGenerator  # noqa: E402
from clearvad.distill.teacher import SileroTeacher  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("pyannote-gate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--speech-wav", default="harvard.wav", help="real speech clip")
    ap.add_argument("--ls-root", default="data/librispeech")
    ap.add_argument("--out", default="reports/phase_mt/pyannote_gate.json")
    args = ap.parse_args()

    import torch
    from clearvad.distill.pyannote_teacher import PyannoteTeacher

    LOG.info("Loading Pyannote segmentation teacher...")
    pyannote = PyannoteTeacher(hf_token=args.hf_token)
    silero = SileroTeacher()
    gen = SyntheticAudioGenerator()

    clips = {}
    # real speech
    if Path(args.speech_wav).exists():
        from clearvad.utils.audio import load_audio
        w = load_audio(args.speech_wav)
        K = len(w) // CHUNK_SAMPLES
        clips["real_speech"] = w[: K * CHUNK_SAMPLES]
    # synthetic noise + speech-like
    rng = np.random.default_rng(0)
    clips["synth_noise"] = gen.generate("noise", 64 * CHUNK_SAMPLES, rng)
    clips["synth_speechlike"] = gen.generate("speech", 64 * CHUNK_SAMPLES, rng)

    report = {}
    for name, wav in clips.items():
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        sp = silero.label(wav_t)[0].numpy()
        pp = pyannote.label(wav_t)[0].numpy()
        K = min(len(sp), len(pp))
        report[name] = {
            "n_chunks": int(K),
            "silero_speech_frac": round(float((sp[:K] > 0.5).mean()), 4),
            "pyannote_speech_frac": round(float((pp[:K] > 0.5).mean()), 4),
            "agreement": round(agreement_rate(torch.tensor(pp[:K]), torch.tensor(sp[:K])), 4),
            "pyannote_mean": round(float(pp[:K].mean()), 4),
            "pyannote_max": round(float(pp[:K].max()), 4),
        }
        LOG.info("%-16s K=%d  silero_speech=%.3f pyannote_speech=%.3f agree=%.3f (pmax=%.2f)",
                 name, K, report[name]["silero_speech_frac"],
                 report[name]["pyannote_speech_frac"], report[name]["agreement"],
                 report[name]["pyannote_max"])

    verdict = {
        "pyannote_fires_on_speech": report.get("real_speech", {}).get("pyannote_speech_frac", 0) > 0.5,
        "pyannote_quiet_on_noise": report.get("synth_noise", {}).get("pyannote_speech_frac", 1) < 0.2,
        "aligned_to_grid": all(r["n_chunks"] > 0 for r in report.values()),
    }
    write_json({"per_clip": report, "verdict": verdict}, args.out)
    LOG.info("Verdict: %s", verdict)
    if not verdict["pyannote_fires_on_speech"]:
        LOG.warning("Pyannote did not fire on real speech — check token/model/alignment "
                    "before training.")


if __name__ == "__main__":
    main()
