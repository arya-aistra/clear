#!/usr/bin/env python
"""Phase 3 — generate synthetic audio and MEASURE teacher activation (DFKD viability check).

For each category, generates clips, labels them with Silero, and reports the teacher's
speech fraction. If categories B/C/D don't make Silero fire (low speech fraction), DFKD
will collapse to "all silence" — this script surfaces that BEFORE training.

Writes reports/phase3/synthetic_activation.json (+ optional example WAVs).

Run:
    python scripts/generate_synthetic.py --clips-per-cat 64 --clip-chunks 64
    python scripts/generate_synthetic.py --save-wavs reports/phase3/samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE  # noqa: E402
from clearvad.distill.synthetic_gen import CATEGORIES, SyntheticAudioGenerator  # noqa: E402
from clearvad.distill.teacher import SileroTeacher  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("synthgen")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-per-cat", type=int, default=64)
    ap.add_argument("--clip-chunks", type=int, default=64)
    ap.add_argument("--save-wavs", default=None, help="dir to dump one example WAV per category")
    ap.add_argument("--out", default="reports/phase3/synthetic_activation.json")
    args = ap.parse_args()

    gen = SyntheticAudioGenerator()
    teacher = SileroTeacher()
    n_samples = args.clip_chunks * CHUNK_SAMPLES

    results = {}
    for cat in CATEGORIES:
        rng = np.random.default_rng(hash(cat) % (2 ** 32))
        audio = np.stack([gen.generate(cat, n_samples, rng) for _ in range(args.clips_per_cat)])
        probs = teacher.label(teacher.torch.from_numpy(audio)).cpu().numpy()  # [N, K]
        speech_frac = float((probs > 0.5).mean())
        clips_with_speech = float(((probs > 0.5).any(axis=1)).mean())
        results[cat] = {
            "mean_speech_frac": round(speech_frac, 4),
            "frac_clips_with_any_speech": round(clips_with_speech, 4),
            "mean_prob": round(float(probs.mean()), 4),
            "max_prob": round(float(probs.max()), 4),
        }
        LOG.info("%-11s speech_frac=%.3f  clips_with_speech=%.3f  max_prob=%.3f",
                 cat, speech_frac, clips_with_speech, results[cat]["max_prob"])

        if args.save_wavs:
            try:
                import soundfile as sf
                d = Path(args.save_wavs); d.mkdir(parents=True, exist_ok=True)
                sf.write(str(d / f"{cat}.wav"), audio[0], SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("save wav failed: %r", exc)

    verdict = {
        "speech_category_fires": results["speech"]["frac_clips_with_any_speech"] >= 0.5,
        "transition_category_fires": results["transition"]["frac_clips_with_any_speech"] >= 0.5,
        "noise_stays_silent": results["noise"]["mean_speech_frac"] <= 0.2,
    }
    payload = {"config": vars(args), "per_category": results, "verdict": verdict}
    write_json(payload, args.out)
    LOG.info("Verdict: %s", verdict)
    LOG.info("Wrote %s", args.out)
    if not (verdict["speech_category_fires"] and verdict["transition_category_fires"]):
        LOG.warning("Synthetic speech-like audio does NOT reliably fire Silero. DFKD may "
                    "collapse to all-silence. Consider richer synthesis or unlabeled real audio.")


if __name__ == "__main__":
    main()
