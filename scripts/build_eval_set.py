#!/usr/bin/env python
"""Build the controlled independent eval set and benchmark Silero on it (the BAR).

Ground truth is from construction (held-out LibriSpeech test-clean + known silence gaps),
NOT from Silero — so it's a fair bar. Reports Silero's frame F1/FAR/MR + onset/endpoint
latency + short-silence detection. Caches the set for later head-to-head vs ClearVAD.

Run:
    python scripts/build_eval_set.py --n-sequences 60 --seq-seconds 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad.distill.real_data import RealSpeechSource  # noqa: E402
from clearvad.evaluation.eval_dataset import build_eval_set, short_silence_detection  # noqa: E402
from clearvad.evaluation.metrics import probs_to_labels, summarize  # noqa: E402
from clearvad.model.silero_compat import SileroVAD  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("evalset")


def aggregate(dicts, keys):
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if k in d and d[k] == d[k]]  # drop NaN
        out[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=60)
    ap.add_argument("--seq-seconds", type=float, default=12.0)
    ap.add_argument("--ls-url", default="test-clean")  # held out from training (dev-clean)
    ap.add_argument("--buffer-seconds", type=float, default=1200.0)
    ap.add_argument("--local-speech-dir", default=None)
    ap.add_argument("--cache", default="data/eval/controlled_eval.npz")
    ap.add_argument("--out", default="reports/phase8/silero_eval_bar.json")
    # HARD real-world eval: mix real noise into speech at low SNR. Use a HELD-OUT noise corpus
    # (different from training) to make the noise-robustness claim airtight (closes Flag 2).
    ap.add_argument("--noise-source", default="none", choices=["none", "musan", "local", "hf"])
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--noise-hf-repo", default="voice-biomarkers/DEMAND-acoustic-noise",
                    help="HF noise dataset for held-out eval (e.g. DEMAND, ESC-50)")
    ap.add_argument("--snr-min", type=float, default=5.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--noise-prob", type=float, default=0.4)
    args = ap.parse_args()

    LOG.info("Building speech buffer from %s ...", args.ls_url)
    src = RealSpeechSource(
        source="local" if args.local_speech_dir else "torchaudio",
        local_dir=args.local_speech_dir, ls_url=args.ls_url,
        buffer_seconds=args.buffer_seconds)

    noise_source = None
    if args.noise_source != "none" or args.noise_dir:
        from clearvad.distill.real_noise import RealNoiseSource
        if args.noise_dir:
            src_kind = "local"
        elif args.noise_source == "hf":
            src_kind = "hf"
        else:
            src_kind = "openslr"
        noise_source = RealNoiseSource(source=src_kind, local_dir=args.noise_dir,
                                       hf_repo=args.noise_hf_repo, buffer_seconds=1200.0)

    LOG.info("Constructing %d eval sequences (%.0fs each, noise=%s, SNR=%.0f-%.0f)...",
             args.n_sequences, args.seq_seconds, args.noise_dir or args.noise_source,
             args.snr_min, args.snr_max)
    seqs = build_eval_set(src.buffer, n_sequences=args.n_sequences, seq_seconds=args.seq_seconds,
                          noise_source=noise_source, noise_prob=args.noise_prob,
                          snr_range=(args.snr_min, args.snr_max))

    # cache audio + labels + gaps
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache,
        audio=np.stack([s.audio for s in seqs]),
        labels=np.stack([s.labels for s in seqs]),
        gaps=json.dumps([s.silence_gaps for s in seqs]),
    )
    speech_frac = float(np.mean([s.labels.mean() for s in seqs]))
    LOG.info("Cached %d sequences to %s  (speech_frac=%.3f)", len(seqs), args.cache, speech_frac)

    LOG.info("Benchmarking Silero (RAW, threshold 0.5) against independent labels...")
    vad = SileroVAD(onnx=False)
    per_seq, all_gap_detections = [], []
    for s in seqs:
        probs = vad.probabilities(s.audio, reset=True)
        K = min(len(probs), len(s.labels))
        m = summarize(probs[:K], s.labels[:K], threshold=0.5)
        per_seq.append(m)
        pred = probs_to_labels(probs[:K], 0.5)
        all_gap_detections.append(short_silence_detection(pred, s.silence_gaps))

    agg = aggregate(per_seq, ["f1", "far", "mr", "auc",
                              "onset_latency_mean_ms", "endpoint_latency_mean_ms"])
    # short-silence detection rate by gap duration (averaged over sequences)
    durs = sorted({d for gd in all_gap_detections for d in gd})
    ss = {}
    for d in durs:
        rates = [gd[d] for gd in all_gap_detections if d in gd]
        ss[f"{int(d)}ms"] = round(float(np.mean(rates)), 4) if rates else float("nan")

    bar = {
        "eval_set": {"n_sequences": len(seqs), "seq_seconds": args.seq_seconds,
                     "speech_frac": round(speech_frac, 4), "ls_url": args.ls_url},
        "silero_raw": agg,
        "silero_raw_short_silence_detection": ss,
        "note": ("This is the BAR: Silero RAW (no post-processing) on independent labels. "
                 "Frame MR is mildly inflated by intra-speech pauses labeled as speech "
                 "(affects all models equally). ClearVAD is measured on the SAME set."),
    }
    write_json(bar, args.out)
    LOG.info("SILERO BAR: F1=%.3f FAR=%.3f MR=%.3f endpoint=%.1fms onset=%.1fms",
             agg["f1"], agg["far"], agg["mr"], agg["endpoint_latency_mean_ms"],
             agg["onset_latency_mean_ms"])
    LOG.info("Silero short-silence detection by duration: %s", ss)
    LOG.info("Wrote %s and cached eval set to %s", args.out, args.cache)


if __name__ == "__main__":
    main()
