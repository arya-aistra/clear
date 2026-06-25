#!/usr/bin/env python
"""Phase 8 / Flag-1 closer — build a FRAME-ACCURATE eval via forced alignment.

Our other evals use segment-level labels (a speech segment = all-speech, incl. intra-utterance
pauses), which biases F1 toward a model trained to that convention and penalizes a granular VAD
like Silero. This eval removes that bias: we **force-align** LibriSpeech transcripts to audio
(torchaudio MMS_FA) to get true word boundaries, then label inter-word / leading / trailing
silence as NON-speech. Evaluating both models against these frame-accurate labels is the honest
test of whether ClearVAD's accuracy win is real or just convention-alignment.

Caches `data/eval/aligned_eval.npz` (audio/labels/gaps) for `compare_models.py`.

Run (after LibriSpeech test-clean is downloaded by an earlier build_eval_set):
    python scripts/build_aligned_eval.py --max-utts 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE  # noqa: E402
from clearvad.distill.forced_align import ForcedAligner  # noqa: E402
from clearvad.utils.audio import load_audio  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402

LOG = get_logger("aligned-eval")


def find_utterances(ls_dir: Path, max_utts: int):
    """Yield (flac_path, transcript) from a LibriSpeech split dir."""
    items = []
    for trans in sorted(ls_dir.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.partition(" ")
            flac = trans.parent / f"{uid}.flac"
            if flac.exists() and text.strip():
                items.append((flac, text.strip()))
            if len(items) >= max_utts:
                return items
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-dir", default="data/librispeech/LibriSpeech/test-clean")
    ap.add_argument("--max-utts", type=int, default=300)
    ap.add_argument("--utts-per-seq", type=int, default=4)
    ap.add_argument("--gap-ms", type=float, default=400.0, help="true silence between utterances")
    ap.add_argument("--min-silence-ms", type=float, default=0.0,
                    help="fill interior gaps shorter than this (0 = strict/raw eval). "
                         "Match the training value for an apples-to-apples comparison.")
    ap.add_argument("--pad-ms", type=float, default=0.0,
                    help="extend each word span by this each side (match training for fairness). "
                         "Corrects MMS_FA trimming speech edges; applied identically to all models.")
    ap.add_argument("--cache", default="data/eval/aligned_eval.npz")
    ap.add_argument("--out", default="reports/phase8/aligned_eval_info.json")
    ap.add_argument("--device", default="cuda")
    # HARD + frame-accurate: mix held-out real noise (e.g. DEMAND) across the whole sequence at a
    # random SNR (speech-in-noise + noisy silence). Labels are unchanged. Tests robustness/general-
    # ization — where a small model can beat Silero, vs clean LibriSpeech (Silero's home turf).
    ap.add_argument("--noise-source", default="none", choices=["none", "musan", "local", "hf"])
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--noise-hf-repo", default="voice-biomarkers/DEMAND-acoustic-noise")
    ap.add_argument("--snr-min", type=float, default=0.0)
    ap.add_argument("--snr-max", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    ls_dir = Path(args.ls_dir)
    if not ls_dir.exists():
        raise SystemExit(f"{ls_dir} not found — run build_eval_set.py --ls-url test-clean first "
                         "to download/extract LibriSpeech.")

    aligner = ForcedAligner(device=args.device)
    utts = find_utterances(ls_dir, args.max_utts)
    LOG.info("Aligning %d utterances (min_silence=%.0fms, pad=%.0fms)...", len(utts),
             args.min_silence_ms, args.pad_ms)

    noise_source, noise_rng = None, np.random.default_rng(args.seed)
    if args.noise_source != "none" or args.noise_dir:
        from clearvad.distill.real_noise import RealNoiseSource
        kind = "local" if args.noise_dir else ("hf" if args.noise_source == "hf" else "openslr")
        noise_source = RealNoiseSource(source=kind, local_dir=args.noise_dir,
                                       hf_repo=args.noise_hf_repo, buffer_seconds=1200.0)
        LOG.info("Mixing held-out noise (%s) at SNR %.0f-%.0f dB into eval sequences",
                 args.noise_dir or args.noise_source, args.snr_min, args.snr_max)

    def utt_labels(flac: Path, text: str):
        """Return (audio[L], sample_speech_mask[L]) with frame-accurate speech from alignment."""
        wav = load_audio(flac, SAMPLE_RATE)
        mask = aligner.speech_mask(wav, text, min_silence_ms=args.min_silence_ms,
                                   pad_ms=args.pad_ms, sr=SAMPLE_RATE)
        if mask is None:
            return None
        return wav, mask

    # build sequences by concatenating utterances with true-silence gaps
    gap_samples = int(args.gap_ms / 1000 * SAMPLE_RATE)
    seqs_audio, seqs_labels, seqs_gaps = [], [], []
    buf_a, buf_m = [], []
    n_done = 0
    for flac, text in utts:
        try:
            r = utt_labels(flac, text)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("align failed %s: %r", flac.name, exc)
            continue
        if r is None:
            continue
        wav, mask = r
        buf_a.append(wav); buf_m.append(mask)
        buf_a.append(np.zeros(gap_samples, np.float32)); buf_m.append(np.zeros(gap_samples, np.float32))
        n_done += 1
        if len(buf_a) // 2 >= args.utts_per_seq:
            audio = np.concatenate(buf_a); smask = np.concatenate(buf_m)
            K = len(audio) // CHUNK_SAMPLES
            audio = audio[:K * CHUNK_SAMPLES]
            if noise_source is not None:                 # speech-in-noise + noisy silence (hard)
                from clearvad.utils.audio import mix_at_snr
                noise = noise_source.sample(len(audio), noise_rng)
                audio, _ = mix_at_snr(audio, noise, float(noise_rng.uniform(args.snr_min, args.snr_max)))
                audio = audio.astype(np.float32)
            labels = np.array([1.0 if smask[k * CHUNK_SAMPLES:(k + 1) * CHUNK_SAMPLES].mean() >= 0.5
                               else 0.0 for k in range(K)], dtype=np.float32)
            # record true-silence gaps (runs of 0) for the short-silence metric
            gaps = []
            k = 0
            while k < K:
                if labels[k] == 0:
                    j = k
                    while j < K and labels[j] == 0:
                        j += 1
                    gaps.append((k, j, round((j - k) * CHUNK_SAMPLES / SAMPLE_RATE * 1000, 1)))
                    k = j
                else:
                    k += 1
            seqs_audio.append(audio); seqs_labels.append(labels); seqs_gaps.append(gaps)
            buf_a, buf_m = [], []

    if not seqs_audio:
        raise SystemExit("No aligned sequences produced — check MMS_FA / transcripts.")

    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache,
        audio=np.array(seqs_audio, dtype=object),
        labels=np.array(seqs_labels, dtype=object),
        gaps=json.dumps([[list(g) for g in gs] for gs in seqs_gaps]),
    )
    speech_frac = float(np.mean([l.mean() for l in seqs_labels]))
    write_json({"n_sequences": len(seqs_audio), "n_utts_aligned": n_done,
                "speech_frac": round(speech_frac, 4), "cache": args.cache}, args.out)
    LOG.info("Aligned eval: %d sequences from %d utts, speech_frac=%.3f (intra-pauses are silence)",
             len(seqs_audio), n_done, speech_frac)
    LOG.info("Cached %s — now run compare_models.py --cache %s", args.cache, args.cache)


if __name__ == "__main__":
    main()
