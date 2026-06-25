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
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE  # noqa: E402
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


def normalize(text: str) -> List[str]:
    """MMS_FA wants lowercase latin words, no punctuation."""
    text = re.sub(r"[^a-z' ]", " ", text.lower())
    return [w for w in text.split() if w]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-dir", default="data/librispeech/LibriSpeech/test-clean")
    ap.add_argument("--max-utts", type=int, default=300)
    ap.add_argument("--utts-per-seq", type=int, default=4)
    ap.add_argument("--gap-ms", type=float, default=400.0, help="true silence between utterances")
    ap.add_argument("--cache", default="data/eval/aligned_eval.npz")
    ap.add_argument("--out", default="reports/phase8/aligned_eval_info.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import torchaudio

    ls_dir = Path(args.ls_dir)
    if not ls_dir.exists():
        raise SystemExit(f"{ls_dir} not found — run build_eval_set.py --ls-url test-clean first "
                         "to download/extract LibriSpeech.")

    LOG.info("Loading MMS_FA forced aligner...")
    bundle = torchaudio.pipelines.MMS_FA
    device = args.device if torch.cuda.is_available() else "cpu"
    model = bundle.get_model().to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    utts = find_utterances(ls_dir, args.max_utts)
    LOG.info("Aligning %d utterances...", len(utts))

    def utt_labels(flac: Path, text: str):
        """Return (audio[L], sample_speech_mask[L]) with frame-accurate speech from alignment."""
        wav = load_audio(flac, SAMPLE_RATE)
        words = normalize(text)
        if not words:
            return None
        wt = torch.from_numpy(wav).unsqueeze(0).to(device)
        with torch.inference_mode():
            emission, _ = model(wt)
        token_spans = aligner(emission[0], tokenizer(words))
        ratio = wav.shape[0] / emission.shape[1]
        mask = np.zeros(wav.shape[0], dtype=np.float32)
        for spans in token_spans:                      # one entry per word
            if not spans:
                continue
            s = int(spans[0].start * ratio)
            e = int(spans[-1].end * ratio)
            mask[max(s, 0):min(e, len(mask))] = 1.0
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
