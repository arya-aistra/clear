"""Forced alignment (torchaudio MMS_FA) → frame-accurate speech masks.

Shared by the frame-accurate EVAL (build_aligned_eval.py) and frame-accurate TRAINING labels
(RealSpeechSource aligned mode). Produces a sample-resolution speech mask: 1 inside an aligned
word span, 0 for inter-word / leading / trailing silence.

`min_silence_ms` smoothing fills interior silence gaps shorter than the threshold (treats them
as speech). This follows standard VAD label convention — co-articulation micro-gaps between
words (tens of ms) are NOT real pauses, and no VAD (incl. Silero) is expected to release on
them. Training labels use smoothing (~100 ms) so the model learns genuine-pause detection, not
micro-gap chatter. The strict eval can run raw (0 ms) since all models are scored identically.
"""

from __future__ import annotations

import re

import numpy as np

from clearvad.utils.logging_utils import get_logger

LOG = get_logger("align")


def normalize_transcript(text: str):
    """MMS_FA wants lowercase latin words, no punctuation."""
    text = re.sub(r"[^a-z' ]", " ", text.lower())
    return [w for w in text.split() if w]


def fill_short_silences(mask: np.ndarray, min_gap_samples: int) -> np.ndarray:
    """Fill INTERIOR runs of silence shorter than min_gap_samples (bounded by speech) → speech.
    Leading/trailing silence is left untouched (those are real non-speech regions)."""
    if min_gap_samples <= 0:
        return mask
    out = mask.copy()
    n = len(out)
    k = 0
    while k < n:
        if out[k] == 0:
            j = k
            while j < n and out[j] == 0:
                j += 1
            if k > 0 and j < n and (j - k) < min_gap_samples:  # interior + short
                out[k:j] = 1.0
            k = j
        else:
            k += 1
    return out


class ForcedAligner:
    """Wraps MMS_FA. One instance reused across many utterances."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        import torchaudio

        self.torch = torch
        bundle = torchaudio.pipelines.MMS_FA
        self.device = device if torch.cuda.is_available() else "cpu"
        LOG.info("Loading MMS_FA forced aligner on %s ...", self.device)
        self.model = bundle.get_model().to(self.device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def speech_mask(self, wav: np.ndarray, transcript: str,
                    min_silence_ms: float = 0.0, pad_ms: float = 0.0, sr: int = 16000):
        """Return a sample-resolution speech mask [len(wav)] (1=speech), or None if untokenizable.

        pad_ms: extend each word span by this much on each side. MMS_FA places token boundaries
        slightly inside the acoustic word, so the raw mask trims real speech at word edges (=> a
        model trained on it under-detects speech, high miss-rate). ~40 ms recovers the true extent.
        """
        words = normalize_transcript(transcript)
        if not words:
            return None
        wt = self.torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            emission, _ = self.model(wt)
        token_spans = self.aligner(emission[0], self.tokenizer(words))
        ratio = wav.shape[0] / emission.shape[1]
        pad = int(pad_ms / 1000 * sr)
        mask = np.zeros(wav.shape[0], dtype=np.float32)
        for spans in token_spans:                       # one entry per word
            if not spans:
                continue
            s = int(spans[0].start * ratio) - pad
            e = int(spans[-1].end * ratio) + pad
            mask[max(s, 0):min(e, len(mask))] = 1.0
        if min_silence_ms > 0:
            mask = fill_short_silences(mask, int(min_silence_ms / 1000 * sr))
        return mask
