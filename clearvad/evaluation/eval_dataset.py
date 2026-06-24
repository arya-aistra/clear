"""Controlled, INDEPENDENTLY-labeled VAD evaluation set.

Ground truth comes from *construction*, NOT from Silero — so it's a fair bar for comparing
Silero vs ClearVAD. Real speech (held-out LibriSpeech test-clean) is interleaved with
controlled silence / noise gaps at known boundaries. We get exact labels for:
  * frame F1 / FAR / MR (with a documented caveat: intra-utterance pauses inside a speech
    segment are labeled speech, so MR is mildly inflated — equally for every model),
  * onset / endpoint latency (boundaries are exact),
  * short-silence detection (inserted gaps of known duration).

Reuses RealSpeechSource (torchaudio LibriSpeech download → soundfile) for the speech buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from clearvad import CHUNK_MS, CHUNK_SAMPLES, SAMPLE_RATE
from clearvad.distill.synthetic_gen import SyntheticAudioGenerator
from clearvad.utils.audio import mix_at_snr, rms_normalize


@dataclass
class EvalSequence:
    audio: np.ndarray            # [L] float32
    labels: np.ndarray           # [K] bool (per 512-sample chunk; True = speech)
    silence_gaps: List[Tuple[int, int, float]]  # (start_frame, end_frame, dur_ms) inserted gaps


def _to_chunk_labels(sample_labels: np.ndarray, n_chunks: int) -> np.ndarray:
    """Majority-vote sample labels (1=speech) down to per-chunk labels."""
    out = np.zeros(n_chunks, dtype=bool)
    for k in range(n_chunks):
        seg = sample_labels[k * CHUNK_SAMPLES:(k + 1) * CHUNK_SAMPLES]
        if seg.size:
            out[k] = seg.mean() >= 0.5
    return out


def build_eval_set(
    speech_buffer: np.ndarray,
    n_sequences: int = 60,
    seq_seconds: float = 12.0,
    speech_seg_ms=(500, 2500),
    silence_gap_ms=(50, 100, 150, 200, 300, 500, 800),
    noise_prob: float = 0.4,
    snr_range=(5.0, 20.0),
    seed: int = 12345,
    normalize_dbfs: float = -23.0,
    noise_source=None,
) -> List[EvalSequence]:
    """Construct controlled eval sequences with exact speech/silence boundaries.

    Pass `noise_source` (with .sample(n, rng), e.g. MUSAN) + a low `snr_range` to build a HARD,
    real-world noisy eval (speech-in-noise + real-noise backgrounds) — the deployment condition.
    """
    rng = np.random.default_rng(seed)
    gen = SyntheticAudioGenerator(SAMPLE_RATE)

    def _noise(n):
        return noise_source.sample(n, rng) if noise_source is not None else gen.noise(n, rng)

    L = int(seq_seconds * SAMPLE_RATE)
    K = L // CHUNK_SAMPLES
    buf, BN = speech_buffer, len(speech_buffer)
    sequences: List[EvalSequence] = []

    for _ in range(n_sequences):
        audio = np.zeros(L, dtype=np.float32)
        sample_lab = np.zeros(L, dtype=np.float32)
        gaps: List[Tuple[int, int, float]] = []
        pos = 0
        place_speech = True  # start with speech
        while pos < L:
            if place_speech:
                dur_ms = rng.uniform(*speech_seg_ms)
                seg = int(dur_ms / 1000 * SAMPLE_RATE)
                seg = min(seg, L - pos)
                s = int(rng.integers(0, max(BN - seg, 1)))
                clip = buf[s:s + seg].copy()
                if rng.random() < noise_prob:
                    clip, _ = mix_at_snr(clip, _noise(seg), float(rng.uniform(*snr_range)))
                audio[pos:pos + seg] = clip
                sample_lab[pos:pos + seg] = 1.0
                pos += seg
            else:
                dur_ms = float(rng.choice(silence_gap_ms))
                seg = int(dur_ms / 1000 * SAMPLE_RATE)
                seg = min(seg, L - pos)
                # gap is either true silence or low-level non-speech noise (still label 0)
                if rng.random() < 0.5:
                    bg = _noise(seg)
                    audio[pos:pos + seg] = (bg / (np.max(np.abs(bg)) + 1e-9) * 0.1).astype(np.float32)
                gap_start_f = pos // CHUNK_SAMPLES
                gap_end_f = (pos + seg) // CHUNK_SAMPLES
                if gap_end_f > gap_start_f:
                    gaps.append((gap_start_f, gap_end_f, dur_ms))
                pos += seg
            place_speech = not place_speech

        if normalize_dbfs is not None:
            audio = rms_normalize(audio, normalize_dbfs)
        labels = _to_chunk_labels(sample_lab, K)
        sequences.append(EvalSequence(audio=audio[:K * CHUNK_SAMPLES], labels=labels, silence_gaps=gaps))
    return sequences


def short_silence_detection(pred_mask: np.ndarray, gaps, threshold_frames: int = 0) -> dict:
    """For each inserted gap, did the prediction go to silence anywhere inside it?

    Returns per-duration detection rate (the short-silence claim metric).
    """
    by_dur: dict = {}
    for (s, e, dur_ms) in gaps:
        window = pred_mask[s:e]
        detected = bool((~window).any())  # any predicted-silence frame inside the gap
        by_dur.setdefault(dur_ms, []).append(detected)
    return {dur: round(float(np.mean(v)), 4) for dur, v in sorted(by_dur.items())}
