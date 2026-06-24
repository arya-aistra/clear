"""Constructed, GROUND-TRUTH-labeled training data — the fix for the multi-teacher failure.

Root cause of the failed teacher runs: training clips were continuous real speech (all-speech)
+ synthetic (which strong teachers don't fire on), so teacher labels degenerated into
"real vs synthetic" and the student never learned within-audio speech/silence boundaries.

Fix: build training clips that interleave REAL speech segments with REAL silence / noise gaps
at KNOWN boundaries. We therefore have **true per-chunk labels** (we place the silences), so we
can train ClearVAD SUPERVISED on ground truth — which can genuinely exceed Silero (Silero was
never trained on these). Labels use no human VAD annotation (silence is placed programmatically;
speech is unlabeled LibriSpeech). Optionally blends in a teacher's soft labels for robustness.

This pool is a drop-in for the trainer: `.sample()`, `.holdout()`, `.refresh()`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from clearvad import CHUNK_SAMPLES, CONTEXT_SAMPLES
from clearvad.utils.audio import mix_at_snr, rms_normalize


def construct_clip(buffer: np.ndarray, n_samples: int, rng: np.random.Generator,
                   gen, speech_seg_ms=(300, 1200), silence_ms=(200, 900),
                   noise_prob: float = 0.5, snr_range=(3.0, 20.0),
                   normalize_dbfs: float = -23.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build one clip = alternating real speech / silence(or noise) -> (audio[L], labels[K] bool).

    Any single segment is capped at half the clip, so every clip alternates at least once and
    therefore contains BOTH speech and silence (avoids the all-speech degeneracy)."""
    audio = np.zeros(n_samples, dtype=np.float32)
    sample_lab = np.zeros(n_samples, dtype=np.float32)
    BN = len(buffer)
    half = max(n_samples // 2, CHUNK_SAMPLES)
    pos = 0
    place_speech = bool(rng.random() < 0.6)
    while pos < n_samples:
        if place_speech:
            seg = int(rng.uniform(*speech_seg_ms) / 1000 * 16000)
            seg = min(seg, n_samples - pos, half)
            s = int(rng.integers(0, max(BN - seg, 1)))
            clip = buffer[s:s + seg].copy()
            if rng.random() < noise_prob:
                clip, _ = mix_at_snr(clip, gen.noise(seg, rng), float(rng.uniform(*snr_range)))
            audio[pos:pos + seg] = clip
            sample_lab[pos:pos + seg] = 1.0
            pos += seg
        else:
            seg = int(rng.uniform(*silence_ms) / 1000 * 16000)
            seg = min(seg, n_samples - pos, half)
            if rng.random() < 0.5:                       # real silence vs low-level noise
                audio[pos:pos + seg] = gen.noise(seg, rng) * 0.1
            pos += seg
        place_speech = not place_speech

    if normalize_dbfs is not None:
        audio = rms_normalize(audio, normalize_dbfs)
    K = n_samples // CHUNK_SAMPLES
    labels = np.zeros(K, dtype=np.float32)
    for k in range(K):
        labels[k] = 1.0 if sample_lab[k * CHUNK_SAMPLES:(k + 1) * CHUNK_SAMPLES].mean() >= 0.5 else 0.0
    return audio[: K * CHUNK_SAMPLES], labels


class ConstructedDataPool:
    """Refreshable pool of constructed clips with true labels (+ optional teacher soft labels)."""

    def __init__(self, real_source, generator, pool_size: int, clip_chunks: int,
                 teacher=None, true_weight: float = 1.0, label_smooth: float = 0.03,
                 label_batch: int = 256) -> None:
        self.real = real_source
        self.gen = generator
        self.teacher = teacher                  # optional auxiliary soft-label teacher
        self.pool_size = pool_size
        self.clip_chunks = clip_chunks
        self.clip_samples = clip_chunks * CHUNK_SAMPLES
        self.true_weight = float(true_weight)   # 1.0 = pure supervised; <1 blends teacher soft
        self.label_smooth = float(label_smooth)
        self.label_batch = label_batch
        self.audio = None                       # [P, L] cpu tensor
        self.target = None                      # [P, K] cpu tensor (blended, smoothed)
        self._holdout = None

    def _build(self, seed: int):
        import torch
        rng = np.random.default_rng(seed)
        audio = np.stack([construct_clip(self.real.buffer, self.clip_samples, rng, self.gen)[0]
                          for _ in range(self.pool_size)])
        # recompute labels deterministically alongside audio (same rng stream order)
        rng2 = np.random.default_rng(seed)
        labels = np.stack([construct_clip(self.real.buffer, self.clip_samples, rng2, self.gen)[1]
                           for _ in range(self.pool_size)])
        audio_t = torch.from_numpy(audio)
        true = torch.from_numpy(labels)
        # label smoothing on the hard targets (stabilizes the temperature-KL term)
        if self.label_smooth > 0:
            true = true * (1 - 2 * self.label_smooth) + self.label_smooth
        target = true
        if self.teacher is not None and self.true_weight < 1.0:
            soft_chunks = []
            for i in range(0, self.pool_size, self.label_batch):
                soft_chunks.append(self.teacher.label(audio_t[i:i + self.label_batch]))
            soft = torch.cat(soft_chunks, dim=0)[:, : true.shape[1]]
            target = self.true_weight * true + (1 - self.true_weight) * soft
        return audio_t, target

    def refresh(self, seed: int, **_ignore) -> dict:
        self.audio, self.target = self._build(seed)
        return {"pool_seed": seed, "speech_frac": round(float((self.target > 0.5).float().mean()), 4)}

    def _windows(self, audio):
        import torch
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        K = L // CHUNK_SAMPLES
        win = CONTEXT_SAMPLES + CHUNK_SAMPLES
        padded = torch.cat([audio.new_zeros(B, CONTEXT_SAMPLES), audio], dim=1)
        out = audio.new_zeros(B, K, win)
        for i in range(K):
            out[:, i] = padded[:, i * CHUNK_SAMPLES:i * CHUNK_SAMPLES + win]
        return out

    def sample(self, batch_size: int, rng: np.random.Generator, device):
        import torch
        idx = torch.from_numpy(rng.integers(0, self.pool_size, size=batch_size))
        windows = self._windows(self.audio.index_select(0, idx)).to(device)
        target = self.target.index_select(0, idx).to(device)
        return windows, target

    def holdout(self, n: int = 64, seed: int = 987654):
        if self._holdout is None:
            import torch
            rng = np.random.default_rng(seed)
            audio = np.stack([construct_clip(self.real.buffer, self.clip_samples, rng, self.gen)[0]
                              for _ in range(n)])
            rng2 = np.random.default_rng(seed)
            labels = np.stack([construct_clip(self.real.buffer, self.clip_samples, rng2, self.gen)[1]
                               for _ in range(n)])
            windows = self._windows(torch.from_numpy(audio))
            self._holdout = (windows, torch.from_numpy(labels))
        return self._holdout
