"""Real-audio source for distillation — real speech + controllable synthetic noise.

The synthetic-only trial capped at speech_f1≈0.36 because Silero's "speech" signal on
synthetic audio is sparse/weak. Real speech gives the teacher dense, confident, in-
distribution labels → a real speech detector. Labels still come ONLY from the teacher
(no human VAD annotations) — this is "zero external LABELS" distillation.

`RealSpeechSource` prefetches a buffer of real speech ONCE (HF streaming dataset or a local
wav directory), then serves random clips cheaply each pool refresh. Clips can be mixed with
our synthetic noise at a random SNR for augmentation/robustness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from clearvad import SAMPLE_RATE
from clearvad.utils.audio import mix_at_snr, resample, rms_normalize
from clearvad.utils.logging_utils import get_logger

LOG = get_logger("realdata")


class RealSpeechSource:
    """Buffer of real speech sampled into fixed-length clips.

    Parameters
    ----------
    dataset_name / config / split : HF `datasets` spec (e.g. "librispeech_asr","clean",
        "train.clean.100"). Ignored if `local_dir` is given.
    local_dir : directory of wav/flac files to use instead of HF (de-risks network/auth).
    buffer_seconds : how much audio to prefetch into the in-memory buffer.
    """

    def __init__(
        self,
        dataset_name: str = "librispeech_asr",
        config: Optional[str] = "clean",
        split: str = "train.clean.100",
        local_dir: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        buffer_seconds: float = 1800.0,
        normalize_dbfs: Optional[float] = -23.0,
        seed: int = 1234,
    ) -> None:
        self.sr = sample_rate
        self.normalize_dbfs = normalize_dbfs
        self.buffer = self._prefetch(dataset_name, config, split, local_dir,
                                     buffer_seconds, seed)
        LOG.info("Real-speech buffer: %.1f s (%d samples)",
                 len(self.buffer) / self.sr, len(self.buffer))

    def _prefetch(self, dataset_name, config, split, local_dir, buffer_seconds, seed):
        target = int(buffer_seconds * self.sr)
        if local_dir:
            return self._from_local_dir(local_dir, target)
        return self._from_hf(dataset_name, config, split, target)

    def _from_local_dir(self, local_dir: str, target: int) -> np.ndarray:
        from clearvad.utils.audio import load_audio
        files = sorted([p for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg")
                        for p in Path(local_dir).rglob(ext)])
        if not files:
            raise FileNotFoundError(f"No audio files under {local_dir}")
        chunks, total = [], 0
        for f in files:
            try:
                a = load_audio(f, self.sr)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("skip %s: %r", f, exc)
                continue
            chunks.append(a.astype(np.float32))
            total += len(a)
            if total >= target:
                break
        if not chunks:
            raise RuntimeError(f"Could not load any audio from {local_dir}")
        return np.concatenate(chunks)[:target]

    def _from_hf(self, dataset_name, config, split, target: int) -> np.ndarray:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Real-audio path needs the `datasets` package: "
                              "pip install datasets soundfile") from exc
        LOG.info("Streaming HF dataset %s/%s [%s] ...", dataset_name, config, split)
        try:
            ds = load_dataset(dataset_name, config, split=split, streaming=True,
                              trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load HF dataset {dataset_name}/{config} [{split}]: {exc!r}. "
                f"Check access/network, or pass local_dir=<speech folder>."
            ) from exc
        chunks, total = [], 0
        for ex in ds:
            audio = ex.get("audio")
            if audio is None:
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", self.sr))
            if sr != self.sr:
                arr = resample(arr, sr, self.sr)
            chunks.append(arr)
            total += len(arr)
            if total >= target:
                break
        if not chunks:
            raise RuntimeError(f"HF dataset {dataset_name} yielded no audio.")
        return np.concatenate(chunks)[:target]

    # ------------------------------------------------------------ sampling
    def sample_clips(self, n: int, clip_samples: int, rng: np.random.Generator,
                     noise_gen=None, noise_prob: float = 0.5,
                     snr_range=(0.0, 20.0)) -> np.ndarray:
        """Return [n, clip_samples] random real-speech clips (optionally noise-mixed)."""
        buf, L = self.buffer, len(self.buffer)
        out = np.zeros((n, clip_samples), dtype=np.float32)
        for i in range(n):
            if L > clip_samples:
                s = int(rng.integers(0, L - clip_samples))
                clip = buf[s:s + clip_samples].copy()
            else:
                reps = int(np.ceil(clip_samples / max(L, 1)))
                clip = np.tile(buf, reps)[:clip_samples].copy()
            if noise_gen is not None and rng.random() < noise_prob:
                noise = noise_gen.noise(clip_samples, rng)
                clip, _ = mix_at_snr(clip, noise, float(rng.uniform(*snr_range)))
            if self.normalize_dbfs is not None:
                clip = rms_normalize(clip, self.normalize_dbfs)
            out[i] = clip
        return out
