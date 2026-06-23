"""Real-audio source for distillation — real speech + controllable synthetic noise.

The synthetic-only trial capped at speech_f1≈0.36 because Silero's "speech" signal on
synthetic audio is sparse/weak. Real speech gives the teacher dense, confident, in-
distribution labels → a real speech detector. Labels still come ONLY from the teacher
(no human VAD annotations) — this is "zero external LABELS" distillation.

`RealSpeechSource` prefetches a buffer of real speech ONCE, then serves random clips cheaply
each pool refresh, optionally mixed with synthetic noise at a random SNR.

Three sources (in order of reliability):
  source="torchaudio"  → torchaudio LibriSpeech downloader (default; OpenSLR, 16 kHz, no HF
                          dataset-script issues). dev-clean (~337 MB, 5.4 h) is plenty.
  source="local"       → a directory of wav/flac files (zero-risk; pass local_dir).
  source="hf"          → HuggingFace `datasets` streaming (modern Parquet repos only).
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
    def __init__(
        self,
        source: str = "torchaudio",
        local_dir: Optional[str] = None,
        # torchaudio LibriSpeech
        ls_url: str = "dev-clean",
        ls_root: str = "data/librispeech",
        # HF (modern Parquet repos only; the legacy script-based `librispeech_asr` is dead)
        hf_dataset: str = "openslr/librispeech_asr",
        hf_config: Optional[str] = "clean",
        hf_split: str = "train.clean.100",
        sample_rate: int = SAMPLE_RATE,
        buffer_seconds: float = 1800.0,
        normalize_dbfs: Optional[float] = -23.0,
        seed: int = 1234,
    ) -> None:
        self.sr = sample_rate
        self.normalize_dbfs = normalize_dbfs
        if local_dir:
            source = "local"
        target = int(buffer_seconds * self.sr)
        if source == "local":
            self.buffer = self._from_local_dir(local_dir, target)
        elif source == "torchaudio":
            self.buffer = self._from_torchaudio(ls_root, ls_url, target)
        elif source == "hf":
            self.buffer = self._from_hf(hf_dataset, hf_config, hf_split, target)
        else:
            raise ValueError(f"unknown real-speech source: {source!r}")
        LOG.info("Real-speech buffer: %.1f s (%d samples) from source=%s",
                 len(self.buffer) / self.sr, len(self.buffer), source)

    # ------------------------------------------------------------ sources
    def _from_torchaudio(self, root: str, url: str, target: int) -> np.ndarray:
        import torchaudio
        Path(root).mkdir(parents=True, exist_ok=True)
        LOG.info("Loading torchaudio LibriSpeech [%s] into %s (downloads on first run)...",
                 url, root)
        ds = torchaudio.datasets.LIBRISPEECH(root, url=url, download=True)
        chunks, total, n = [], 0, len(ds)
        for i in range(n):
            wav, sr, *_ = ds[i]                      # wav: [1, T] float32
            a = wav.squeeze(0).contiguous().numpy().astype(np.float32)
            if sr != self.sr:
                a = resample(a, sr, self.sr)
            chunks.append(a)
            total += len(a)
            if total >= target:
                break
        return np.concatenate(chunks)[:target]

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
            raise ImportError("HF path needs `datasets`: pip install datasets soundfile") from exc
        LOG.info("Streaming HF dataset %s/%s [%s] ...", dataset_name, config, split)
        try:
            # NOTE: no trust_remote_code (removed in modern datasets); use Parquet repos only.
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load HF dataset {dataset_name}/{config} [{split}]: {exc!r}. "
                f"Use --speech-source torchaudio (default) or --local-speech-dir instead."
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
