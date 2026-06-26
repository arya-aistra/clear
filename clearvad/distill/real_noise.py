"""Real-world noise source (MUSAN) for realistic, hard, still-labeled VAD training.

Real deployment is call-centre / babble / traffic / construction / music — not synthetic
noise. We mix REAL noise (MUSAN noise+music subsets) into the constructed clips: as the
non-speech segments AND mixed into speech at low SNR (the hard speech-in-noise case). Labels
stay ground-truth (we know which segments are speech) — no human annotation needed.

Sources (in order of reliability on this box):
  source="local"   → a directory of noise wav/flac (zero risk; pass --noise-dir)
  source="openslr" → downloads MUSAN (musan.tar.gz, ~11 GB) from OpenSLR — the same host
                     LibriSpeech downloaded from; extracts only the noise+music subsets.

Install note: no extra deps (torch.hub + tarfile + soundfile, all already present).
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from clearvad import SAMPLE_RATE
from clearvad.utils.audio import load_audio, resample
from clearvad.utils.logging_utils import get_logger

LOG = get_logger("realnoise")
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"


class RealNoiseSource:
    def __init__(self, source: str = "openslr", local_dir: Optional[str] = None,
                 root: str = "data/musan", subsets: Sequence[str] = ("noise", "music"),
                 buffer_seconds: float = 1800.0, sample_rate: int = SAMPLE_RATE,
                 max_files: int = 400, seed: int = 1234,
                 hf_repo: Optional[str] = None) -> None:
        self.sr = sample_rate
        target = int(buffer_seconds * self.sr)
        if local_dir:
            self.buffer = self._load_buffer(Path(local_dir), subsets, target, max_files, seed)
        elif source == "openslr":
            self.buffer = self._load_buffer(self._ensure_musan(root, subsets), subsets,
                                            target, max_files, seed)
        elif source == "hf":
            self.buffer = self._buffer_from_hf(hf_repo, target)
        else:
            raise ValueError(f"unknown noise source {source!r}")
        LOG.info("Real-noise buffer: %.1f s (%d samples) from %s",
                 len(self.buffer) / self.sr, len(self.buffer), local_dir or hf_repo or source)

    def _buffer_from_hf(self, hf_repo: Optional[str], target: int) -> np.ndarray:
        """Stream a Parquet-format HF noise dataset (e.g. DEMAND) and decode audio via soundfile
        (datasets' Audio(decode=False) → raw bytes → soundfile; avoids torchcodec)."""
        if not hf_repo:
            raise ValueError("source='hf' requires hf_repo (e.g. 'voice-biomarkers/DEMAND-acoustic-noise')")
        import io
        import soundfile as sf
        from datasets import Audio, load_dataset
        from clearvad.utils.audio import to_mono

        LOG.info("Streaming HF noise dataset %s (parquet) ...", hf_repo)
        split = "train"                                   # some noise repos (e.g. FSD50k) lack 'train'
        try:
            from datasets import get_dataset_split_names
            avail = get_dataset_split_names(hf_repo)
            if "train" not in avail and avail:
                split = avail[0]
                LOG.info("  no 'train' split; using %r (available: %s)", split, avail)
        except Exception:  # noqa: BLE001
            pass
        ds = load_dataset(hf_repo, split=split, streaming=True)
        audio_col = None
        for name, feat in (getattr(ds, "features", None) or {}).items():
            if feat.__class__.__name__ == "Audio":
                audio_col = name
                break
        audio_col = audio_col or "audio"
        try:
            ds = ds.cast_column(audio_col, Audio(decode=False))   # get raw bytes, decode ourselves
        except Exception:  # noqa: BLE001
            pass
        chunks, total = [], 0
        for ex in ds:
            a = ex.get(audio_col)
            if not isinstance(a, dict):
                continue
            if a.get("bytes") is not None:
                arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
            elif a.get("array") is not None:
                arr, sr = np.asarray(a["array"], dtype=np.float32), int(a.get("sampling_rate", self.sr))
            else:
                continue
            arr = to_mono(arr)
            if sr != self.sr:
                arr = resample(arr, sr, self.sr)
            chunks.append(arr.astype(np.float32))
            total += len(arr)
            if total >= target:
                break
        if not chunks:
            raise RuntimeError(f"HF noise dataset {hf_repo} yielded no audio (column={audio_col}).")
        return np.concatenate(chunks)[:target]

    # ------------------------------------------------------------ acquisition
    def _ensure_musan(self, root: str, subsets) -> Path:
        import torch
        root_p = Path(root)
        musan_dir = root_p / "musan"
        if any((musan_dir / s).exists() for s in subsets):
            return musan_dir
        root_p.mkdir(parents=True, exist_ok=True)
        tar_path = root_p / "musan.tar.gz"
        if not tar_path.exists():
            LOG.info("Downloading MUSAN (~11 GB) from %s ...", MUSAN_URL)
            torch.hub.download_url_to_file(MUSAN_URL, str(tar_path), progress=True)
        LOG.info("Extracting MUSAN subsets %s ...", tuple(subsets))
        with tarfile.open(tar_path, "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if any(f"/{s}/" in m.name for s in subsets) or m.isdir()]
            tf.extractall(root_p, members=members)
        return musan_dir

    def _load_buffer(self, audio_root: Path, subsets, target: int, max_files: int,
                     seed: int) -> np.ndarray:
        files: List[Path] = []
        for s in subsets:
            d = audio_root / s
            base = d if d.exists() else audio_root
            files += [p for ext in ("*.wav", "*.flac") for p in base.rglob(ext)]
        if not files:
            files = [p for ext in ("*.wav", "*.flac") for p in Path(audio_root).rglob(ext)]
        if not files:
            raise FileNotFoundError(
                f"No noise audio (.wav/.flac) under {audio_root}. If this is an HF repo with an "
                f"unusual layout, download/extract it manually and pass --noise-dir <that folder>.")
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        chunks, total = [], 0
        for f in files[:max_files]:
            try:
                a = load_audio(f, self.sr)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("skip noise %s: %r", f, exc)
                continue
            chunks.append(a.astype(np.float32))
            total += len(a)
            if total >= target:
                break
        if not chunks:
            raise RuntimeError(f"Could not load noise from {audio_root}")
        return np.concatenate(chunks)[:target]

    # ------------------------------------------------------------ sampling
    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """A random n_samples slice of real noise (tiled if the buffer is shorter)."""
        buf, L = self.buffer, len(self.buffer)
        if L >= n_samples:
            s = int(rng.integers(0, L - n_samples + 1))
            return buf[s:s + n_samples].copy()
        reps = int(np.ceil(n_samples / max(L, 1)))
        return np.tile(buf, reps)[:n_samples].copy()


# Token registry for --noise-sources (train on diverse noise; hold one corpus out for eval).
# urbansound8k/fsd50k add street/traffic/HVAC/public-space noise that overlaps DEMAND's acoustic
# environments — the families the musan+esc50 model under-rejected on the held-out noisy eval.
_NOISE_PRESETS = {
    "musan": dict(source="openslr"),
    "esc50": dict(source="hf", hf_repo="ashraq/esc50"),
    "demand": dict(source="hf", hf_repo="voice-biomarkers/DEMAND-acoustic-noise"),
    "urbansound8k": dict(source="hf", hf_repo="danavery/urbansound8K"),
    "fsd50k": dict(source="hf", hf_repo="Fhrozen/FSD50k"),
}


def _build_one(token: str, buffer_seconds: float) -> "RealNoiseSource":
    """Parse a noise token: a preset name, 'hf:<repo>', or 'local:<dir>'."""
    if token in _NOISE_PRESETS:
        return RealNoiseSource(buffer_seconds=buffer_seconds, **_NOISE_PRESETS[token])
    if token.startswith("hf:"):
        return RealNoiseSource(source="hf", hf_repo=token[3:], buffer_seconds=buffer_seconds)
    if token.startswith("local:"):
        return RealNoiseSource(source="local", local_dir=token[6:], buffer_seconds=buffer_seconds)
    raise ValueError(f"unknown noise token {token!r} (presets: {list(_NOISE_PRESETS)} | hf:<repo> | local:<dir>)")


class MultiNoiseSource:
    """Combine several noise corpora; each .sample() draws from a randomly chosen corpus.

    Training on diverse noise families (and HOLDING ONE OUT for eval) is the lever for real
    noise-robustness — the single-corpus (MUSAN-only) model failed to reject unseen DEMAND noise.
    Robust to a corpus failing to load (warns + skips, as long as >=1 loads)."""

    def __init__(self, tokens, buffer_seconds: float = 1800.0) -> None:
        self.sources = []
        for tok in tokens:
            try:
                self.sources.append(_build_one(tok, buffer_seconds))
                LOG.info("Noise corpus loaded: %s", tok)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Noise corpus %s failed to load (skipping): %r", tok, exc)
        if not self.sources:
            raise RuntimeError(f"No noise corpora loaded from {tokens}")
        LOG.info("MultiNoiseSource: %d corpora (%s)", len(self.sources), ", ".join(tokens))

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        src = self.sources[int(rng.integers(len(self.sources)))]
        return src.sample(n_samples, rng)
