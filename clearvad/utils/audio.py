"""Audio IO, resampling, and chunking utilities (numpy-first, torch-free).

These helpers underpin the frozen evaluation protocol: a waveform is always
normalized to mono float32 @ 16 kHz, then sliced into non-overlapping 512-sample
chunks. Frame index <-> time conversions all reference CHUNK_MS so that latency
metrics are comparable across every model and phase.
"""

from __future__ import annotations

import os
from typing import Iterator, Tuple, Union

import numpy as np

from clearvad import CHUNK_MS, CHUNK_SAMPLES, SAMPLE_RATE

PathLike = Union[str, os.PathLike]


def load_audio(path: PathLike, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file to mono float32 at ``target_sr``.

    Uses soundfile for decoding (wav/flac/ogg). For mp3, soundfile>=0.12 with libsndfile
    >=1.1 works; otherwise the caller should pre-convert. Returns 1D float32 array in
    [-1, 1].
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_audio requires soundfile (pip install soundfile)") from exc

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    data = to_mono(data)
    if sr != target_sr:
        data = resample(data, sr, target_sr)
    return np.ascontiguousarray(data, dtype=np.float32)


def to_mono(wav: np.ndarray) -> np.ndarray:
    """Collapse a multi-channel array to mono float32."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        # soundfile returns [samples, channels]
        ch_axis = 1 if wav.shape[1] <= wav.shape[0] else 0
        wav = wav.mean(axis=ch_axis)
    return wav.reshape(-1)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resample (scipy), torchaudio fallback."""
    if orig_sr == target_sr:
        return np.asarray(wav, dtype=np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        out = resample_poly(wav.astype(np.float64), up, down)
        return out.astype(np.float32)
    except ImportError:
        pass
    # torchaudio fallback
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    out = AF.resample(t, orig_sr, target_sr).squeeze(0).numpy()
    return out.astype(np.float32)


def iter_chunks(wav: np.ndarray, chunk_samples: int = CHUNK_SAMPLES,
                drop_last: bool = True) -> Iterator[np.ndarray]:
    """Yield non-overlapping chunks of ``chunk_samples``.

    With ``drop_last=True`` (default, Silero-compatible) the trailing partial chunk is
    dropped. With ``drop_last=False`` the last chunk is zero-padded to full length.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = len(wav)
    full = n // chunk_samples
    for i in range(full):
        yield wav[i * chunk_samples : (i + 1) * chunk_samples]
    if not drop_last and n % chunk_samples:
        tail = wav[full * chunk_samples :]
        pad = np.zeros(chunk_samples, dtype=np.float32)
        pad[: len(tail)] = tail
        yield pad


def num_chunks(num_samples: int, chunk_samples: int = CHUNK_SAMPLES) -> int:
    return num_samples // chunk_samples


def frame_to_ms(frame_idx: int, chunk_ms: float = CHUNK_MS) -> float:
    """Center time (ms) of frame ``frame_idx`` (0-based)."""
    return (frame_idx + 0.5) * chunk_ms


def ms_to_frame(ms: float, chunk_ms: float = CHUNK_MS) -> int:
    return int(round(ms / chunk_ms - 0.5))


def samples_to_ms(n: int, sr: int = SAMPLE_RATE) -> float:
    return n / sr * 1000.0


def rms_normalize(wav: np.ndarray, target_dbfs: float = -23.0,
                  eps: float = 1e-9) -> np.ndarray:
    """Normalize to a target RMS level (dBFS). Used to standardize input loudness."""
    wav = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt(np.mean(wav ** 2) + eps))
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / (rms + eps)
    return (wav * gain).astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
               eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """Mix ``speech`` and ``noise`` at a target SNR (dB).

    Returns (mixture, scaled_noise). ``noise`` is tiled/truncated to match speech length.
    """
    speech = np.asarray(speech, dtype=np.float32).reshape(-1)
    noise = np.asarray(noise, dtype=np.float32).reshape(-1)
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    noise = noise[: len(speech)]
    sp = float(np.mean(speech ** 2) + eps)
    npw = float(np.mean(noise ** 2) + eps)
    target = sp / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target / npw))
    scaled_noise = (noise * scale).astype(np.float32)
    return (speech + scaled_noise).astype(np.float32), scaled_noise
