"""Adapters for third-party VADs used as benchmark baselines, mapped onto ClearVAD's eval grid.

Each adapter: `.load()` (raises if the package/model is unavailable) then `.probs(audio_1d)` →
np.ndarray[K] aligned to 512-sample (32 ms) chunks. The benchmark wraps `.load()` in try/except so
an uninstalled model is reported "unavailable" rather than crashing the run.

Currently supported baseline: NeMo Frame-VAD MarbleNet (via the existing teacher). Silero and WebRTC
are wired directly in scripts/benchmark_all.py.
"""

from __future__ import annotations

import numpy as np

from clearvad import CHUNK_SAMPLES


def align_to_chunks(src_probs, src_hop_samples: float, n_samples: int,
                    chunk: int = CHUNK_SAMPLES) -> np.ndarray:
    """Map probabilities sampled every `src_hop_samples` to K=n_samples//chunk chunk-probs
    (mean of the source frames overlapping each chunk). Utility for any frame-rate baseline."""
    K = n_samples // chunk
    out = np.zeros(K, dtype=np.float32)
    src = np.asarray(src_probs, dtype=np.float32).reshape(-1)
    if src.size == 0 or src_hop_samples <= 0:
        return out
    for k in range(K):
        f0 = int((k * chunk) // src_hop_samples)
        f1 = max(f0 + 1, int(np.ceil(((k + 1) * chunk) / src_hop_samples)))
        seg = src[f0:min(f1, src.size)]
        out[k] = float(seg.mean()) if seg.size else float(src[min(f0, src.size - 1)])
    return out


class NeMoVAD:
    name = "nemo"
    params = None

    def load(self) -> None:
        import torch
        from clearvad.distill.nemo_teacher import NeMoMarbleTeacher
        self.teacher = NeMoMarbleTeacher()
        self.torch = torch

    def probs(self, audio: np.ndarray) -> np.ndarray:
        wav = self.torch.from_numpy(np.ascontiguousarray(audio, np.float32)).unsqueeze(0)
        out = self.teacher.label(wav)                       # [1, K] aligned to 512-chunks
        return np.asarray(out, dtype=np.float32).reshape(-1)


REGISTRY = {a.name: a for a in (NeMoVAD,)}
