"""FireRedVAD as an OPTIONAL auxiliary teacher (Apache-2.0; SOTA, 0.6M params, 97.57% F1).

NOTE: FireRedVAD's API is file-based (`detect_full(wav_path)`) at a 10 ms hop, so labeling is
slower (temp-WAV per clip) and is intended as an *auxiliary* soft-label blended with the
construction's ground-truth labels — not the primary signal. The accuracy beat comes from the
true labels; FireRedVAD only adds robustness if it helps.

Install: pip install fireredvad ; huggingface-cli download FireRedTeam/FireRedVAD \
             --local-dir pretrained_models/FireRedVAD
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from clearvad import CHUNK_SAMPLES, CONTEXT_SAMPLES, SAMPLE_RATE


class FireRedVADTeacher:
    def __init__(self, model_dir: str = "pretrained_models/FireRedVAD/Stream-VAD",
                 device: str = "cpu", speech_threshold: float = 0.4) -> None:
        try:
            from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FireRedVADTeacher needs fireredvad: pip install fireredvad ; then "
                "huggingface-cli download FireRedTeam/FireRedVAD --local-dir pretrained_models/FireRedVAD"
            ) from exc
        cfg = FireRedStreamVadConfig(use_gpu=(device == "cuda"), smooth_window_size=5,
                                     speech_threshold=speech_threshold, pad_start_frame=5,
                                     min_speech_frame=8, max_speech_frame=2000,
                                     min_silence_frame=20, chunk_max_frame=30000)
        self.vad = FireRedStreamVad.from_pretrained(model_dir, cfg)
        self.sr = SAMPLE_RATE

    @property
    def torch(self):
        import torch
        return torch

    @staticmethod
    def _resample(vec: np.ndarray, k: int) -> np.ndarray:
        f = len(vec)
        if f == 0:
            return np.zeros(k, dtype=np.float32)
        src = (np.arange(f) + 0.5) / f
        tgt = (np.arange(k) + 0.5) / k
        return np.interp(tgt, src, vec).astype(np.float32)

    def _label_one(self, wav: np.ndarray, k: int) -> np.ndarray:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            sf.write(tf.name, np.clip(wav, -1, 1).astype(np.float32), self.sr, subtype="PCM_16")
            frame_results, _ = self.vad.detect_full(tf.name)
        probs = np.array([getattr(fr, "smoothed_prob", 0.0) for fr in frame_results],
                         dtype=np.float32)
        return self._resample(probs, k)

    def label(self, audio):
        import torch
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        audio = audio.to(torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        K = L // CHUNK_SAMPLES
        a = audio.cpu().numpy()
        out = np.stack([self._label_one(a[b], K) for b in range(B)])
        return torch.from_numpy(out.astype(np.float32))

    def build_student_windows(self, audio):
        torch = self.torch
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        audio = audio.to(torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        K = L // CHUNK_SAMPLES
        win = CONTEXT_SAMPLES + CHUNK_SAMPLES
        padded = torch.cat([audio.new_zeros(B, CONTEXT_SAMPLES), audio], dim=1)
        windows = audio.new_zeros(B, K, win)
        for i in range(K):
            windows[:, i] = padded[:, i * CHUNK_SAMPLES:i * CHUNK_SAMPLES + win]
        return windows
