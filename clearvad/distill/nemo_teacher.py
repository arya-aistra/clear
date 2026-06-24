"""NVIDIA NeMo Frame-VAD MarbleNet as a teacher — pure-PyTorch (no native libs).

Frame-VAD MarbleNet v2.0: ROC-AUC 92-97, 91.5K params, outputs a speech probability per
~20 ms frame. Pure PyTorch (avoids TEN VAD's libc++ / Pyannote's powerset). The 20 ms frame
rate does NOT divide our 32 ms chunk, so we resample with correct frame-center times and
verify alignment with the cross-correlation gate (scripts/test_nemo.py) before training.

Still label-free. ⚠️ nemo_toolkit may conflict with very new torch — install carefully.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clearvad import CHUNK_SAMPLES, CONTEXT_SAMPLES, SAMPLE_RATE


class NeMoMarbleTeacher:
    def __init__(self, model_name: str = "nvidia/frame_vad_multilingual_marblenet_v2.0",
                 device: Optional[str] = None, sub_batch: int = 16) -> None:
        try:
            import torch
            from nemo.collections.asr.models import EncDecFrameClassificationModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "NeMoMarbleTeacher needs nemo_toolkit[asr]. Install carefully (it may try to "
                "change torch): pip install 'nemo_toolkit[asr]'"
            ) from exc
        # strict=False: the published checkpoint omits the loss module's "loss.weight"
        # (not needed for inference); current NeMo registers it and would otherwise fail.
        try:
            self.model = EncDecFrameClassificationModel.from_pretrained(model_name, strict=False)
        except TypeError:
            # older NeMo from_pretrained without strict kwarg -> restore_from path
            self.model = EncDecFrameClassificationModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(self.device)
        self.sr = SAMPLE_RATE
        self.sub_batch = sub_batch
        self._logged_shape = False

    @property
    def torch(self):
        import torch
        return torch

    @staticmethod
    def _resample(vec: np.ndarray, k: int) -> np.ndarray:
        f = len(vec)
        if f == 0:
            return np.zeros(k, dtype=np.float32)
        src = (np.arange(f) + 0.5) / f      # frame centers in [0,1]
        tgt = (np.arange(k) + 0.5) / k
        return np.interp(tgt, src, vec).astype(np.float32)

    def _frame_speech(self, logits):
        """logits -> per-frame speech prob [B, F], handling [B,F,C] or [B,C,F] orientations."""
        import torch
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = torch.as_tensor(logits).float()
        if not self._logged_shape:
            from clearvad.utils.logging_utils import get_logger
            get_logger("nemo").info("NeMo logits shape=%s", tuple(logits.shape))
            self._logged_shape = True
        if logits.dim() == 2:                       # [B, F] (single logit per frame)
            return torch.sigmoid(logits)
        # 3D: find the class axis (size 1 or 2), speech = last class (or sigmoid if 1)
        if logits.shape[-1] in (1, 2):              # [B, F, C]
            if logits.shape[-1] == 1:
                return torch.sigmoid(logits[..., 0])
            return torch.softmax(logits, dim=-1)[..., 1]
        if logits.shape[1] in (1, 2):               # [B, C, F]
            if logits.shape[1] == 1:
                return torch.sigmoid(logits[:, 0, :])
            return torch.softmax(logits, dim=1)[:, 1, :]
        # fallback: assume last dim is frames, take sigmoid of mean over channels
        return torch.sigmoid(logits.mean(dim=1))

    def label(self, audio):
        """audio: [B, L] -> speech probs [B, K] torch float32, K = L//512."""
        import torch

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        audio = audio.to(torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        B, L = audio.shape
        K = L // CHUNK_SAMPLES
        out = np.zeros((B, K), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, B, self.sub_batch):
                wav = audio[i:i + self.sub_batch].to(self.device)
                lens = torch.full((wav.shape[0],), wav.shape[1], dtype=torch.int64,
                                  device=self.device)
                try:
                    logits = self.model(input_signal=wav, input_signal_length=lens)
                except TypeError:
                    logits = self.model.forward(input_signal=wav, input_signal_length=lens)
                speech = self._frame_speech(logits).cpu().numpy()   # [b, F]
                for j in range(speech.shape[0]):
                    out[i + j] = self._resample(speech[j], K)
        return torch.from_numpy(out)

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
        for idx in range(K):
            windows[:, idx] = padded[:, idx * CHUNK_SAMPLES:idx * CHUNK_SAMPLES + win]
        return windows
