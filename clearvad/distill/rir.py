"""Real room impulse responses (OpenSLR SLR28, RIRS_NOISES) for far-field / reverberant robustness.

Synthetic reverb is a crude proxy; convolving with REAL measured/simulated RIRs is what teaches a
VAD to stay robust to room acoustics and microphone distance — a major real-deployment factor the
MUSAN-only model never saw. Same acquisition pattern as MUSAN (torch.hub download + zip extract).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from clearvad import SAMPLE_RATE
from clearvad.utils.audio import load_audio
from clearvad.utils.logging_utils import get_logger

LOG = get_logger("rir")
RIR_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"


class RIRSource:
    def __init__(self, source: str = "openslr", local_dir: Optional[str] = None,
                 root: str = "data/rirs", sample_rate: int = SAMPLE_RATE,
                 max_files: int = 600, max_rir_seconds: float = 2.0, seed: int = 7) -> None:
        self.sr = sample_rate
        max_len = int(max_rir_seconds * sample_rate)
        base = Path(local_dir) if local_dir else self._ensure(root)
        self.rirs = self._load(base, max_files, max_len, seed)
        if not self.rirs:
            raise RuntimeError(f"No RIRs loaded from {base}")
        LOG.info("RIRSource: %d impulse responses from %s", len(self.rirs), local_dir or source)

    def _ensure(self, root: str) -> Path:
        import torch
        root_p = Path(root)
        out = root_p / "RIRS_NOISES"
        if out.exists():
            return out
        root_p.mkdir(parents=True, exist_ok=True)
        zip_path = root_p / "rirs_noises.zip"
        if not zip_path.exists():
            LOG.info("Downloading RIRS_NOISES (SLR28) from %s ...", RIR_URL)
            torch.hub.download_url_to_file(RIR_URL, str(zip_path), progress=True)
        LOG.info("Extracting RIRS_NOISES ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root_p)
        return out

    def _load(self, base: Path, max_files: int, max_len: int, seed: int) -> List[np.ndarray]:
        # prefer the clean simulated RIRs; they are short impulses (not the long isotropic noises)
        sim = base / "simulated_rirs"
        files = sorted((sim if sim.exists() else base).rglob("*.wav"))
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        out: List[np.ndarray] = []
        for f in files:
            try:
                a = load_audio(f, self.sr)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("skip rir %s: %r", f, exc)
                continue
            if len(a) == 0 or len(a) > max_len:        # skip long files (likely noise, not a RIR)
                continue
            n = float(np.sqrt(np.sum(a * a)) + 1e-9)
            out.append((a / n).astype(np.float32))
            if len(out) >= max_files:
                break
        return out

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self.rirs[int(rng.integers(len(self.rirs)))]
