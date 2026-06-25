"""WebRTC VAD baseline (standard weak baseline reported in VAD papers).

webrtcvad operates on 10/20/30 ms int16 frames and returns a hard speech/non-speech decision.
We run it on 30 ms frames and aggregate to ClearVAD's 32 ms chunk grid (fraction of speech
frames per chunk → a coarse [0,1] score so ROC/PR are at least defined).

Install: pip install webrtcvad
"""

from __future__ import annotations

import numpy as np

from clearvad import CHUNK_SAMPLES, SAMPLE_RATE

_FRAME = 480  # 30 ms @ 16 kHz


def webrtc_probs(audio: np.ndarray, sr: int = SAMPLE_RATE, mode: int = 2) -> np.ndarray:
    """audio: [L] float -> per-32ms-chunk speech score [K] in [0,1]."""
    import webrtcvad

    vad = webrtcvad.Vad(mode)                       # 0..3 (3 = most aggressive)
    x = (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
    n_frames = len(x) // _FRAME
    decisions = np.array([1.0 if vad.is_speech(x[i * _FRAME:(i + 1) * _FRAME].tobytes(), sr)
                          else 0.0 for i in range(n_frames)], dtype=np.float32)
    K = len(audio) // CHUNK_SAMPLES
    probs = np.zeros(K, dtype=np.float32)
    if n_frames == 0:
        return probs
    for k in range(K):
        f0 = (k * CHUNK_SAMPLES) // _FRAME
        f1 = max(f0 + 1, ((k + 1) * CHUNK_SAMPLES + _FRAME - 1) // _FRAME)
        seg = decisions[f0:min(f1, n_frames)]
        probs[k] = float(seg.mean()) if seg.size else float(decisions[min(f0, n_frames - 1)])
    return probs
