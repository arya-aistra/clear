"""Hysteresis VAD post-processor (streaming-canonical, offline = the same loop).

Asymmetric thresholds (onset > offset) + a min-silence/pad end-holdoff convert raw per-chunk
probabilities into stable speech/silence decisions and control the endpoint operating point:

  * onset_threshold  : prob to declare speech START (immediate).
  * offset_threshold : prob below which we begin counting silence (hysteresis: < onset).
  * min_silence_ms   : silence must persist this long before we declare speech END.
  * speech_pad_ms    : extra tail kept as speech (avoids clipping); adds to the end-holdoff.
  * min_speech_ms    : (offline segment cleaner) drop speech segments shorter than this.

Endpoint latency ≈ raw-decay + (min_silence + speech_pad) frames → a *low-latency* profile
uses a high offset + small min_silence + zero pad; a *high-precision* profile uses a larger
hysteresis gap + pad. The streaming `step` is causal; `process` just runs it over an array, so
chunked == offline by construction (verified in tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from clearvad import CHUNK_MS
from clearvad.evaluation.metrics import frames_to_segments

Segment = Tuple[int, int]


def frames_for_ms(ms: float, chunk_ms: float = CHUNK_MS) -> int:
    return int(round(ms / chunk_ms))


@dataclass
class SmootherOutput:
    is_speech: bool
    speech_started: bool
    speech_ended: bool


class HysteresisSmoother:
    def __init__(
        self,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.35,
        min_speech_ms: float = 100.0,
        min_silence_ms: float = 50.0,
        speech_pad_ms: float = 30.0,
        chunk_ms: float = CHUNK_MS,
    ) -> None:
        if offset_threshold > onset_threshold:
            raise ValueError("offset_threshold must be <= onset_threshold")
        self.onset = float(onset_threshold)
        self.offset = float(offset_threshold)
        self.chunk_ms = chunk_ms
        self.min_speech_frames = frames_for_ms(min_speech_ms, chunk_ms)
        self.min_silence_frames = frames_for_ms(min_silence_ms, chunk_ms)
        self.pad_frames = frames_for_ms(speech_pad_ms, chunk_ms)
        self.reset()

    def reset(self) -> None:
        self._triggered = False
        self._silence_run = 0

    # ---------------------------------------------------------------- streaming
    def step(self, prob: float) -> SmootherOutput:
        """Causal per-chunk update. Returns is_speech + start/end events for this frame."""
        started = ended = False
        end_holdoff = self.min_silence_frames + self.pad_frames
        if not self._triggered:
            if prob >= self.onset:
                self._triggered = True
                self._silence_run = 0
                started = True
        else:
            if prob < self.offset:
                self._silence_run += 1
                if self._silence_run >= end_holdoff:
                    self._triggered = False
                    ended = True
            else:
                self._silence_run = 0
        return SmootherOutput(is_speech=self._triggered, speech_started=started,
                              speech_ended=ended)

    # ---------------------------------------------------------------- offline
    def process(self, probs, reset: bool = True) -> np.ndarray:
        """Run the streaming smoother over a prob array → post-processed speech mask [T]."""
        if reset:
            self.reset()
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        mask = np.zeros(len(probs), dtype=bool)
        for i, p in enumerate(probs):
            mask[i] = self.step(float(p)).is_speech
        return mask

    def segments(self, probs, apply_min_speech: bool = True) -> List[Segment]:
        """Post-processed speech segments [start, end). Optionally drops sub-min_speech ones."""
        mask = self.process(probs)
        segs = [list(s) for s in frames_to_segments(mask)]
        if apply_min_speech and self.min_speech_frames > 1:
            segs = [s for s in segs if (s[1] - s[0]) >= self.min_speech_frames]
        return [(s[0], s[1]) for s in segs]

    def segments_seconds(self, probs) -> List[dict]:
        """Segments as {start_ms, end_ms} for the file API."""
        out = []
        for (s, e) in self.segments(probs):
            out.append({"start_ms": round(s * self.chunk_ms, 1),
                        "end_ms": round(e * self.chunk_ms, 1)})
        return out

    @classmethod
    def from_config(cls, cfg: dict, chunk_ms: float = CHUNK_MS) -> "HysteresisSmoother":
        return cls(
            onset_threshold=cfg.get("onset_threshold", 0.5),
            offset_threshold=cfg.get("offset_threshold", 0.35),
            min_speech_ms=cfg.get("min_speech_ms", 100.0),
            min_silence_ms=cfg.get("min_silence_ms", 50.0),
            speech_pad_ms=cfg.get("speech_pad_ms", 30.0),
            chunk_ms=chunk_ms,
        )
