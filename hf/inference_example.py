#!/usr/bin/env python
"""Standalone ClearVAD-CfC inference (ONNX Runtime only — no PyTorch, no clearvad package).

Copy this next to clearvad_lite.onnx and run:
    pip install onnxruntime numpy soundfile
    python inference_example.py speech.wav --threshold 0.5 --min-speech-ms 100 --min-silence-ms 100
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import soundfile as sf

SR, CHUNK, CTX = 16000, 512, 64
WIN = CHUNK + CTX  # 576


def resample_linear(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return x.astype(np.float32)
    n = int(round(len(x) * SR / sr_in))
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)


def stream(sess: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    state_shape = [d if isinstance(d, int) and d > 0 else 1 for d in sess.get_inputs()[1].shape]
    state = np.zeros(state_shape, dtype=np.float32)
    prev = np.zeros(CTX, dtype=np.float32)
    probs = []
    for i in range(len(audio) // CHUNK):
        chunk = audio[i * CHUNK:(i + 1) * CHUNK]
        window = np.concatenate([prev, chunk]).reshape(1, WIN).astype(np.float32)
        prob, state = sess.run(None, {"chunk": window, "state": state})
        probs.append(float(np.asarray(prob).reshape(-1)[0]))
        prev = chunk[-CTX:]
    return np.asarray(probs, dtype=np.float32)


def segments(probs: np.ndarray, threshold: float, min_speech_ms: float, min_silence_ms: float):
    chunk_s = CHUNK / SR
    mask = probs >= threshold
    min_sp = max(1, round(min_speech_ms / 1000 / chunk_s))
    min_si = max(1, round(min_silence_ms / 1000 / chunk_s))
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j, gap = i + 1, 0
            while j < n and (mask[j] or gap < min_si):
                gap = 0 if mask[j] else gap + 1
                j += 1
            end = j - gap
            if end - i >= min_sp:
                out.append((round(i * chunk_s, 3), round(end * chunk_s, 3)))
            i = j
        else:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="clearvad_lite.onnx")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-speech-ms", type=float, default=100.0)
    ap.add_argument("--min-silence-ms", type=float, default=100.0)
    args = ap.parse_args()

    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    audio = resample_linear(audio, sr)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    probs = stream(sess, audio)
    segs = segments(probs, args.threshold, args.min_speech_ms, args.min_silence_ms)

    print(f"{len(probs)} chunks @ 32 ms | speech ratio {float((probs >= args.threshold).mean()):.3f}")
    print(f"{len(segs)} speech segments (s):")
    for s, e in segs:
        print(f"  {s:7.2f} - {e:7.2f}  ({e - s:.2f}s)")


if __name__ == "__main__":
    main()
