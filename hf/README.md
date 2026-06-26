---
license: mit
language:
- en
library_name: onnxruntime
tags:
- voice-activity-detection
- vad
- speech
- onnx
- int8
- streaming
- edge
- cfc
- continuous-time
---

# ClearVAD-CfC — a tiny, INT8, continuous-time Voice Activity Detector

ClearVAD is a compact streaming **voice activity detector** with a novel **closed-form
continuous-time (CfC)** temporal core (Hasani et al., 2022) — an architecture essentially
untouched for VAD. It is trained with **no human VAD labels** (forced-alignment targets) and ships
as a single **INT8 ONNX** binary (~0.46 MB) for CPU/edge serving.

On a frame-accurate evaluation (forced-aligned labels, scored identically for every model) ClearVAD
**matches the strongest open frame-level VADs on accuracy** and has the **lowest false-alarm rate and
best short-pause sensitivity** of the models tested, while being the **only one that runs in INT8**.

## Files

| file | precision | size | use |
|------|-----------|------|-----|
| `clearvad_lite.onnx` | INT8 | ~0.46 MB | **recommended** — smallest, edge/CPU |
| `clearvad_fp16.onnx` | FP16 | ~0.65 MB | half precision |
| `clearvad_base.onnx` | FP32 | ~1.24 MB | reference / highest fidelity |

## Streaming contract (frozen)

```
inputs : chunk [B, 576] float32   (64-sample left context + 512-sample chunk @ 16 kHz)
         state [B, 2, 128] float32 (recurrent state; zeros to start)
outputs: prob  [B, 1] float32     (speech probability in (0,1) for the 32 ms chunk)
         new_state [B, 2, 128]    (carry into the next chunk)
```
Audio is 16 kHz mono. Each chunk is 512 samples (32 ms); the 64-sample left context comes from the
tail of the previous chunk. Streaming is exactly equal to offline processing.

## Usage (ONNX Runtime, no PyTorch)

```python
import numpy as np, onnxruntime as ort, soundfile as sf

CHUNK, CTX, SR = 512, 64, 16000
sess = ort.InferenceSession("clearvad_lite.onnx", providers=["CPUExecutionProvider"])
state = np.zeros([1, 2, 128], dtype=np.float32)         # see model 'state' input shape

audio, sr = sf.read("speech.wav", dtype="float32")       # resample to 16k mono beforehand
if audio.ndim > 1: audio = audio.mean(1)

probs, prev = [], np.zeros(CTX, np.float32)
for i in range(len(audio) // CHUNK):
    chunk = audio[i*CHUNK:(i+1)*CHUNK]
    window = np.concatenate([prev, chunk]).reshape(1, CHUNK + CTX).astype(np.float32)
    prob, state = sess.run(None, {"chunk": window, "state": state})
    probs.append(float(prob.reshape(-1)[0]))
    prev = chunk[-CTX:]
probs = np.array(probs)                                  # one speech probability per 32 ms
speech = probs >= 0.5                                    # threshold to taste
```

A complete script with segment extraction (threshold + min-speech/min-silence) is in
`inference_example.py`.

## Benchmark (frame-accurate, forced-aligned labels; identical for every model)

**Clean:**

| metric | **ClearVAD** | Silero v5 | NeMo Frame-VAD | WebRTC |
|--------|--------------|-----------|----------------|--------|
| AUROC | 0.968 | 0.972 | 0.972 | 0.776 |
| F1 | 0.958 | 0.958 | 0.951 | 0.922 |
| False-alarm rate | **0.130** | 0.224 | 0.315 | 0.516 |
| Short-pause @256 ms | **1.00** | 0.88 | 0.19 | 0.51 |
| Params / INT8 | 303k / ✅ 0.46 MB | 309k / ❌ | CNN / ❌ | — |

**Noisy (held-out DEMAND @ 0–12 dB):** AUROC — Silero 0.970, NeMo 0.966, ClearVAD 0.947, WebRTC 0.882.

**Honest framing:** Silero and NeMo are the accuracy co-leaders (~0.97 AUROC); ClearVAD is competitive
(within ~0.4 pt on clean, F1 level with Silero), has the **lowest false-alarm rate and best
short-pause detection** of the set, and is the **only ~300k-param, INT8-deployable** model. We do
**not** claim accuracy superiority over Silero/NeMo. INT8 costs 1.29 pp F1 vs FP32.

## Training (no human labels)

Frame-accurate targets via torchaudio **MMS_FA forced alignment** of LibriSpeech transcripts
(intra-word/leading/trailing silence = non-speech; 100 ms min-silence smoothing, 40 ms edge padding).
~20 h of speech, mixed with real noise (MUSAN, ESC-50) at 0–20 dB SNR, augmented with real room
impulse responses (OpenSLR SLR28), mu-law codec, and gain. Pure-supervised on the alignment labels.

## Limitations

- Silero/NeMo edge AUROC; ClearVAD matches on clean and is near-parity (~2 pt) on noisy.
- Labels are forced-alignment-derived (English read speech); a fully human-labeled benchmark
  (AVA-Speech) is the recommended external validation.
- 16 kHz mono only; resample other rates beforehand.

## License & citation

MIT. Architecture inspired by Hasani et al., *Closed-form continuous-time neural networks* (2022).
Training/eval methodology and the full multi-model comparison: see the project repository.
