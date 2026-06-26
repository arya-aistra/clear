"""Phase 7 — FastAPI serving for the INT8 ClearVAD-CfC ONNX binary.

Lean by design: depends only on numpy + onnxruntime + fastapi (+ soundfile for file upload) — no
torch, no clearvad model code — so the Docker image is small. The streaming contract is frozen:
  (chunk[1,576], state[1,2,128]) -> (prob[1,1], new_state[1,2,128])
state = 64-sample left context + 512-sample chunk @ 16 kHz; carried across chunks.

Endpoints:
  GET  /health            — liveness + model info
  POST /vad               — batch: JSON {audio:[float], sample_rate?, threshold?, min_speech_ms?,
                            min_silence_ms?} -> per-chunk probs + speech segments (seconds)
  POST /vad/file          — same but multipart wav/flac upload
  WS   /stream            — real-time: send raw 512-sample float32 chunks, receive prob per chunk
                            (per-connection state — ideal for voice agents)

Run: CLEARVAD_MODEL=dist/clearvad_lite.onnx uvicorn clearvad.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

# Frozen audio interface (project-wide constants; hardcoded to keep serving torch/clearvad-free).
SAMPLE_RATE = 16000
CHUNK = 512        # 32 ms
CONTEXT = 64       # left context
WIN = CONTEXT + CHUNK  # 576


def _linear_resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    n_out = int(round(x.size * sr_out / sr_in))
    xp = np.linspace(0.0, 1.0, x.size, dtype=np.float64)
    fp = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
    return np.interp(fp, xp, x).astype(np.float32)


def _segments(probs: np.ndarray, threshold: float, min_speech_ms: float,
              min_silence_ms: float) -> List[dict]:
    """Threshold + min-duration hysteresis → speech segments in seconds."""
    chunk_s = CHUNK / SAMPLE_RATE
    mask = probs >= threshold
    min_sp = max(1, int(round(min_speech_ms / 1000 / chunk_s)))
    min_si = max(1, int(round(min_silence_ms / 1000 / chunk_s)))
    # bridge short silence gaps
    segs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i + 1
            gap = 0
            while j < n and (mask[j] or gap < min_si):
                if mask[j]:
                    gap = 0
                else:
                    gap += 1
                j += 1
            end = j - gap
            if (end - i) >= min_sp:
                segs.append({"start_s": round(i * chunk_s, 3), "end_s": round(end * chunk_s, 3)})
            i = j
        else:
            i += 1
    return segs


class _Model:
    def __init__(self, path: str, threads: int = 1) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        st = self.sess.get_inputs()[1].shape  # 'state'
        self.state_shape = [d if isinstance(d, int) and d > 0 else 1 for d in st]
        self.path = path

    def new_state(self) -> np.ndarray:
        return np.zeros(self.state_shape, dtype=np.float32)

    def step(self, window: np.ndarray, state: np.ndarray):
        prob, new_state = self.sess.run(
            None, {"chunk": window.reshape(1, WIN).astype(np.float32), "state": state})
        return float(np.asarray(prob).reshape(-1)[0]), new_state.astype(np.float32)

    def stream(self, audio: np.ndarray) -> np.ndarray:
        """Per-call local state (concurrency-safe — does not mutate shared model state)."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        K = len(audio) // CHUNK
        state = self.new_state()
        prev = np.zeros(CONTEXT, dtype=np.float32)
        out = np.empty(K, dtype=np.float32)
        for i in range(K):
            chunk = audio[i * CHUNK:(i + 1) * CHUNK]
            out[i], state = self.step(np.concatenate([prev, chunk]), state)
            prev = chunk[-CONTEXT:]
        return out


# ---------------------------------------------------------------- FastAPI app
try:
    from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError("serving needs fastapi+pydantic: pip install -r requirements-serve.txt") from exc

app = FastAPI(title="ClearVAD-CfC", version="1.0",
              description="Compact INT8 continuous-time VAD (CfC core).")
MODEL: Optional[_Model] = None


class VADRequest(BaseModel):
    audio: List[float]
    sample_rate: int = SAMPLE_RATE
    threshold: float = 0.5
    min_speech_ms: float = 0.0
    min_silence_ms: float = 0.0


def _model() -> _Model:
    global MODEL
    if MODEL is None:
        path = os.environ.get("CLEARVAD_MODEL", "dist/clearvad_lite.onnx")
        MODEL = _Model(path, threads=int(os.environ.get("CLEARVAD_THREADS", "1")))
    return MODEL


@app.on_event("startup")
def _load() -> None:
    _model()


@app.get("/health")
def health() -> dict:
    m = _model()
    return {"status": "ok", "model": os.path.basename(m.path), "state_shape": m.state_shape,
            "sample_rate": SAMPLE_RATE, "chunk_ms": round(CHUNK / SAMPLE_RATE * 1000, 1)}


def _run(audio: np.ndarray, sr: int, threshold: float, min_speech_ms: float,
         min_silence_ms: float) -> dict:
    audio = _linear_resample(np.asarray(audio, dtype=np.float32).reshape(-1), sr)
    probs = _model().stream(audio)
    segs = _segments(probs, threshold, min_speech_ms, min_silence_ms)
    return {"num_chunks": int(len(probs)), "chunk_ms": round(CHUNK / SAMPLE_RATE * 1000, 1),
            "speech_prob": [round(float(p), 4) for p in probs], "segments": segs,
            "speech_ratio": round(float(np.mean(probs >= threshold)) if len(probs) else 0.0, 4)}


@app.post("/vad")
def vad(req: VADRequest) -> dict:
    return _run(np.asarray(req.audio, dtype=np.float32), req.sample_rate,
                req.threshold, req.min_speech_ms, req.min_silence_ms)


@app.post("/vad/file")
async def vad_file(file: UploadFile = File(...), threshold: float = 0.5,
                   min_speech_ms: float = 0.0, min_silence_ms: float = 0.0) -> dict:
    import io
    import soundfile as sf
    raw = await file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _run(audio, int(sr), threshold, min_speech_ms, min_silence_ms)


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """Real-time: client sends raw float32 little-endian 512-sample chunks; server replies
    {prob} per chunk, keeping per-connection state. Ideal for voice-agent endpointing."""
    await ws.accept()
    m = _model()
    state = m.new_state()
    prev = np.zeros(CONTEXT, dtype=np.float32)
    try:
        while True:
            buf = await ws.receive_bytes()
            chunk = np.frombuffer(buf, dtype=np.float32)
            if chunk.size < CHUNK:
                chunk = np.concatenate([chunk, np.zeros(CHUNK - chunk.size, np.float32)])
            chunk = chunk[:CHUNK]
            prob, state = m.step(np.concatenate([prev, chunk]), state)
            prev = chunk[-CONTEXT:]
            await ws.send_json({"prob": round(prob, 4)})
    except WebSocketDisconnect:
        return
