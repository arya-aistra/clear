"""Benchmarking web app — test ClearVAD against other VADs live, with realtime metrics + a session
leaderboard. Heavier than the lean serving app (loads torch + competitor models); intended for the
demo/benchmark UI, not edge deployment.

  GET  /              -> the single-page UI
  GET  /models        -> which models are available (+ why any are not)
  POST /analyze       -> run selected models on an uploaded wav: per-model speech track, segments,
                         per-chunk latency, real-time factor, speech ratio, agreement-with-consensus
  POST /benchmark     -> run selected models on a labeled eval cache -> accuracy leaderboard

Run: uvicorn clearvad.serving.benchmark_app:app --port 8100
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from clearvad.serving.app import (CHUNK, CONTEXT, SAMPLE_RATE, _linear_resample,
                                  _Model, _segments)

app = FastAPI(title="ClearVAD Benchmark")
_FNS: Dict[str, object] = {}
_NOTES: Dict[str, str] = {}
_CLEARVAD: object = None
_STATIC = Path(__file__).parent / "static" / "index.html"
_SAMPLES = Path(__file__).parent / "static" / "samples"


def _load_models() -> None:
    global _CLEARVAD
    if _FNS or _NOTES:
        return
    try:                                                   # ClearVAD via its INT8 ONNX binary
        m = _Model(os.environ.get("CLEARVAD_MODEL", "dist/clearvad_lite.onnx"))
        _CLEARVAD = m                                      # kept for per-chunk WS streaming
        _FNS["clearvad"] = m.stream
    except Exception as exc:  # noqa: BLE001
        _NOTES["clearvad"] = repr(exc)
    try:
        from clearvad.model.silero_compat import SileroVAD
        sv = SileroVAD(onnx=False)
        _FNS["silero"] = lambda a, sv=sv: sv.probabilities(a, reset=True)
    except Exception as exc:  # noqa: BLE001
        _NOTES["silero"] = repr(exc)
    try:
        from clearvad.evaluation.webrtc_baseline import webrtc_probs
        _FNS["webrtc"] = lambda a: webrtc_probs(a)
    except Exception as exc:  # noqa: BLE001
        _NOTES["webrtc"] = repr(exc)
    try:
        from clearvad.evaluation.external_vads import NeMoVAD
        nv = NeMoVAD(); nv.load()
        _FNS["nemo"] = lambda a, nv=nv: nv.probs(a)
    except Exception as exc:  # noqa: BLE001
        _NOTES["nemo"] = repr(exc)


@app.on_event("startup")
def _startup() -> None:
    _load_models()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _STATIC.read_text(encoding="utf-8")


@app.get("/models")
def models() -> dict:
    _load_models()
    return {"available": list(_FNS), "unavailable": _NOTES}


def _downsample(x: np.ndarray, n: int = 800) -> List[float]:
    if len(x) <= n:
        return [round(float(v), 4) for v in x]
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return [round(float(v), 4) for v in np.asarray(x)[idx]]


def _run_one(name: str, audio: np.ndarray) -> dict:
    fn = _FNS[name]
    t0 = time.perf_counter()
    probs = np.asarray(fn(audio), dtype=np.float32).reshape(-1)
    elapsed = time.perf_counter() - t0
    K = max(len(probs), 1)
    dur = len(audio) / SAMPLE_RATE
    return {"probs": probs, "n_chunks": int(len(probs)),
            "latency_ms_per_chunk": round(elapsed / K * 1000, 4),
            "rtf": round(elapsed / max(dur, 1e-9), 5),
            "compute_ms": round(elapsed * 1000, 2)}


def _decode(raw_bytes: bytes) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _linear_resample(np.asarray(audio, np.float32), int(sr))


def _analyze_core(audio: np.ndarray, models: str, threshold: float, min_speech_ms: float,
                  min_silence_ms: float, filename: str) -> dict:
    chosen = [m for m in models.split(",") if m in _FNS]
    raw = {m: _run_one(m, audio) for m in chosen}
    K = min((r["n_chunks"] for r in raw.values()), default=0)
    # consensus = majority speech vote per chunk across the chosen models
    preds = {m: (raw[m]["probs"][:K] >= threshold).astype(np.int8) for m in chosen}
    consensus = (np.sum(list(preds.values()), axis=0) >= (len(chosen) / 2.0)) if chosen and K else \
        np.zeros(K, np.int8)

    results = {}
    for m in chosen:
        r = raw[m]
        agree = float(np.mean(preds[m] == consensus)) if K else float("nan")
        results[m] = {
            "track": _downsample(r["probs"]),
            "segments": _segments(r["probs"], threshold, min_speech_ms, min_silence_ms),
            "latency_ms_per_chunk": r["latency_ms_per_chunk"], "rtf": r["rtf"],
            "compute_ms": r["compute_ms"], "n_chunks": r["n_chunks"],
            "speech_ratio": round(float(np.mean(r["probs"] >= threshold)) if r["n_chunks"] else 0.0, 4),
            "agreement": round(agree, 4),
        }
    return {"filename": filename, "duration_s": round(len(audio) / SAMPLE_RATE, 2),
            "chunk_ms": round(CHUNK / SAMPLE_RATE * 1000, 1), "models": results}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), models: str = Form("clearvad"),
                  threshold: float = Form(0.5), min_speech_ms: float = Form(100.0),
                  min_silence_ms: float = Form(100.0)) -> dict:
    _load_models()
    audio = _decode(await file.read())
    return _analyze_core(audio, models, threshold, min_speech_ms, min_silence_ms, file.filename)


@app.get("/samples")
def samples() -> dict:
    """Bundled demo wavs (zero-upload)."""
    files = sorted(_SAMPLES.glob("*.wav")) if _SAMPLES.exists() else []
    return {"samples": [f.name for f in files]}


@app.get("/samples/{name}")
def sample_file(name: str):
    p = _SAMPLES / Path(name).name
    return FileResponse(str(p)) if p.exists() else HTMLResponse("not found", status_code=404)


@app.post("/analyze_sample")
def analyze_sample(name: str = Form(...), models: str = Form("clearvad"),
                   threshold: float = Form(0.5), min_speech_ms: float = Form(100.0),
                   min_silence_ms: float = Form(100.0)) -> dict:
    _load_models()
    p = _SAMPLES / Path(name).name
    if not p.exists():
        return {"error": f"sample not found: {name}"}
    return _analyze_core(_decode(p.read_bytes()), models, threshold, min_speech_ms,
                         min_silence_ms, name)


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """Real-time ClearVAD endpointing. Client sends raw float32 little-endian 512-sample (32 ms)
    chunks @ 16 kHz; server replies {prob} per chunk with carried state (per connection)."""
    await ws.accept()
    _load_models()
    m = _CLEARVAD
    if m is None:
        await ws.send_json({"error": "clearvad model not loaded"})
        await ws.close()
        return
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
            await ws.send_json({"prob": round(float(prob), 4)})
    except WebSocketDisconnect:
        return


@app.post("/benchmark")
def benchmark(cache: str = Form("data/eval/aligned_eval_pad40.npz"),
              models: str = Form("clearvad,silero,webrtc,nemo"),
              threshold: float = Form(0.5)) -> dict:
    """Accuracy leaderboard on a labeled frame-accurate cache."""
    _load_models()
    from clearvad.evaluation.metrics import pr_auc, roc_auc, summarize, tpr_at_fpr
    if not Path(cache).exists():
        return {"error": f"cache not found: {cache}"}
    data = np.load(cache, allow_pickle=True)
    audio, labels = data["audio"], data["labels"]
    chosen = [m for m in models.split(",") if m in _FNS]
    out = {}
    for m in chosen:
        fn = _FNS[m]
        per, pp, pl = [], [], []
        for a, lab in zip(audio, labels):
            pr = np.asarray(fn(np.asarray(a, np.float32)), np.float32).reshape(-1)
            K = min(len(pr), len(lab))
            per.append(summarize(pr[:K], np.asarray(lab[:K]), threshold=threshold))
            pp.append(pr[:K]); pl.append(np.asarray(lab[:K]))
        P, L = np.concatenate(pp), np.concatenate(pl).astype(bool)
        out[m] = {"auroc": round(roc_auc(P, L), 4), "pr_auc": round(pr_auc(P, L), 4),
                  "f1": round(float(np.mean([d["f1"] for d in per])), 4),
                  "far": round(float(np.mean([d["far"] for d in per])), 4),
                  "mr": round(float(np.mean([d["mr"] for d in per])), 4),
                  "tpr_at_fpr0.315": round(tpr_at_fpr(P, L, 0.315), 4)}
    return {"cache": cache, "n_sequences": int(len(audio)), "models": out}
