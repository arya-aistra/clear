#!/usr/bin/env python
"""Phase 0.4 — benchmark Silero VAD v5 as the CPU baseline.

Measures (single-thread, ONNX Runtime CPU):
  * end-to-end per-chunk latency (the high-level streaming call: context concat + ORT)
  * isolated pure-ORT session.run latency (inference only)
  * mean / p50 / p90 / p99 latency, throughput (chunks/s), real-time factor
  * parameter count, model size on disk, state (streaming) overhead bytes
  * process RSS (best-effort)

Writes reports/phase0_baseline.json. This file is the FROZEN baseline every later phase
compares against.

Run:
    python scripts/benchmark_silero.py --measure-chunks 10000 --warmup 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clearvad import CHUNK_MS, CHUNK_SAMPLES, SAMPLE_RATE  # noqa: E402
from clearvad.model.silero_compat import SileroVAD  # noqa: E402
from clearvad.utils.logging_utils import get_logger, write_json  # noqa: E402
from scripts.dissect_silero import find_silero_onnx_path  # noqa: E402

LOG = get_logger("bench")


# ----------------------------------------------------------------- memory / stats
def rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes
        return ru / 1e3 if sys.platform != "darwin" else ru / 1e6
    except Exception:  # noqa: BLE001
        return None


def latency_stats(times_ms: List[float], percentiles=(50, 90, 99)) -> Dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    out = {
        "mean_ms": round(float(arr.mean()), 5),
        "std_ms": round(float(arr.std()), 5),
        "min_ms": round(float(arr.min()), 5),
        "max_ms": round(float(arr.max()), 5),
    }
    for p in percentiles:
        out[f"p{p}_ms"] = round(float(np.percentile(arr, p)), 5)
    return out


# ----------------------------------------------------------------- end-to-end
def bench_end_to_end(warmup: int, measure: int) -> Dict[str, Any]:
    """High-level streaming call latency (what production pays per chunk)."""
    vad = SileroVAD(onnx=True)
    vad.reset_states(1)
    rng = np.random.default_rng(0)
    chunks = [rng.standard_normal(CHUNK_SAMPLES).astype(np.float32) * 0.05
              for _ in range(max(warmup, 1) + measure)]

    for i in range(warmup):
        vad.forward(chunks[i])

    times: List[float] = []
    for i in range(measure):
        c = chunks[warmup + i]
        t0 = time.perf_counter()
        vad.forward(c)
        times.append((time.perf_counter() - t0) * 1000.0)

    stats = latency_stats(times)
    stats["throughput_chunks_per_s"] = round(1000.0 / stats["mean_ms"], 1)
    stats["rtf"] = round(stats["mean_ms"] / CHUNK_MS, 6)
    return stats


# ----------------------------------------------------------------- pure ORT
_ORT_NP = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16,
    "tensor(double)": np.float64, "tensor(int64)": np.int64,
    "tensor(int32)": np.int32, "tensor(int8)": np.int8,
}


def _np_dtype(ort_type: str):
    return _ORT_NP.get(ort_type, np.float32)


def _resolve_shape(shape, role: str, audio_len: int) -> List[int]:
    """Replace dynamic dims with concrete sizes based on the input's role."""
    out = []
    dyn_idx = [i for i, d in enumerate(shape) if not isinstance(d, int) or d < 0]
    last_dyn = dyn_idx[-1] if dyn_idx else None
    for i, d in enumerate(shape):
        if isinstance(d, int) and d > 0:
            out.append(d)
        elif role == "audio" and i == last_dyn:
            out.append(audio_len)
        else:
            out.append(1)
    return out


def _classify(inp) -> str:
    name = inp.name.lower()
    dt = _np_dtype(inp.type)
    if "sr" in name or dt in (np.int64, np.int32):
        return "sr"
    if "state" in name or "context" in name or (128 in [d for d in inp.shape if isinstance(d, int)]):
        return "state"
    return "audio"


def bench_pure_ort(onnx_path: str, warmup: int, measure: int) -> Dict[str, Any]:
    """Isolated single-thread ORT session.run latency, carrying state across calls."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])

    in_meta = sess.get_inputs()
    out_meta = sess.get_outputs()
    roles = {m.name: _classify(m) for m in in_meta}

    # find which output feeds back into the state input
    state_in = next((m for m in in_meta if roles[m.name] == "state"), None)
    state_out_name = None
    for o in out_meta:
        on = o.name.lower()
        if "state" in on:
            state_out_name = o.name
            break
    if state_out_name is None and state_in is not None and len(out_meta) > 1:
        state_out_name = out_meta[-1].name  # convention: stateN is last output

    def build_feeds(audio_len: int, state_val) -> Dict[str, np.ndarray]:
        feeds: Dict[str, np.ndarray] = {}
        for m in in_meta:
            role = roles[m.name]
            dt = _np_dtype(m.type)
            if role == "sr":
                feeds[m.name] = np.array(SAMPLE_RATE, dtype=dt)
            elif role == "state":
                feeds[m.name] = (state_val if state_val is not None
                                 else np.zeros(_resolve_shape(m.shape, "state", audio_len), dtype=dt))
            else:
                feeds[m.name] = np.zeros(_resolve_shape(m.shape, "audio", audio_len), dtype=dt)
        return feeds

    # determine accepted audio length: try 512 then 576
    audio_len = CHUNK_SAMPLES
    state_val = None
    for candidate in (CHUNK_SAMPLES, CHUNK_SAMPLES + 64):
        try:
            _ = sess.run(None, build_feeds(candidate, None))
            audio_len = candidate
            break
        except Exception as exc:  # noqa: BLE001
            LOG.warning("pure-ORT audio_len=%d failed: %r", candidate, exc)
    else:
        raise RuntimeError("Could not find a working audio input length for pure-ORT bench.")

    out_names = [o.name for o in out_meta]
    state_out_idx = out_names.index(state_out_name) if state_out_name in out_names else None

    def step(feeds):
        outs = sess.run(None, feeds)
        new_state = outs[state_out_idx] if state_out_idx is not None else None
        return new_state

    feeds = build_feeds(audio_len, None)
    state_overhead_bytes = None
    if state_in is not None:
        sv = feeds[state_in.name]
        state_overhead_bytes = int(sv.nbytes)

    for _ in range(warmup):
        state_val = step(feeds)
        if state_in is not None and state_val is not None:
            feeds[state_in.name] = state_val.astype(_np_dtype(state_in.type))

    times: List[float] = []
    for _ in range(measure):
        t0 = time.perf_counter()
        state_val = step(feeds)
        times.append((time.perf_counter() - t0) * 1000.0)
        if state_in is not None and state_val is not None:
            feeds[state_in.name] = state_val.astype(_np_dtype(state_in.type))

    stats = latency_stats(times)
    stats["throughput_chunks_per_s"] = round(1000.0 / stats["mean_ms"], 1)
    stats["rtf"] = round(stats["mean_ms"] / CHUNK_MS, 6)
    stats["audio_input_len"] = audio_len
    stats["state_overhead_bytes"] = state_overhead_bytes
    return stats


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--measure-chunks", type=int, default=10000)
    ap.add_argument("--out", default="reports/phase0_baseline.json")
    args = ap.parse_args()

    rss_start = rss_mb()
    result: Dict[str, Any] = {
        "model": "silero_vad_v5",
        "config": {
            "sample_rate": SAMPLE_RATE, "chunk_samples": CHUNK_SAMPLES,
            "chunk_ms": CHUNK_MS, "warmup": args.warmup,
            "measure_chunks": args.measure_chunks, "threads": 1,
            "provider": "CPUExecutionProvider",
        },
    }

    LOG.info("Param count + model size...")
    vad_jit = SileroVAD(onnx=False)
    result["jit_param_count"] = vad_jit.parameter_count()
    onnx_path = find_silero_onnx_path()
    if onnx_path and os.path.exists(onnx_path):
        result["onnx_path"] = onnx_path
        result["onnx_size_mb"] = round(os.path.getsize(onnx_path) / 1e6, 4)

    LOG.info("End-to-end streaming latency (%d chunks)...", args.measure_chunks)
    result["latency_end_to_end"] = bench_end_to_end(args.warmup, args.measure_chunks)

    if onnx_path:
        LOG.info("Pure-ORT inference latency...")
        try:
            result["latency_pure_ort"] = bench_pure_ort(onnx_path, args.warmup, args.measure_chunks)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("pure-ORT bench failed: %r", exc)
            result["latency_pure_ort_error"] = repr(exc)

    rss_end = rss_mb()
    result["memory_rss_mb"] = {"start": rss_start, "end": rss_end}

    write_json(result, args.out)
    LOG.info("Wrote %s", args.out)
    e2e = result["latency_end_to_end"]
    LOG.info("END-TO-END: mean=%.4f ms  p99=%.4f ms  RTF=%.4f",
             e2e["mean_ms"], e2e.get("p99_ms", float("nan")), e2e["rtf"])
    if "latency_pure_ort" in result:
        po = result["latency_pure_ort"]
        LOG.info("PURE-ORT : mean=%.4f ms  p99=%.4f ms", po["mean_ms"], po.get("p99_ms"))


if __name__ == "__main__":
    main()
