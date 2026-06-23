"""CPU latency benchmark for a ClearVAD ONNX binary (single-thread, ORT).

Measures per-chunk latency (mean/p50/p90/p99), throughput, real-time factor, and binary
size — the deployment numbers. State is carried across calls exactly as in production.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from clearvad import CHUNK_MS
from clearvad.export.validate_onnx import WIN, OrtVADRunner


def _rss_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 2)
    except Exception:  # noqa: BLE001
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return round(ru / 1e3, 2)
        except Exception:  # noqa: BLE001
            return None


def benchmark(onnx_path: str, warmup: int = 200, measure: int = 10000,
              threads: int = 1) -> Dict:
    runner = OrtVADRunner(onnx_path, threads=threads)
    rng = np.random.default_rng(0)
    windows = [(rng.standard_normal(WIN).astype(np.float32) * 0.1)
               for _ in range(warmup + measure)]

    for i in range(warmup):
        runner.step(windows[i])

    times: List[float] = []
    for i in range(measure):
        t0 = time.perf_counter()
        runner.step(windows[warmup + i])
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times)
    mean_ms = float(arr.mean())
    return {
        "model": os.path.basename(onnx_path),
        "size_mb": round(os.path.getsize(onnx_path) / 1e6, 4),
        "mean_ms": round(mean_ms, 5),
        "p50_ms": round(float(np.percentile(arr, 50)), 5),
        "p90_ms": round(float(np.percentile(arr, 90)), 5),
        "p99_ms": round(float(np.percentile(arr, 99)), 5),
        "throughput_chunks_per_s": round(1000.0 / mean_ms, 1),
        "rtf": round(mean_ms / CHUNK_MS, 6),
        "rss_mb": _rss_mb(),
        "threads": threads,
        "measure_chunks": measure,
    }
