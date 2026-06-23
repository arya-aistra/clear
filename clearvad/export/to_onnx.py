"""FP32 ONNX export of the streaming ClearVAD model.

Exports the per-chunk deployment graph:
    (chunk[B, 576], state[B, d_inner, d_state]) -> (prob[B, 1], new_state[B, d_inner, d_state])

The G-SSM runs in 'loop' scan mode (default) so the T_enc=3 recurrence unrolls into static
ops — opset-17, no Scan/Loop. State is an explicit graph input/output, carried externally by
the serving layer. Prefers the legacy TorchScript exporter (predictable unrolling); falls back
to the dynamo exporter.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from clearvad import TOTAL_INPUT_SAMPLES


def _export_bytes(model, args, **kw) -> bytes:
    import torch
    try:
        buf = io.BytesIO()
        torch.onnx.export(model, args, buf, dynamo=False, **kw)
        buf.seek(0)
        return buf.read()
    except TypeError:
        pass
    except Exception:
        pass
    buf = io.BytesIO()
    try:
        torch.onnx.export(model, args, buf, dynamo=True, **kw)
    except TypeError:
        buf = io.BytesIO()
        torch.onnx.export(model, args, buf, **kw)
    buf.seek(0)
    return buf.read()


def export_fp32(model, out_path: str, opset: int = 17, dynamic_batch: bool = True,
                input_samples: int = TOTAL_INPUT_SAMPLES) -> str:
    """Export the streaming model to an FP32 ONNX file. Returns the path."""
    import torch

    model = model.eval()
    chunk = torch.zeros(1, input_samples, dtype=torch.float32)
    state = model.reset_state(1)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"chunk": {0: "batch"}, "state": {0: "batch"},
                        "prob": {0: "batch"}, "new_state": {0: "batch"}}
    model_bytes = _export_bytes(
        model, (chunk, state),
        input_names=["chunk", "state"], output_names=["prob", "new_state"],
        dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(model_bytes)
    return out_path


def onnx_size_mb(path: str) -> float:
    import os
    return round(os.path.getsize(path) / 1e6, 4)
