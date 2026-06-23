"""Shared ONNX export helper for tests (not collected by pytest)."""

import io


def export_to_bytes(model, args, **kw):
    """Export a torch module to ONNX bytes, preferring the legacy exporter.

    torch>=2.x defaults to the dynamo exporter (needs onnxscript); the legacy exporter
    gives predictable unrolling of explicit loops and exact opset control. Try legacy
    (dynamo=False) first, fall back to dynamo/default.
    """
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
