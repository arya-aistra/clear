#!/usr/bin/env python
"""Phase 0.1 — record the exact environment for reproducibility.

Writes ``environment.yaml`` (library versions, CUDA/GPU, ORT providers, CPU info).

Run:
    python scripts/log_environment.py
"""

from __future__ import annotations

import argparse
import platform
import sys
from importlib import import_module
from pathlib import Path

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clearvad.utils.config import save_yaml  # noqa: E402
from clearvad.utils.logging_utils import get_logger  # noqa: E402

LOG = get_logger("env")

PACKAGES = [
    "torch", "torchaudio", "numpy", "scipy", "onnx", "onnxruntime",
    "onnxconverter_common", "silero_vad", "fastapi", "uvicorn", "pydantic",
    "pytest", "hydra", "yaml", "soundfile",
]


def _version(modname: str):
    try:
        mod = import_module(modname)
        # str() coerces str-subclasses like torch's TorchVersion to a plain string.
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001
        return f"NOT INSTALLED ({type(exc).__name__})"


def _torch_info() -> dict:
    info = {}
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        cuda_v = getattr(torch.version, "cuda", None)
        info["cuda_version"] = str(cuda_v) if cuda_v is not None else None
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i)
                                 for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    return info


def _ort_info() -> dict:
    info = {}
    try:
        import onnxruntime as ort

        info["onnxruntime_version"] = ort.__version__
        info["available_providers"] = ort.get_available_providers()
        # Phase 0 requires the CPU build for honest latency claims.
        info["cpu_provider_present"] = "CPUExecutionProvider" in ort.get_available_providers()
        info["gpu_provider_present"] = any(
            p in ort.get_available_providers()
            for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        )
    except Exception as exc:  # noqa: BLE001
        info["onnxruntime_error"] = repr(exc)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="environment.yaml")
    args = ap.parse_args()

    env = {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {name: _version(name) for name in PACKAGES},
        "torch": _torch_info(),
        "onnxruntime": _ort_info(),
    }

    save_yaml(env, args.out)
    LOG.info("Wrote %s", args.out)
    LOG.info("torch.cuda_available=%s", env["torch"].get("cuda_available"))
    LOG.info("ORT providers=%s", env["onnxruntime"].get("available_providers"))
    if env["onnxruntime"].get("gpu_provider_present"):
        LOG.warning(
            "onnxruntime-gpu detected. ClearVAD latency claims are CPU-ONLY. "
            "Benchmarks force CPUExecutionProvider + single thread, but a CPU-only "
            "ORT build is preferred to avoid ambiguity."
        )


if __name__ == "__main__":
    main()
