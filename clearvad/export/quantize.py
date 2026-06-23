"""FP16 conversion + INT8 static quantization of the ClearVAD ONNX graph.

INT8 strategy (the Silero-failure lesson from Phase 0): quantize ONLY Conv/Gemm/MatMul
(the frontend conv, encoder convs, head conv, and the G-SSM dense projections). The G-SSM
recurrence ops (Exp/Mul/Add/ReduceSum/Softplus/Sigmoid) are left in FP32 for numeric
stability — these are exactly the kind of ops whose naive quantization broke Silero's INT8.

Static quantization needs calibration data: representative (chunk, state) inputs. We collect
them by streaming synthetic audio through the FP32 torch model (states carried as in
deployment), so activation ranges match real streaming.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from clearvad import CHUNK_SAMPLES, CONTEXT_SAMPLES, TOTAL_INPUT_SAMPLES

QUANT_OP_TYPES = ["Conv", "Gemm", "MatMul"]


# --------------------------------------------------------------------- FP16
# Keep the G-SSM recurrence + shape ops in FP32 so the converter inserts casts cleanly
# (mixed fp16/fp32 around these ops otherwise yields type-mismatch errors at load).
_FP16_BLOCK = ["Unsqueeze", "Slice", "Gather", "Shape", "ReduceSum", "ReduceMean",
               "Softplus", "Range", "ConstantOfShape", "Pad"]


def export_fp16(fp32_path: str, out_path: str) -> str:
    """Best-effort FP16 conversion. Validates the result loads; raises (and cleans up) if not.
    FP16 is NOT the deployment target (INT8 is) and is only faster on FP16-capable CPUs."""
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16

    model = onnx.load(fp32_path)
    block = list(set(getattr(float16, "DEFAULT_OP_BLOCK_LIST", []) or []) | set(_FP16_BLOCK))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True, op_block_list=block)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, out_path)
    try:
        ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        Path(out_path).unlink(missing_ok=True)
        raise RuntimeError(f"FP16 model failed to load: {exc}") from exc
    return out_path


# --------------------------------------------------------------------- calibration
def collect_calibration_samples(model, generator, n_chunks: int = 1000,
                                clip_chunks: int = 64, seed: int = 4321
                                ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stream synthetic audio through the FP32 torch model; collect (chunk[1,576], state) pairs.

    Mirrors deployment: builds 576-windows (64 ctx + 512) with context carried across chunks,
    runs model.forward carrying state, and records the (window, state_in) fed at each step.
    """
    import torch

    model = model.eval()
    rng = np.random.default_rng(seed)
    samples: List[Tuple[np.ndarray, np.ndarray]] = []
    state = model.reset_state(1)
    prev_ctx = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    with torch.no_grad():
        while len(samples) < n_chunks:
            audio, _ = generator.generate_batch(1, clip_chunks * CHUNK_SAMPLES, seed=int(rng.integers(1 << 30)))
            audio = audio[0]
            for i in range(clip_chunks):
                chunk512 = audio[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]
                window = np.concatenate([prev_ctx, chunk512]).astype(np.float32)  # [576]
                state_np = state.detach().cpu().numpy().astype(np.float32)
                samples.append((window[None, :].copy(), state_np.copy()))
                _, state = model(torch.from_numpy(window[None, :]), state)
                prev_ctx = chunk512[-CONTEXT_SAMPLES:].copy()
                if len(samples) >= n_chunks:
                    break
            state = model.reset_state(1)
            prev_ctx = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    return samples


def _make_reader(samples):
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self, data):
            self._it = iter([{"chunk": c, "state": s} for c, s in data])

        def get_next(self):
            return next(self._it, None)

    return _Reader(samples)


# --------------------------------------------------------------------- INT8
def quantize_int8(fp32_path: str, out_path: str, calibration_samples,
                  op_types: Optional[List[str]] = None,
                  per_channel: bool = True) -> str:
    """Static INT8 quantization of Conv/Gemm/MatMul only (G-SSM recurrence stays FP32)."""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    op_types = op_types or QUANT_OP_TYPES
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    prep = str(Path(out_path).with_suffix(".prep.onnx"))
    # folds BatchNorm into convs, runs shape inference — required for clean static quant
    quant_pre_process(fp32_path, prep, skip_symbolic_shape=False)
    quantize_static(
        prep, out_path, _make_reader(calibration_samples),
        quant_format=QuantFormat.QDQ,              # QDQ = best INT8 perf on x64 CPU (ORT guidance)
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
    )
    try:
        Path(prep).unlink()
    except OSError:
        pass
    return out_path
