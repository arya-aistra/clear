"""Phase 5 tests — FP32 ONNX export + runtime validation (works on an untrained model)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.requires_torch
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from clearvad import TOTAL_INPUT_SAMPLES  # noqa: E402
from clearvad.export.to_onnx import export_fp32, onnx_size_mb  # noqa: E402
from clearvad.export.validate_onnx import OrtVADRunner, validate_all  # noqa: E402
from clearvad.model.clearvad_model import ClearVADModel  # noqa: E402


def _model(seed=0):
    torch.manual_seed(seed)
    return ClearVADModel().eval()


def test_export_fp32_creates_file():
    m = _model()
    with tempfile.TemporaryDirectory() as d:
        p = export_fp32(m, str(Path(d) / "m.onnx"))
        assert Path(p).exists() and onnx_size_mb(p) > 0


def test_fp32_numeric_and_state_carry():
    m = _model()
    with tempfile.TemporaryDirectory() as d:
        p = export_fp32(m, str(Path(d) / "m.onnx"))
        res = validate_all(p, m, numeric=True, atol=1e-3)
        assert res["numeric_equivalence"]["passed"], res["numeric_equivalence"]
        assert res["state_carry"]["passed"], res["state_carry"]
        assert res["determinism"]["passed"]
        assert res["all_passed"]


def test_runner_shapes():
    m = _model()
    with tempfile.TemporaryDirectory() as d:
        p = export_fp32(m, str(Path(d) / "m.onnx"))
        r = OrtVADRunner(p)
        prob = r.step(np.zeros(TOTAL_INPUT_SAMPLES, dtype=np.float32))
        assert 0.0 <= prob <= 1.0
        assert list(r.state_shape) == [1, 256, 16]
