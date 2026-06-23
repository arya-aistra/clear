"""Phase 5 tests — INT8 static quantization loads, runs, carries state, and is smaller.

(Numeric equivalence is NOT expected for INT8; we check it runs + state carry holds + the
binary is meaningfully smaller. Exact F1 degradation is measured by scripts/export_onnx.py.)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.requires_torch
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from clearvad.distill.synthetic_gen import SyntheticAudioGenerator  # noqa: E402
from clearvad.export.quantize import collect_calibration_samples, quantize_int8  # noqa: E402
from clearvad.export.to_onnx import export_fp32, onnx_size_mb  # noqa: E402
from clearvad.export.validate_onnx import OrtVADRunner, validate_all  # noqa: E402
from clearvad.model.clearvad_model import ClearVADModel  # noqa: E402


def _model(seed=0):
    torch.manual_seed(seed)
    return ClearVADModel().eval()


def test_int8_quantize_runs_and_carries_state():
    m = _model()
    gen = SyntheticAudioGenerator()
    with tempfile.TemporaryDirectory() as d:
        fp32 = export_fp32(m, str(Path(d) / "m.onnx"))
        calib = collect_calibration_samples(m, gen, n_chunks=64, clip_chunks=16)
        int8 = quantize_int8(fp32, str(Path(d) / "m_int8.onnx"), calib)
        assert Path(int8).exists()
        # INT8 binary should be smaller than FP32
        assert onnx_size_mb(int8) < onnx_size_mb(fp32)
        # it runs + state carry holds (numeric equivalence not required for INT8)
        res = validate_all(int8, m, numeric=False)
        assert res["determinism"]["passed"]
        # produces valid probabilities
        r = OrtVADRunner(int8)
        probs = r.stream(np.random.default_rng(0).standard_normal(20 * 512).astype(np.float32) * 0.1)
        assert np.all((probs >= 0) & (probs <= 1))
