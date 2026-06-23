"""Phase 2 tests for the STFT-style front-end."""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_torch
torch = pytest.importorskip("torch")

from clearvad import TOTAL_INPUT_SAMPLES  # noqa: E402
from clearvad.model.frontend import STFTFrontend  # noqa: E402


def test_frontend_shape():
    fe = STFTFrontend().eval()
    x = torch.randn(4, TOTAL_INPUT_SAMPLES)
    out = fe(x)
    L = fe.output_frames(TOTAL_INPUT_SAMPLES)
    assert out.shape == (4, 129, L)
    assert L >= 1


def test_frontend_accepts_3d():
    fe = STFTFrontend().eval()
    x = torch.randn(2, 1, TOTAL_INPUT_SAMPLES)
    assert fe(x).shape[1] == 129


def test_frontend_magnitude_nonnegative():
    fe = STFTFrontend().eval()
    out = fe(torch.randn(2, TOTAL_INPUT_SAMPLES))
    assert torch.all(out >= 0)


def test_frontend_load_basis():
    fe = STFTFrontend()
    w = torch.randn_like(fe.conv.weight)
    fe.load_basis(w)
    assert torch.allclose(fe.conv.weight, w)
    with pytest.raises(ValueError):
        fe.load_basis(torch.randn(10, 1, 5))


def test_frontend_learnable_flag():
    fe = STFTFrontend(learnable=False)
    assert fe.conv.weight.requires_grad is False
    fe2 = STFTFrontend(learnable=True)
    assert fe2.conv.weight.requires_grad is True


def test_frontend_param_count():
    fe = STFTFrontend()
    # 258 * 1 * 256 = 66048, no bias
    assert fe.parameter_count() == 258 * 256
