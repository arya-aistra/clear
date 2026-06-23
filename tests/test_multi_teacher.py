"""Tests for multi-teacher ensemble combination + LayerNorm-before-G-SSM model option."""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_torch
torch = pytest.importorskip("torch")

from clearvad import TOTAL_INPUT_SAMPLES  # noqa: E402
from clearvad.distill.multi_teacher import combine_probs  # noqa: E402
from clearvad.model.clearvad_model import ClearVADConfig, ClearVADModel  # noqa: E402


def test_combine_probs_weighted_average():
    a = torch.tensor([[1.0, 0.0, 1.0]])
    b = torch.tensor([[0.0, 0.0, 0.0]])
    out = combine_probs([a, b], [0.5, 0.5])
    assert torch.allclose(out, torch.tensor([[0.5, 0.0, 0.5]]))


def test_combine_probs_weights_normalized():
    a = torch.tensor([[1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0]])
    # weights need not sum to 1; they are normalized internally
    out = combine_probs([a, b], [3.0, 1.0])
    assert torch.allclose(out, torch.tensor([[0.75, 0.75]]))


def test_combine_probs_trims_to_common_length():
    a = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0]])
    out = combine_probs([a, b], [0.5, 0.5])
    assert out.shape[-1] == 2


def test_model_input_norm_param_count_and_shapes():
    base = ClearVADModel(ClearVADConfig()).eval()
    mt = ClearVADModel(ClearVADConfig(gssm_input_norm=True)).eval()
    # LayerNorm(128) adds 256 params (weight+bias)
    assert mt.parameter_count() - base.parameter_count() == 256
    prob, state = mt(torch.randn(2, TOTAL_INPUT_SAMPLES), None)
    assert prob.shape == (2, 1) and state.shape == (2, 256, 16)


def test_model_input_norm_streaming_equivalence():
    m = ClearVADModel(ClearVADConfig(gssm_input_norm=True)).double().eval()
    with torch.no_grad():
        m.gssm.A_log.fill_(-8.0)
    B, K = 2, 5
    chunks = [torch.randn(B, TOTAL_INPUT_SAMPLES, dtype=torch.float64) for _ in range(K)]
    feats = [m.features(c) for c in chunks]
    state, stream = None, []
    for f in feats:
        y, state = m.gssm(f, state)
        stream.append(m.head(y))
    stream = torch.cat(stream, dim=1)
    t_enc = feats[0].shape[1]
    cat = torch.cat(feats, dim=1)
    y_all, _ = m.gssm(cat, None)
    off = torch.cat([m.head(y_all[:, k * t_enc:(k + 1) * t_enc]) for k in range(K)], dim=1)
    assert torch.allclose(stream, off, atol=1e-5)
