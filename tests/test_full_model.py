"""Phase 2 tests for the assembled ClearVAD model."""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_torch
torch = pytest.importorskip("torch")

from clearvad import TOTAL_INPUT_SAMPLES  # noqa: E402
from clearvad.model.clearvad_model import ClearVADConfig, ClearVADModel  # noqa: E402
from tests.export_helpers import export_to_bytes  # noqa: E402

D_INNER, D_STATE = 256, 16


def _model(seed=0, **cfg_over):
    torch.manual_seed(seed)
    cfg = ClearVADConfig(**cfg_over)
    return ClearVADModel(cfg).eval()


def test_full_model_shape():
    m = _model()
    chunk = torch.randn(4, TOTAL_INPUT_SAMPLES)
    state = m.reset_state(4)
    prob, new_state = m(chunk, state)
    assert prob.shape == (4, 1)
    assert new_state.shape == (4, D_INNER, D_STATE)
    assert torch.all((prob >= 0) & (prob <= 1))


def test_full_model_none_state_ok():
    m = _model()
    prob, st = m(torch.randn(2, TOTAL_INPUT_SAMPLES), None)
    assert prob.shape == (2, 1) and st.shape == (2, D_INNER, D_STATE)


def test_full_model_state_carry():
    """A non-zero input state must propagate to both the output prob and the new state.

    Note: on an *untrained* random-init model the state magnitude accumulated from one
    noise chunk is tiny (~1e-6 effect), so we inject an explicit non-zero state to test
    that state is genuinely plumbed through (the exact carry is already proven by
    test_full_model_streaming_equivalence)."""
    m = _model()
    c2 = torch.randn(1, TOTAL_INPUT_SAMPLES)
    big_state = torch.full((1, D_INNER, D_STATE), 4.0)
    p_fresh, s_fresh = m(c2, None)
    p_carry, s_carry = m(c2, big_state)
    assert (p_carry - p_fresh).abs().item() > 1e-4, "input state did not affect output prob"
    assert not torch.allclose(s_carry, s_fresh, atol=1e-4), "input state did not affect new state"


def test_full_model_state_reset_zero_equals_none():
    m = _model()
    chunk = torch.randn(2, TOTAL_INPUT_SAMPLES)
    p_none, s_none = m(chunk, None)
    p_zero, s_zero = m(chunk, m.reset_state(2))
    assert torch.allclose(p_none, p_zero, atol=1e-6)
    assert torch.allclose(s_none, s_zero, atol=1e-6)


def test_full_model_streaming_equivalence():
    """Chunked streaming (carry G-SSM state) == offline (concat features, one G-SSM pass)."""
    m = _model().double().eval()
    B, K = 2, 6
    chunks = [torch.randn(B, TOTAL_INPUT_SAMPLES, dtype=torch.float64) for _ in range(K)]

    # streaming: per-chunk features + incremental G-SSM
    feats = [m.features(c) for c in chunks]
    state, probs_stream = None, []
    for f in feats:
        y, state = m.gssm(f, state)
        probs_stream.append(m.head(y))
    probs_stream = torch.cat(probs_stream, dim=1)          # [B, K]

    # offline: concat features, single G-SSM pass, head per chunk
    T_enc = feats[0].shape[1]
    cat = torch.cat(feats, dim=1)                          # [B, K*T_enc, C]
    y_all, _ = m.gssm(cat, None)
    probs_off = torch.cat(
        [m.head(y_all[:, k * T_enc:(k + 1) * T_enc]) for k in range(K)], dim=1
    )

    max_diff = (probs_stream - probs_off).abs().max().item()
    assert torch.allclose(probs_stream, probs_off, atol=1e-5), f"max diff {max_diff:.2e}"


def test_full_model_param_count_under_300k():
    m = _model()
    counts = m.count_by_module()
    print(f"\nClearVAD param breakdown: {counts}")
    assert counts["total"] < 300_000, counts
    # and smaller than Silero's 16k ONNX (~309,633 params)
    assert counts["total"] < 309_633


def test_full_model_lite_smaller_than_base():
    base = _model()
    lite = _model(gssm_tie_projections=True, head_reduction="last")
    assert lite.parameter_count() < base.parameter_count()


def test_full_model_onnx_export():
    onnx = pytest.importorskip("onnx")  # noqa: F841
    ort = pytest.importorskip("onnxruntime")

    m = _model()
    chunk = torch.randn(1, TOTAL_INPUT_SAMPLES)
    state = m.reset_state(1)

    model_bytes = export_to_bytes(
        m, (chunk, state),
        input_names=["chunk", "state"],
        output_names=["prob", "new_state"],
        dynamic_axes={"chunk": {0: "batch"}, "state": {0: "batch"},
                      "prob": {0: "batch"}, "new_state": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    with torch.no_grad():
        ref_prob, ref_state = m(chunk, state)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(model_bytes, sess_options=so,
                                providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"chunk": chunk.numpy(), "state": state.numpy()})
    np.testing.assert_allclose(ref_prob.numpy(), outs[0], atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(ref_state.numpy(), outs[1], atol=1e-4, rtol=1e-3)


def test_full_model_from_config_dict():
    cfg = {
        "frontend": {"out_channels": 258, "kernel_size": 256, "stride": 128},
        "encoder": {"channel_schedule": [129, 128, 64, 64, 128], "depthwise_separable": True},
        "gssm": {"d_model": 128, "d_inner": 256, "d_state": 16, "dt_rank": 8},
        "head": {"reduction": "mean"},
    }
    m = ClearVADModel.from_config(cfg).eval()
    prob, st = m(torch.randn(1, TOTAL_INPUT_SAMPLES), None)
    assert prob.shape == (1, 1)
