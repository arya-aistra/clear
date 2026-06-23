"""Phase 1 tests for the Gated SSM (G-SSM) block.

Covers shape, state I/O, streaming==offline equivalence, zero/None state equivalence,
gradient flow, ONNX exportability (T=1 streaming and T=16 chunk), and the parameter
budget (< LSTM it replaces; tied variant < 100K).
"""

import io

import numpy as np
import pytest

pytestmark = pytest.mark.requires_torch

torch = pytest.importorskip("torch")

from clearvad.model.gssm import GSSM  # noqa: E402

D_MODEL, D_INNER, D_STATE, DT_RANK = 128, 256, 16, 8
LSTM_EQUIV_PARAMS = 4 * 128 * (128 + 128) + 4 * 128  # 131,584 (spec reference)


def _export_to_bytes(m, args, **kw):
    """Export to ONNX bytes, preferring the legacy TorchScript exporter.

    torch>=2.x defaults to the dynamo exporter (needs `onnxscript`); the legacy exporter
    gives predictable unrolling of our explicit T-loop and exact opset control, which is
    what Phase 5 deployment wants. Try legacy (dynamo=False) first, fall back to dynamo.
    """
    # Preferred: legacy exporter.
    try:
        buf = io.BytesIO()
        torch.onnx.export(m, args, buf, dynamo=False, **kw)
        buf.seek(0)
        return buf.read()
    except TypeError:
        pass          # this torch has no `dynamo` kwarg -> plain default below
    except Exception:
        pass          # legacy path unavailable -> try dynamo below
    # Fallback: dynamo / default exporter.
    buf = io.BytesIO()
    try:
        torch.onnx.export(m, args, buf, dynamo=True, **kw)
    except TypeError:
        buf = io.BytesIO()
        torch.onnx.export(m, args, buf, **kw)
    buf.seek(0)
    return buf.read()


def _make(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(d_model=D_MODEL, d_inner=D_INNER, d_state=D_STATE, dt_rank=DT_RANK)
    cfg.update(kw)
    return GSSM(**cfg).eval()


def test_gssm_shape():
    m = _make()
    x = torch.randn(4, 10, D_MODEL)
    out, state = m(x)
    assert out.shape == (4, 10, D_MODEL)
    assert state.shape == (4, D_INNER, D_STATE)


def test_gssm_state_shapes():
    m = _make()
    x = torch.randn(2, 5, D_MODEL)
    s0 = m.initial_state(2)
    assert s0.shape == (2, D_INNER, D_STATE)
    out, s1 = m(x, s0)
    assert s1.shape == (2, D_INNER, D_STATE)
    # state should actually advance
    assert not torch.allclose(s0, s1)


def test_gssm_streaming_equivalence():
    """offline over T == T sequential T=1 online steps carrying state (within 1e-5)."""
    m = _make().double()  # float64 to assert tight numerical equivalence
    B, T = 3, 32
    x = torch.randn(B, T, D_MODEL, dtype=torch.float64)

    out_offline, state_offline = m(x)

    out_chunks = []
    state = None
    for t in range(T):
        o_t, state = m(x[:, t : t + 1], state)
        out_chunks.append(o_t)
    out_online = torch.cat(out_chunks, dim=1)

    assert torch.allclose(out_offline, out_online, atol=1e-5, rtol=1e-5), \
        f"max diff {(out_offline - out_online).abs().max().item():.2e}"
    assert torch.allclose(state_offline, state, atol=1e-5, rtol=1e-5)


def test_gssm_state_reset_zero_equals_none():
    m = _make()
    x = torch.randn(2, 7, D_MODEL)
    out_none, s_none = m(x, None)
    out_zero, s_zero = m(x, m.initial_state(2))
    assert torch.allclose(out_none, out_zero, atol=1e-6)
    assert torch.allclose(s_none, s_zero, atol=1e-6)


def test_gssm_gradient_flow():
    m = _make()
    x = torch.randn(2, 16, D_MODEL, requires_grad=True)
    out, _ = m(x)
    loss = out.pow(2).mean()
    loss.backward()
    # every parameter receives a finite gradient
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert torch.isfinite(x.grad).all()
    # at least some gradient signal is non-trivial (no total vanish)
    total = sum(p.grad.abs().sum() for p in m.parameters())
    assert total > 0


def test_gssm_no_future_leakage():
    """Causality: output at frame t must not depend on inputs at t' > t."""
    m = _make()
    x = torch.randn(1, 12, D_MODEL)
    out_a, _ = m(x)
    x2 = x.clone()
    x2[:, 8:] += torch.randn_like(x2[:, 8:])  # perturb the future
    out_b, _ = m(x2)
    # frames before the perturbation must be identical
    assert torch.allclose(out_a[:, :8], out_b[:, :8], atol=1e-6)


@pytest.mark.parametrize("T", [1, 16])
def test_gssm_onnx_exportable(T):
    onnx = pytest.importorskip("onnx")  # noqa: F841
    ort = pytest.importorskip("onnxruntime")

    m = _make()
    x = torch.randn(1, T, D_MODEL)
    state = m.initial_state(1)

    model_bytes = _export_to_bytes(
        m, (x, state),
        input_names=["x", "state"],
        output_names=["out", "new_state"],
        dynamic_axes={"x": {0: "batch"}, "state": {0: "batch"},
                      "out": {0: "batch"}, "new_state": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    with torch.no_grad():
        ref_out, ref_state = m(x, state)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(model_bytes, sess_options=so,
                                providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"x": x.numpy(), "state": state.numpy()})
    ort_out, ort_state = outs[0], outs[1]

    np.testing.assert_allclose(ref_out.numpy(), ort_out, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(ref_state.numpy(), ort_state, atol=1e-4, rtol=1e-3)


def test_gssm_param_count_under_lstm():
    m = _make()  # faithful, bias-free
    n = m.parameter_count()
    print(f"\nfaithful G-SSM params: {n} (LSTM equiv: {LSTM_EQUIV_PARAMS})")
    # HARD gate: fewer params than the LSTM it replaces
    assert n < LSTM_EQUIV_PARAMS, f"{n} !< {LSTM_EQUIV_PARAMS}"


def test_gssm_tied_under_100k():
    m = _make(tie_projections=True)
    n = m.parameter_count()
    print(f"\ntied G-SSM params: {n}")
    # SOFT target from spec: < 100K at d_inner=256 (only reachable via tying)
    assert n < 100_000, f"{n} !< 100000"
    # tied block still functions
    x = torch.randn(2, 8, D_MODEL)
    out, st = m(x)
    assert out.shape == (2, 8, D_MODEL) and st.shape == (2, D_INNER, D_STATE)
