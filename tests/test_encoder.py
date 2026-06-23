"""Phase 2 tests for the depthwise-separable convolutional encoder."""

import pytest

pytestmark = pytest.mark.requires_torch
torch = pytest.importorskip("torch")

from clearvad.model.encoder import ConvEncoder, DWSepConv1d, PlainConv1d  # noqa: E402


def test_encoder_shape_preserves_time():
    enc = ConvEncoder().eval()
    x = torch.randn(3, 129, 5)         # [B, 129, T]
    out = enc(x)
    assert out.shape == (3, 128, 5)    # channels -> 128, T preserved by 'same' padding


def test_encoder_in_out_channels():
    enc = ConvEncoder()
    assert enc.in_channels == 129
    assert enc.out_channels == 128


def test_dwsep_is_lighter_than_plain():
    dw = DWSepConv1d(129, 128).eval()
    pl = PlainConv1d(129, 128).eval()
    n_dw = sum(p.numel() for p in dw.parameters())
    n_pl = sum(p.numel() for p in pl.parameters())
    assert n_dw < n_pl


def test_encoder_param_count_reasonable():
    enc = ConvEncoder(depthwise_separable=True)
    n = enc.parameter_count()
    print(f"\ndepthwise-separable encoder params: {n}")
    # much lighter than Silero's ~111K plain-conv encoder
    assert n < 60_000


def test_encoder_plain_variant_shape():
    enc = ConvEncoder(depthwise_separable=False).eval()
    out = enc(torch.randn(2, 129, 4))
    assert out.shape == (2, 128, 4)


def test_encoder_gradient_flow():
    enc = ConvEncoder().train()
    x = torch.randn(4, 129, 8, requires_grad=True)
    enc(x).pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
