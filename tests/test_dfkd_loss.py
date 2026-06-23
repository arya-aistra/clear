"""Phase 3 tests for the distillation loss."""

import pytest

pytestmark = pytest.mark.requires_torch
torch = pytest.importorskip("torch")

from clearvad.distill.losses import (  # noqa: E402
    DFKDLoss,
    agreement_rate,
    bernoulli_kl,
    transition_weight,
)


def test_loss_finite():
    torch.manual_seed(0)
    s = torch.randn(4, 64)
    t = torch.rand(4, 64)
    loss = DFKDLoss(lambda_soft=1.0, lambda_boundary=2.0, lambda_smooth=0.5)
    val, parts = loss(s, t)
    assert torch.isfinite(val)
    for k in ("total", "soft", "boundary", "smooth"):
        assert k in parts and parts[k] == parts[k]  # not NaN


def test_soft_loss_zero_when_matched():
    # student logits == teacher logits -> distributions match -> L_soft ≈ 0
    t = torch.rand(2, 32).clamp(0.05, 0.95)
    teacher_logits = (t / (1 - t)).log()
    loss = DFKDLoss(lambda_soft=1.0, lambda_boundary=0.0, lambda_smooth=0.0)
    val, parts = loss(teacher_logits, t)
    assert parts["soft"] < 1e-4
    assert parts["boundary"] == 0.0 and parts["smooth"] == 0.0


def test_transition_weight_fires():
    # teacher_hard: silence then speech -> one change at index 4
    th = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)
    w = transition_weight(th, width=1, weight=5.0)
    assert w.shape == th.shape
    # frames adjacent to the change are upweighted; far frames stay at 1.0
    assert w[0, 3] > 1.0 and w[0, 4] > 1.0
    assert abs(w[0, 0].item() - 1.0) < 1e-6
    assert abs(w[0, 7].item() - 1.0) < 1e-6


def test_bernoulli_kl_nonneg_and_zero():
    p = torch.rand(100).clamp(0.01, 0.99)
    assert torch.all(bernoulli_kl(p, p) < 1e-5)          # KL(p||p)=0
    assert torch.all(bernoulli_kl(p, torch.full_like(p, 0.5)) >= -1e-6)


def test_gradient_flows_to_student():
    s = torch.randn(2, 16, requires_grad=True)
    t = torch.rand(2, 16)
    loss = DFKDLoss()
    val, _ = loss(s, t)
    val.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_stage1_only_soft_active():
    s = torch.randn(2, 32)
    t = torch.rand(2, 32)
    loss = DFKDLoss(lambda_soft=1.0, lambda_boundary=0.0, lambda_smooth=0.0)
    val, parts = loss(s, t)
    assert parts["boundary"] == 0.0 and parts["smooth"] == 0.0
    assert parts["soft"] > 0.0


def test_agreement_rate():
    s = torch.tensor([[0.9, 0.1, 0.8]])
    t = torch.tensor([[0.7, 0.2, 0.3]])
    # speech mask: s>0.5 -> [1,0,1]; t>0.5 -> [1,0,0]; agree on 2/3
    assert abs(agreement_rate(s, t) - 2 / 3) < 1e-6
