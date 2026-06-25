"""Tests for the standard VAD metrics (PR-AUC, TPR@FPR) used for cross-model comparison."""

import numpy as np

from clearvad.evaluation.metrics import pr_auc, roc_auc, tpr_at_fpr


def test_pr_auc_perfect_separation():
    label = np.array([0, 0, 1, 1], dtype=bool)
    assert pr_auc(np.array([0.1, 0.2, 0.8, 0.9]), label) > 0.99


def test_pr_auc_range():
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, 200).astype(bool)
    probs = rng.random(200)
    v = pr_auc(probs, label)
    assert 0.0 <= v <= 1.0


def test_tpr_at_fpr_perfect():
    label = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    # perfectly separable -> TPR should be 1.0 at any modest FPR
    assert tpr_at_fpr(np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), label, 0.315) == 1.0


def test_tpr_at_fpr_matches_definition():
    # negatives uniformly spread; at FPR=0.5 threshold sits at the negative median
    rng = np.random.default_rng(1)
    neg = rng.random(1000) * 0.5            # negatives in [0, 0.5]
    pos = rng.random(1000) * 0.5 + 0.5      # positives in [0.5, 1.0]
    probs = np.concatenate([neg, pos])
    label = np.concatenate([np.zeros(1000), np.ones(1000)]).astype(bool)
    # fully separable at 0.5 -> TPR ~1.0 at FPR 0.315
    assert tpr_at_fpr(probs, label, 0.315) > 0.95


def test_roc_auc_still_works():
    label = np.array([0, 0, 1, 1], dtype=bool)
    assert abs(roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), label) - 1.0) < 1e-9
