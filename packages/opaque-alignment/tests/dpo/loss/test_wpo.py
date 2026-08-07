# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + vmap-safety tests for the WPO reweighting helper (plan §7.1, §3.4).

Covers :func:`wpo_weights` — the WPO per-example reweighting (arXiv:2406.11827).
A hand-computed avg-logp→exp case, the **detach** invariant (a non-detached
weight would couple the gradient and break DP Tier 1, §3.3), and the
all-zero-mask ``clamp(min=1)`` div-by-zero guard.

Imports target the concrete impl paths because the public façade is wired by a
later unit (γ.W).
"""

from __future__ import annotations

import pytest
import torch
from torch.func import vmap

from opaque.api.alignment.dpo.loss._wpo import wpo_weights

# ---------------------------------------------------------------------------
# wpo_weights — per-example reweighting (arXiv:2406.11827)
# ---------------------------------------------------------------------------


def test_wpo_weights_hand_computed() -> None:
    """avg_logp = masked-mean per row, weight = exp(avg_logp)."""
    # Row 0: logps [-1, -2, 0], mask [1, 1, 0] → avg = -1.5 → exp(-1.5).
    # Row 1: logps [-0.5, -0.5, -0.5], mask [1, 1, 1] → avg = -0.5 → exp(-0.5).
    logps = torch.tensor([[-1.0, -2.0, 0.0], [-0.5, -0.5, -0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask)
    expected = torch.tensor([torch.tensor(-1.5).exp(), torch.tensor(-0.5).exp()])
    assert out.shape == (2,)
    assert torch.allclose(out, expected, atol=1e-6)


def test_wpo_weights_is_detached() -> None:
    """The weight is detached even with grad-tracking inputs (DP Tier 1, §3.3)."""
    logps = torch.tensor([[-1.0, -2.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask)
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_wpo_weights_all_zero_mask_no_div0() -> None:
    """An all-zero mask row uses clamp(min=1): avg_logp = 0 → weight = 1."""
    logps = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]])
    mask = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask)
    assert torch.isfinite(out).all()
    # Row 0: numerator is 0 (all masked), denom clamped to 1 → exp(0) = 1.
    assert torch.allclose(out[0], torch.tensor(1.0), atol=1e-6)


def test_wpo_weights_divisor_is_exact_token_count_under_bf16() -> None:
    """The detached WPO average must not round 257 completion tokens to 256."""

    def _uniform_weight(n_valid: int) -> float:
        logps = torch.full((n_valid,), -1.0, dtype=torch.bfloat16)
        mask = torch.ones(n_valid, dtype=torch.bool)
        return wpo_weights(logps, mask).item()

    assert _uniform_weight(256) == pytest.approx(_uniform_weight(257), rel=1e-4)


def test_wpo_weights_vmap_safe() -> None:
    """wpo_weights runs under vmap over a batch axis and stays detached."""
    logps = torch.randn(4, 5)
    mask = torch.ones(4, 5)
    out = vmap(wpo_weights)(logps, mask)
    assert out.shape == (4,)
    assert out.requires_grad is False
