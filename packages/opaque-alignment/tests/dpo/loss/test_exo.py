# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DPO EXO pairwise loss variant (work-unit γ.2).

Covers :func:`exo_loss` (EXO pairwise loss):
- ≥3 hand-computed reference cases (small, analytically tractable inputs).
- Imports target the concrete implementation paths (public façade not wired
  for this work-unit; façade wiring is γ.W).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from opaque.api.alignment.dpo.loss._exo import exo_loss

# ---------------------------------------------------------------------------
# Helper: build a float tensor from a Python scalar (0-dim)
# ---------------------------------------------------------------------------

_T = torch.tensor


# ===========================================================================
# exo_loss
# ===========================================================================


class TestDpoExoPair:
    """Hand-computed reference cases for :func:`exo_loss`."""

    def test_zero_logits_default_ls(self) -> None:
        """c=0, r=0, β=0.1, ls=1e-3 → reference formula evaluated at logits=0."""
        c = _T(0.0)
        r = _T(0.0)
        beta = 0.1
        ls = 1e-3
        logits = _T(beta * 0.0)
        expected = F.sigmoid(logits) * (
            F.logsigmoid(logits) - math.log(1 - ls)
        ) + F.sigmoid(-logits) * (F.logsigmoid(-logits) - math.log(ls))
        out = exo_loss(c, r, beta=beta)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_positive_margin(self) -> None:
        """c=1, r=-1, β=0.5, ls=1e-3."""
        c = _T(1.0)
        r = _T(-1.0)
        beta = 0.5
        ls = 1e-3
        logits = _T(beta * (1.0 - (-1.0)))  # = 1.0
        expected = F.sigmoid(logits) * (
            F.logsigmoid(logits) - math.log(1 - ls)
        ) + F.sigmoid(-logits) * (F.logsigmoid(-logits) - math.log(ls))
        out = exo_loss(c, r, beta=beta)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_label_smoothing_zero_clamped_to_1e3(self) -> None:
        """label_smoothing=0 is silently clamped to 1e-3 (same as default)."""
        c = _T(0.5)
        r = _T(-0.5)
        out_default = exo_loss(c, r, beta=1.0, label_smoothing=1e-3)
        out_zero_ls = exo_loss(c, r, beta=1.0, label_smoothing=0.0)
        # Both should produce the same output: ls=0 is clamped to 1e-3.
        assert torch.allclose(out_default, out_zero_ls, atol=1e-7)

    def test_negative_label_smoothing_clamped(self) -> None:
        """label_smoothing<0 is also clamped to 1e-3 (no log-of-negative)."""
        c = _T(0.5)
        r = _T(-0.5)
        out_neg = exo_loss(c, r, beta=1.0, label_smoothing=-0.5)
        out_ref = exo_loss(c, r, beta=1.0, label_smoothing=1e-3)
        assert torch.allclose(out_neg, out_ref, atol=1e-7)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(1)
        c = torch.randn(8)
        r = torch.randn(8)
        out = exo_loss(c, r, beta=0.2)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_loss_positive(self) -> None:
        """EXO loss is a KL divergence: always ≥ 0."""
        torch.manual_seed(2)
        c = torch.randn(32)
        r = torch.randn(32)
        out = exo_loss(c, r, beta=0.1)
        assert (out >= -1e-5).all(), "EXO loss should be non-negative"
