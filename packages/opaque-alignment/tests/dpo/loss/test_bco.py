# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DPO BCO pairwise loss variant (work-unit γ.2).

Covers :func:`bco_loss` (BCO pairwise loss):
- ≥3 hand-computed reference cases (small, analytically tractable inputs).
- ``bco_loss`` with ``delta != 0`` shifts both terms.
- Imports target the concrete implementation paths (public façade not wired
  for this work-unit; façade wiring is γ.W).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from opaque.api.alignment.dpo.loss._bco import bco_loss

# ---------------------------------------------------------------------------
# Helper: build a float tensor from a Python scalar (0-dim)
# ---------------------------------------------------------------------------

_T = torch.tensor


# ===========================================================================
# bco_loss
# ===========================================================================


class TestDpoBcoPair:
    """Hand-computed reference cases + delta-shift for :func:`bco_loss`."""

    def test_zero_inputs_delta0(self) -> None:
        """c=0, r=0, β=0.1, δ=0 → -logsig(0) - logsig(-0) = 2*log(2)."""
        c = _T(0.0)
        r = _T(0.0)
        out = bco_loss(c, r, beta=0.1, delta=0.0)
        expected = _T(2.0 * math.log(2.0))
        assert out.shape == ()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_positive_chosen_negative_rejected_delta0(self) -> None:
        """c=1, r=-1, β=0.5, δ=0 against reference formula."""
        c = _T(1.0)
        r = _T(-1.0)
        beta = 0.5
        expected = -F.logsigmoid(_T(beta * 1.0)) - F.logsigmoid(_T(-(beta * (-1.0))))
        out = bco_loss(c, r, beta=beta, delta=0.0)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_delta_nonzero_shifts_both_terms(self) -> None:
        """δ ≠ 0 shifts the argument of both logsigmoid calls by δ.

        For (c, r, β) = (1, 0, 0.5):
          term_chosen = -logsig(β*c - δ)
          term_rejected = -logsig(-(β*r - δ))

        When δ increases the chosen-term argument decreases and the
        rejected-term argument decreases (both become harder/softer).
        Concretely we verify the result matches the reference formula
        for δ = 0.2 and δ = -0.2.
        """
        c = _T(1.0)
        r = _T(0.0)
        beta = 0.5

        for delta in [0.2, -0.2]:
            expected = -F.logsigmoid(_T(beta * 1.0 - delta)) - F.logsigmoid(
                _T(-(beta * 0.0 - delta))
            )
            out = bco_loss(c, r, beta=beta, delta=delta)
            assert torch.allclose(out, expected, atol=1e-6), (
                f"delta={delta}: got {out.item():.8f}, expected {expected.item():.8f}"
            )

    def test_delta_zero_vs_nonzero_are_different(self) -> None:
        """δ=0 and δ=0.3 must give strictly different losses for non-trivial input."""
        c = _T(1.0)
        r = _T(-1.0)
        beta = 0.5
        out_d0 = bco_loss(c, r, beta=beta, delta=0.0)
        out_d3 = bco_loss(c, r, beta=beta, delta=0.3)
        assert not torch.allclose(out_d0, out_d3, atol=1e-4)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(5)
        c = torch.randn(8)
        r = torch.randn(8)
        out = bco_loss(c, r, beta=0.2, delta=0.1)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    # ------------------------------------------------------------------
    # TRL parity: opaque bco_loss(delta=0) == TRL bco_pair
    # ------------------------------------------------------------------

    def test_trl_parity_delta_zero(self) -> None:
        """With delta=0, opaque bco_loss matches TRL's bco_pair formula.

        TRL's bco_pair:
            chosen_rewards = beta * chosen_logratios
            rejected_rewards = beta * rejected_logratios
            loss = -logsig(chosen_rewards) - logsig(-rejected_rewards)
        """
        c = _T(0.5)
        r = _T(-0.3)
        beta = 0.1
        # TRL-style computation (no delta in the loss itself):
        trl_loss = -F.logsigmoid(beta * c) - F.logsigmoid(-(beta * r))
        out = bco_loss(c, r, beta=beta, delta=0.0)
        assert torch.allclose(out, trl_loss, atol=1e-6)

    def test_trl_parity_batched(self) -> None:
        """Batched: opaque bco_loss(delta=0) matches TRL elementwise."""
        c = torch.tensor([0.5, -0.2, 1.0, 0.0])
        r = torch.tensor([-0.3, 0.1, -0.5, 0.0])
        beta = 0.1
        trl_loss = -F.logsigmoid(beta * c) - F.logsigmoid(-(beta * r))
        out = bco_loss(c, r, beta=beta, delta=0.0)
        assert torch.allclose(out, trl_loss, atol=1e-6)
