# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for DPO APO, EXO, NCA, BCO, and SPPO loss variants (work-unit γ.2).

Covers:
- :func:`dpo_apo_zero` and :func:`dpo_apo_down` (arXiv:2408.06266)
- :func:`dpo_exo_pair` (EXO pairwise loss)
- :func:`dpo_nca_pair` (NCA pairwise loss)
- :func:`dpo_bco_pair` (BCO pairwise loss)
- :func:`dpo_sppo_hard` (SPPO hard-label loss)

For each function:
- ≥3 hand-computed reference cases (small, analytically tractable inputs).
- vmap-safety (``torch.func.vmap(torch.func.grad(...))``) for at least
  ``dpo_apo_zero`` and ``dpo_sppo_hard``.
- NaN-injection (Tier-1) contract on ``dpo_nca_pair``: a NaN in one example
  propagates only to that example's loss.
- ``dpo_bco_pair`` with ``delta != 0`` shifts both terms.
- Imports target the concrete implementation paths (public façade not wired
  for this work-unit; façade wiring is γ.W).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from opaque.api.alignment.dpo.loss._apo import dpo_apo_down, dpo_apo_zero
from opaque.api.alignment.dpo.loss._bco import dpo_bco_pair
from opaque.api.alignment.dpo.loss._exo import dpo_exo_pair
from opaque.api.alignment.dpo.loss._nca import dpo_nca_pair
from opaque.api.alignment.dpo.loss._sppo import dpo_sppo_hard

# ---------------------------------------------------------------------------
# Helper: build a float tensor from a Python scalar (0-dim)
# ---------------------------------------------------------------------------

_T = torch.tensor


# ===========================================================================
# dpo_apo_zero
# ===========================================================================


class TestDpoApoZero:
    """Hand-computed reference cases for :func:`dpo_apo_zero`."""

    def test_symmetric_zero_inputs(self) -> None:
        """c=0, r=0 → (1-sig(0)) + sig(0) = 0.5 + 0.5 = 1.0."""
        c = _T(0.0)
        r = _T(0.0)
        out = dpo_apo_zero(c, r, beta=0.1)
        assert out.shape == ()
        assert torch.allclose(out, _T(1.0), atol=1e-6)

    def test_c1_rneg1_beta05(self) -> None:
        """c=1, r=-1, β=0.5 → (1-sig(0.5)) + sig(-0.5)."""
        c = _T(1.0)
        r = _T(-1.0)
        out = dpo_apo_zero(c, r, beta=0.5)
        # sig(-0.5) = 1 - sig(0.5); both terms are equal here.
        expected = (1 - torch.sigmoid(_T(0.5))) + torch.sigmoid(_T(-0.5))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_c2_r2_beta1_is_one(self) -> None:
        """c=2, r=2, β=1 → (1-sig(2)) + sig(2) = 1.0 always."""
        c = _T(2.0)
        r = _T(2.0)
        out = dpo_apo_zero(c, r, beta=1.0)
        # For any x: (1-sig(x)) + sig(x) = 1.
        assert torch.allclose(out, _T(1.0), atol=1e-6)

    def test_batched_shape_preserved(self) -> None:
        """(B,) inputs → (B,) output."""
        c = torch.randn(8)
        r = torch.randn(8)
        out = dpo_apo_zero(c, r, beta=0.1)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_vmap_grad_finite(self) -> None:
        """vmap(grad(dpo_apo_zero-sum)) over (4,) batch yields finite grads."""
        b = 4
        torch.manual_seed(42)
        c = torch.randn(b, requires_grad=False)
        r = torch.randn(b, requires_grad=False)

        def per_example(ci: torch.Tensor, ri: torch.Tensor) -> torch.Tensor:
            return dpo_apo_zero(ci, ri, beta=0.1).sum()

        grads_c, grads_r = vmap(grad(per_example, argnums=(0, 1)))(c, r)
        assert grads_c.shape == (b,)
        assert grads_r.shape == (b,)
        assert torch.isfinite(grads_c).all()
        assert torch.isfinite(grads_r).all()


# ===========================================================================
# dpo_apo_down
# ===========================================================================


class TestDpoApoDown:
    """Hand-computed reference cases for :func:`dpo_apo_down`."""

    def test_symmetric_zero_inputs(self) -> None:
        """c=0, r=0 → sig(0) + (1-sig(0)) = 0.5 + 0.5 = 1.0."""
        c = _T(0.0)
        r = _T(0.0)
        out = dpo_apo_down(c, r, beta=0.1)
        assert out.shape == ()
        assert torch.allclose(out, _T(1.0), atol=1e-6)

    def test_c1_r0_beta05_equals_one(self) -> None:
        """c=1, r=0, β=0.5 → sig(0.5) + (1-sig(0.5)) = 1.0."""
        c = _T(1.0)
        r = _T(0.0)
        out = dpo_apo_down(c, r, beta=0.5)
        # b*c = b*(c-r) so lc + lr = sig(x) + (1-sig(x)) = 1
        assert torch.allclose(out, _T(1.0), atol=1e-6)

    def test_c2_r1_beta05(self) -> None:
        """c=2, r=1, β=0.5 → sig(1.0) + (1-sig(0.5))."""
        c = _T(2.0)
        r = _T(1.0)
        beta = 0.5
        expected = torch.sigmoid(_T(beta * 2.0)) + (
            1 - torch.sigmoid(_T(beta * (2.0 - 1.0)))
        )
        out = dpo_apo_down(c, r, beta=beta)
        assert torch.allclose(out, expected, atol=1e-6)
        # Sanity: b*c=1.0 != b*(c-r)=0.5, so result is not trivially 1.
        assert not torch.allclose(out, _T(1.0), atol=1e-4)

    def test_batched_shape_preserved(self) -> None:
        """(B,) inputs → (B,) output."""
        c = torch.randn(6)
        r = torch.randn(6)
        out = dpo_apo_down(c, r, beta=0.2)
        assert out.shape == (6,)
        assert torch.isfinite(out).all()

    def test_loss_nonnegative(self) -> None:
        """Both sigmoid terms are in [0,1] so their sum is in [0,2]."""
        torch.manual_seed(0)
        c = torch.randn(32)
        r = torch.randn(32)
        out = dpo_apo_down(c, r, beta=0.1)
        assert (out >= 0).all()
        assert (out <= 2.0 + 1e-6).all()


# ===========================================================================
# dpo_exo_pair
# ===========================================================================


class TestDpoExoPair:
    """Hand-computed reference cases for :func:`dpo_exo_pair`."""

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
        out = dpo_exo_pair(c, r, beta=beta)
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
        out = dpo_exo_pair(c, r, beta=beta)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_label_smoothing_zero_clamped_to_1e3(self) -> None:
        """label_smoothing=0 is silently clamped to 1e-3 (same as default)."""
        c = _T(0.5)
        r = _T(-0.5)
        out_default = dpo_exo_pair(c, r, beta=1.0, label_smoothing=1e-3)
        out_zero_ls = dpo_exo_pair(c, r, beta=1.0, label_smoothing=0.0)
        # Both should produce the same output: ls=0 is clamped to 1e-3.
        assert torch.allclose(out_default, out_zero_ls, atol=1e-7)

    def test_negative_label_smoothing_clamped(self) -> None:
        """label_smoothing<0 is also clamped to 1e-3 (no log-of-negative)."""
        c = _T(0.5)
        r = _T(-0.5)
        out_neg = dpo_exo_pair(c, r, beta=1.0, label_smoothing=-0.5)
        out_ref = dpo_exo_pair(c, r, beta=1.0, label_smoothing=1e-3)
        assert torch.allclose(out_neg, out_ref, atol=1e-7)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(1)
        c = torch.randn(8)
        r = torch.randn(8)
        out = dpo_exo_pair(c, r, beta=0.2)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_loss_positive(self) -> None:
        """EXO loss is a KL divergence: always ≥ 0."""
        torch.manual_seed(2)
        c = torch.randn(32)
        r = torch.randn(32)
        out = dpo_exo_pair(c, r, beta=0.1)
        assert (out >= -1e-5).all(), "EXO loss should be non-negative"


# ===========================================================================
# dpo_nca_pair
# ===========================================================================


class TestDpoNcaPair:
    """Hand-computed reference cases + NaN-injection for :func:`dpo_nca_pair`."""

    def test_zero_inputs_beta01(self) -> None:
        """c=0, r=0, β=0.1 → -logsig(0) - 0.5*logsig(-0) - 0.5*logsig(-0).

        All three logsigmoid calls evaluate to log(0.5) = -log(2), so
        the total is -(-log2) - 0.5*(-log2) - 0.5*(-log2) = 2*log(2).
        """
        c = _T(0.0)
        r = _T(0.0)
        out = dpo_nca_pair(c, r, beta=0.1)
        expected = _T(2.0 * math.log(2))
        assert out.shape == ()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_positive_chosen_negative_rejected(self) -> None:
        """c=1, r=-1, β=0.5 against reference formula."""
        c = _T(1.0)
        r = _T(-1.0)
        beta = 0.5
        cr = _T(beta * 1.0)
        rr = _T(beta * (-1.0))
        expected = -F.logsigmoid(cr) - 0.5 * F.logsigmoid(-cr) - 0.5 * F.logsigmoid(-rr)
        out = dpo_nca_pair(c, r, beta=beta)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_large_positive_chosen(self) -> None:
        """c=2, r=0, β=1.0 against reference formula."""
        c = _T(2.0)
        r = _T(0.0)
        beta = 1.0
        cr = _T(beta * 2.0)
        rr = _T(beta * 0.0)
        expected = -F.logsigmoid(cr) - 0.5 * F.logsigmoid(-cr) - 0.5 * F.logsigmoid(-rr)
        out = dpo_nca_pair(c, r, beta=beta)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(3)
        c = torch.randn(8)
        r = torch.randn(8)
        out = dpo_nca_pair(c, r, beta=0.3)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_nan_injection_locality(self) -> None:
        """Tier-1 NaN-injection contract: NaN in example 2 → only index 2 is NaN.

        Replacing one example's ``chosen_logratio`` with NaN propagates to
        that example's loss only. All other losses remain finite.
        """
        b = 5
        torch.manual_seed(4)
        c = torch.randn(b)
        r = torch.randn(b)
        c_nan = c.clone()
        c_nan[2] = float("nan")

        out = dpo_nca_pair(c_nan, r, beta=0.5)
        assert out.shape == (b,)
        # Only index 2 is NaN.
        assert torch.isnan(out[2]), "NaN example should produce NaN loss"
        for i in range(b):
            if i != 2:
                assert torch.isfinite(out[i]), f"Example {i} should be finite"


# ===========================================================================
# dpo_bco_pair
# ===========================================================================


class TestDpoBcoPair:
    """Hand-computed reference cases + delta-shift for :func:`dpo_bco_pair`."""

    def test_zero_inputs_delta0(self) -> None:
        """c=0, r=0, β=0.1, δ=0 → -logsig(0) - logsig(-0) = 2*log(2)."""
        c = _T(0.0)
        r = _T(0.0)
        out = dpo_bco_pair(c, r, beta=0.1, delta=0.0)
        expected = _T(2.0 * math.log(2.0))
        assert out.shape == ()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_positive_chosen_negative_rejected_delta0(self) -> None:
        """c=1, r=-1, β=0.5, δ=0 against reference formula."""
        c = _T(1.0)
        r = _T(-1.0)
        beta = 0.5
        expected = -F.logsigmoid(_T(beta * 1.0)) - F.logsigmoid(_T(-(beta * (-1.0))))
        out = dpo_bco_pair(c, r, beta=beta, delta=0.0)
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
            out = dpo_bco_pair(c, r, beta=beta, delta=delta)
            assert torch.allclose(out, expected, atol=1e-6), (
                f"delta={delta}: got {out.item():.8f}, expected {expected.item():.8f}"
            )

    def test_delta_zero_vs_nonzero_are_different(self) -> None:
        """δ=0 and δ=0.3 must give strictly different losses for non-trivial input."""
        c = _T(1.0)
        r = _T(-1.0)
        beta = 0.5
        out_d0 = dpo_bco_pair(c, r, beta=beta, delta=0.0)
        out_d3 = dpo_bco_pair(c, r, beta=beta, delta=0.3)
        assert not torch.allclose(out_d0, out_d3, atol=1e-4)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(5)
        c = torch.randn(8)
        r = torch.randn(8)
        out = dpo_bco_pair(c, r, beta=0.2, delta=0.1)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()


# ===========================================================================
# dpo_sppo_hard
# ===========================================================================


class TestDpoSppoHard:
    """Hand-computed reference cases for :func:`dpo_sppo_hard`."""

    def test_zero_inputs_beta05(self) -> None:
        """c=0, r=0, β=0.5 → target = 1.0; (0-1)^2 + (0+1)^2 = 2.0."""
        c = _T(0.0)
        r = _T(0.0)
        out = dpo_sppo_hard(c, r, beta=0.5)
        assert out.shape == ()
        assert torch.allclose(out, _T(2.0), atol=1e-6)

    def test_at_nash_equilibrium_is_zero(self) -> None:
        """c = 0.5/β, r = -0.5/β → both squared terms are 0 → loss = 0."""
        beta = 0.1
        target = 0.5 / beta  # = 5.0 exactly (float, not int)
        c = _T(target)
        r = _T(-target)
        out = dpo_sppo_hard(c, r, beta=beta)
        assert torch.allclose(out, _T(0.0), atol=1e-6)

    def test_asymmetric_inputs_beta01(self) -> None:
        """c=1, r=-1, β=0.1 → target=5; (1-5)^2 + (-1+5)^2 = 16+16 = 32."""
        c = _T(1.0)
        r = _T(-1.0)
        out = dpo_sppo_hard(c, r, beta=0.1)
        assert torch.allclose(out, _T(32.0), atol=1e-6)

    def test_float_division_not_integer_division(self) -> None:
        """0.5/beta must be floating-point: for beta=3, target=1/6, not 0."""
        beta = 3.0
        target = 0.5 / beta
        # If integer division were used: 0 // 3 = 0 → loss = c^2 + r^2 for c=0, r=0.
        # With float division: target ≈ 0.1667.
        assert abs(target - 1.0 / 6.0) < 1e-9
        c = _T(target)  # exactly at target
        r = _T(-target)
        out = dpo_sppo_hard(c, r, beta=beta)
        assert torch.allclose(out, _T(0.0), atol=1e-6)

    def test_batched_shape_and_finite(self) -> None:
        """(B,) inputs → (B,) finite outputs."""
        torch.manual_seed(6)
        c = torch.randn(8)
        r = torch.randn(8)
        out = dpo_sppo_hard(c, r, beta=0.5)
        assert out.shape == (8,)
        assert torch.isfinite(out).all()

    def test_vmap_grad_finite(self) -> None:
        """vmap(grad(dpo_sppo_hard-sum)) over (4,) batch yields finite grads."""
        b = 4
        torch.manual_seed(7)
        c = torch.randn(b)
        r = torch.randn(b)

        def per_example(ci: torch.Tensor, ri: torch.Tensor) -> torch.Tensor:
            return dpo_sppo_hard(ci, ri, beta=0.1).sum()

        grads_c, grads_r = vmap(grad(per_example, argnums=(0, 1)))(c, r)
        assert grads_c.shape == (b,)
        assert grads_r.shape == (b,)
        assert torch.isfinite(grads_c).all()
        assert torch.isfinite(grads_r).all()

    def test_loss_nonnegative(self) -> None:
        """Squared terms are always ≥ 0."""
        torch.manual_seed(8)
        c = torch.randn(32)
        r = torch.randn(32)
        out = dpo_sppo_hard(c, r, beta=0.2)
        assert (out >= 0).all()
