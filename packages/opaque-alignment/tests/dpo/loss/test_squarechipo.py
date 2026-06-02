"""Tests for the DPO SquareChiPO loss variant (arXiv:2505.21395).

Covers work-unit γ.3 of the opaque-alignment plan (§10, §11.2, §11.3).

Hand-computed reference cases, vmap-safety contract tests (§11.2), and a
NaN-injection DP-purity contract test (§11.3) for :func:`squarechipo_loss`.

Imports target concrete implementation paths because the public façade
`__init__.py` is wired in the separate γ.W wire-up unit.
"""

from __future__ import annotations

import torch
from torch.func import grad, vmap

from opaque.api.alignment.dpo.loss._squarechipo import squarechipo_loss

# ---------------------------------------------------------------------------
# squarechipo_loss
# ---------------------------------------------------------------------------


class TestDpoSquarechipo:
    """Hand-computed reference cases for the SquareChiPO loss (arXiv:2505.21395)."""

    def test_squarechipo_delta_zero_beta_one(self) -> None:
        """Δ = 0, β = 1: 0.5 · (σ(0) − 1)² = 0.5 · (0.5 − 1)² = 0.125."""
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = squarechipo_loss(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(0.125), atol=1e-6)

    def test_squarechipo_large_positive_delta(self) -> None:
        """Large positive Δ: σ(β·Δ) → 1, loss → 0."""
        chosen = torch.tensor(20.0)
        rejected = torch.tensor(0.0)
        out = squarechipo_loss(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert out.item() < 1e-5

    def test_squarechipo_large_negative_delta(self) -> None:
        """Large negative Δ: σ(β·Δ) → 0, loss → 0.5 · (0 − 1)² = 0.5."""
        chosen = torch.tensor(-20.0)
        rejected = torch.tensor(0.0)
        out = squarechipo_loss(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(0.5), atol=1e-4)

    def test_squarechipo_delta_one_beta_one(self) -> None:
        """Δ = 1, β = 1: 0.5 · (σ(1) − 1)²; σ(1) = 1/(1+e^{-1})."""
        chosen = torch.tensor(1.0)
        rejected = torch.tensor(0.0)
        out = squarechipo_loss(chosen, rejected, beta=1.0)
        sig = torch.sigmoid(torch.tensor(1.0)).item()
        expected = 0.5 * (sig - 1.0) ** 2
        assert torch.allclose(out, torch.tensor(expected), atol=1e-6)

    def test_squarechipo_batched_shape(self) -> None:
        """Batched (B,) inputs: output shape matches input shape."""
        chosen = torch.randn(8)
        rejected = torch.randn(8)
        out = squarechipo_loss(chosen, rejected, beta=0.5)
        assert out.shape == (8,)

    def test_squarechipo_per_example_matches_batched(self) -> None:
        """Per-example (0-dim) calls match the corresponding batched row."""
        torch.manual_seed(4)
        chosen = torch.randn(6)
        rejected = torch.randn(6)
        batched = squarechipo_loss(chosen, rejected, beta=0.3)
        per_example = torch.stack(
            [squarechipo_loss(chosen[i], rejected[i], beta=0.3) for i in range(6)]
        )
        assert torch.allclose(batched, per_example, atol=1e-6)

    def test_squarechipo_nonnegative(self) -> None:
        """Loss is always non-negative (squared term)."""
        torch.manual_seed(5)
        chosen = torch.randn(20)
        rejected = torch.randn(20)
        out = squarechipo_loss(chosen, rejected, beta=1.0)
        assert (out >= 0.0).all()

    def test_squarechipo_vmap_grad_finite(self) -> None:
        """vmap(grad(squarechipo_loss)) over a (4,) batch yields finite gradients."""
        torch.manual_seed(6)
        chosen = torch.randn(4)
        rejected = torch.randn(4)

        def per_example(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return squarechipo_loss(c, r, beta=0.1)

        grads_c = vmap(grad(per_example))(chosen, rejected)
        assert grads_c.shape == (4,)
        assert torch.isfinite(grads_c).all()

    def test_squarechipo_nan_injection_locality(self) -> None:
        """NaN in one example's input propagates only to that example (Tier 1).

        Replaces index 2 of a 4-example batch with NaN and asserts that
        only index 2 of the output is NaN (gradient locality invariant, §11.3).
        """
        chosen = torch.tensor([0.5, -0.3, float("nan"), 1.2])
        rejected = torch.tensor([0.0, 0.0, 0.0, 0.0])
        out = squarechipo_loss(chosen, rejected, beta=1.0)
        assert out.shape == (4,)
        assert torch.isnan(out[2])
        assert torch.isfinite(out[0])
        assert torch.isfinite(out[1])
        assert torch.isfinite(out[3])


# ---------------------------------------------------------------------------
# Additional vmap(grad) safety over a (4,) synthetic batch
# ---------------------------------------------------------------------------


class TestVmapGradComposed:
    """Additional vmap(grad) safety tests over a (4,) synthetic batch."""

    def test_squarechipo_vmap_grad_finite_with_grad_wrt_rejected(self) -> None:
        """vmap(grad(squarechipo_loss, argnums=1)) over a (4,) batch: finite."""
        torch.manual_seed(8)
        chosen = torch.randn(4)
        rejected = torch.randn(4)

        def fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return squarechipo_loss(c, r, beta=0.2)

        # Gradient w.r.t. rejected (argnums=1)
        grads_r = vmap(grad(fn, argnums=1))(chosen, rejected)
        assert grads_r.shape == (4,)
        assert torch.isfinite(grads_r).all()
