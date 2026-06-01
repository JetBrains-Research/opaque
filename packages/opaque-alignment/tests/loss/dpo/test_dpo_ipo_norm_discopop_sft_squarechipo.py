"""Tests for DPO variants: IPO, sigmoid_norm, DiscoPOP, SFT, SquareChiPO.

Covers work-unit γ.3 of the opaque-alignment plan (§10, §11.2, §11.3).

Each variant has ≥ 3 hand-computed reference cases, a vmap-safety
contract test (§11.2), and a NaN-injection DP-purity contract test
(§11.3) where applicable.

Imports target concrete implementation paths because the public façade
`__init__.py` is wired in the separate γ.W wire-up unit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from opaque.api.alignment.loss.dpo._discopop import dpo_discopop
from opaque.api.alignment.loss.dpo._ipo import dpo_ipo
from opaque.api.alignment.loss.dpo._sft import dpo_sft
from opaque.api.alignment.loss.dpo._sigmoid_norm import dpo_sigmoid_norm
from opaque.api.alignment.loss.dpo._squarechipo import dpo_squarechipo

# ---------------------------------------------------------------------------
# dpo_ipo
# ---------------------------------------------------------------------------


class TestDpoIpo:
    """Hand-computed reference cases for the IPO squared loss."""

    def test_ipo_delta_zero_beta_one(self) -> None:
        """Δ = 0, β = 1: (0 − 1/2)² = 0.25."""
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = dpo_ipo(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(0.25), atol=1e-6)

    def test_ipo_delta_one_half_beta_one(self) -> None:
        """Δ = 0.5, β = 1: (0.5 − 0.5)² = 0.0."""
        chosen = torch.tensor(0.5)
        rejected = torch.tensor(0.0)
        out = dpo_ipo(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)

    def test_ipo_delta_one_beta_two(self) -> None:
        """Δ = 1, β = 2: (1 − 1/4)² = 0.5625."""
        chosen = torch.tensor(1.5)
        rejected = torch.tensor(0.5)
        out = dpo_ipo(chosen, rejected, beta=2.0)
        expected = (1.0 - 1.0 / (2.0 * 2.0)) ** 2
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(expected), atol=1e-6)

    def test_ipo_negative_delta(self) -> None:
        """Δ = -1, β = 1: (-1 − 0.5)² = 2.25."""
        chosen = torch.tensor(-1.0)
        rejected = torch.tensor(0.0)
        out = dpo_ipo(chosen, rejected, beta=1.0)
        expected = (-1.0 - 0.5) ** 2  # 2.25
        assert torch.allclose(out, torch.tensor(expected), atol=1e-6)

    def test_ipo_batched(self) -> None:
        """Batched (B,) inputs: output shape matches input shape."""
        chosen = torch.tensor([0.0, 0.5, 1.0])
        rejected = torch.tensor([0.0, 0.0, 0.0])
        out = dpo_ipo(chosen, rejected, beta=1.0)
        assert out.shape == (3,)
        # check against per-example results
        for i in range(3):
            expected = dpo_ipo(chosen[i], rejected[i], beta=1.0)
            assert torch.allclose(out[i], expected, atol=1e-6)

    def test_ipo_vmap_grad_finite(self) -> None:
        """vmap(grad(dpo_ipo)) over a (4,) batch yields finite gradients."""
        torch.manual_seed(0)
        chosen = torch.randn(4)
        rejected = torch.randn(4)

        def per_example(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return dpo_ipo(c, r, beta=0.1)

        grads_c = vmap(grad(per_example))(chosen, rejected)
        assert grads_c.shape == (4,)
        assert torch.isfinite(grads_c).all()


# ---------------------------------------------------------------------------
# dpo_sigmoid_norm
# ---------------------------------------------------------------------------


class TestDpoSigmoidNorm:
    """Hand-computed reference cases for the length-normalised sigmoid loss."""

    def test_sigmoid_norm_delta_zero_no_smoothing(self) -> None:
        """Δ = 0, β = 1, ε = 0: -log σ(0) = log 2."""
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = dpo_sigmoid_norm(chosen, rejected, beta=1.0)
        expected = math.log(2.0)  # -log(sigmoid(0)) = -log(0.5) = log 2
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(expected), atol=1e-6)

    def test_sigmoid_norm_large_positive_delta(self) -> None:
        """Large positive Δ → loss near 0 (chosen clearly preferred)."""
        chosen = torch.tensor(10.0)
        rejected = torch.tensor(0.0)
        out = dpo_sigmoid_norm(chosen, rejected, beta=1.0)
        # -log σ(10) ≈ 0 (sigmoid(10) ≈ 1)
        assert out.shape == ()
        assert out.item() < 1e-3

    def test_sigmoid_norm_label_smoothing(self) -> None:
        """With ε > 0, output equals the smoothed sigmoid formula."""
        chosen = torch.tensor(1.0)
        rejected = torch.tensor(0.0)
        beta = 0.5
        eps = 0.1
        out = dpo_sigmoid_norm(chosen, rejected, beta=beta, label_smoothing=eps)
        delta = 1.0
        expected = (
            -F.logsigmoid(torch.tensor(beta * delta)) * (1.0 - eps)
            - F.logsigmoid(torch.tensor(-beta * delta)) * eps
        )
        assert torch.allclose(out, expected, atol=1e-6)

    def test_sigmoid_norm_matches_sigmoid_formula(self) -> None:
        """Result matches explicit formula for arbitrary Δ."""
        torch.manual_seed(1)
        chosen = torch.randn(5)
        rejected = torch.randn(5)
        beta = 0.3
        eps = 0.05
        out = dpo_sigmoid_norm(chosen, rejected, beta=beta, label_smoothing=eps)
        delta = chosen - rejected
        ref = (
            -F.logsigmoid(beta * delta) * (1.0 - eps)
            - F.logsigmoid(-beta * delta) * eps
        )
        assert torch.allclose(out, ref, atol=1e-6)

    def test_sigmoid_norm_batched_shape(self) -> None:
        """Batched (B,) inputs: output shape matches input shape."""
        chosen = torch.randn(6)
        rejected = torch.randn(6)
        out = dpo_sigmoid_norm(chosen, rejected, beta=0.2)
        assert out.shape == (6,)

    def test_sigmoid_norm_no_smoothing_equals_sigmoid(self) -> None:
        """With ε = 0, sigmoid_norm equals standard sigmoid formula."""
        torch.manual_seed(2)
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        beta = 0.5
        out_norm = dpo_sigmoid_norm(chosen, rejected, beta=beta, label_smoothing=0.0)
        # Reference: -logsigmoid(beta * (chosen - rejected))
        delta = chosen - rejected
        ref = -F.logsigmoid(beta * delta)
        assert torch.allclose(out_norm, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# dpo_discopop
# ---------------------------------------------------------------------------


class TestDpoDiscopop:
    """Hand-computed reference cases for the DiscoPOP blended loss."""

    def test_discopop_delta_zero(self) -> None:
        """Δ = 0: logits = 0, gate = 0.5, logistic = log 2, exp = 1.
        L = log(2) * 0.5 + 1 * 0.5 = (log(2) + 1) / 2 ≈ 0.8466.
        """
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = dpo_discopop(chosen, rejected, beta=1.0)
        logits = 0.0
        tau = 0.05
        gate = torch.sigmoid(torch.tensor(logits / tau)).item()
        logistic = -F.logsigmoid(torch.tensor(logits)).item()
        exp_comp = math.exp(-logits)
        expected = logistic * (1 - gate) + exp_comp * gate
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(expected), atol=1e-5)

    def test_discopop_large_positive_logits(self) -> None:
        """Large positive logits: gate → 1, exp component dominates → exp(-logits) ≈ 0."""
        chosen = torch.tensor(10.0)
        rejected = torch.tensor(0.0)
        out = dpo_discopop(chosen, rejected, beta=1.0)
        # logits = 10; gate = sigmoid(10/0.05) ≈ 1; exp(-10) ≈ 4.5e-5
        assert out.shape == ()
        assert 0.0 <= out.item() < 1e-3

    def test_discopop_large_negative_logits_finite(self) -> None:
        """Large negative logits: output is finite (exp clamp prevents overflow)."""
        chosen = torch.tensor(-50.0)
        rejected = torch.tensor(0.0)
        out = dpo_discopop(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.isfinite(out)
        # logits = -50; clamped to -80; exp(-clamped) = exp(80) ≈ 5.5e34 but gate ≈ 0
        # so the contribution is nearly zero. Logistic component dominates.
        assert out.item() > 0.0

    def test_discopop_batched_shape(self) -> None:
        """Batched (B,) inputs: output shape matches input shape."""
        chosen = torch.randn(5)
        rejected = torch.randn(5)
        out = dpo_discopop(chosen, rejected, beta=0.5)
        assert out.shape == (5,)

    def test_discopop_per_example_matches_batched(self) -> None:
        """Per-example (0-dim) calls match the corresponding batched row."""
        torch.manual_seed(3)
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        batched = dpo_discopop(chosen, rejected, beta=0.5)
        per_example = torch.stack(
            [dpo_discopop(chosen[i], rejected[i], beta=0.5) for i in range(4)]
        )
        assert torch.allclose(batched, per_example, atol=1e-6)

    def test_discopop_custom_tau(self) -> None:
        """Custom tau=1.0 changes the modulation gate but keeps output finite."""
        chosen = torch.tensor(1.0)
        rejected = torch.tensor(0.0)
        out_default = dpo_discopop(chosen, rejected, beta=1.0, discopop_tau=0.05)
        out_tau1 = dpo_discopop(chosen, rejected, beta=1.0, discopop_tau=1.0)
        # Different tau → different gate → different output
        assert not torch.allclose(out_default, out_tau1, atol=1e-4)
        assert torch.isfinite(out_tau1)

    def test_discopop_nan_injection_one_example(self) -> None:
        """NaN in one example's input propagates only to that example (Tier 1).

        Replaces the second of 4 chosen logratios with NaN and asserts that
        only index 1 is NaN in the output — indices 0, 2, 3 are unaffected.
        """
        chosen = torch.tensor([1.0, float("nan"), -0.5, 0.2])
        rejected = torch.tensor([0.0, 0.0, 0.0, 0.0])
        out = dpo_discopop(chosen, rejected, beta=1.0)
        assert out.shape == (4,)
        assert torch.isnan(out[1])
        assert torch.isfinite(out[0])
        assert torch.isfinite(out[2])
        assert torch.isfinite(out[3])


# ---------------------------------------------------------------------------
# dpo_sft
# ---------------------------------------------------------------------------


class TestDpoSft:
    """Tests for the SFT regulariser: -chosen_logp."""

    def test_sft_nll_scalar(self) -> None:
        """Returns -chosen_logp for a 0-dim tensor."""
        chosen_logp = torch.tensor(-3.5)
        out = dpo_sft(chosen_logp)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(3.5), atol=1e-7)

    def test_sft_nll_batched(self) -> None:
        """Returns -chosen_logp element-wise for a (B,) tensor."""
        chosen_logp = torch.tensor([-1.0, -2.5, -0.3])
        out = dpo_sft(chosen_logp)
        assert out.shape == (3,)
        assert torch.allclose(out, -chosen_logp, atol=1e-7)

    def test_sft_ignores_extra_positional_args(self) -> None:
        """Extra positional arguments (e.g. rejected_logratio) are silently ignored."""
        chosen_logp = torch.tensor(-2.0)
        rejected_logratio = torch.tensor(99.0)  # should be ignored
        out = dpo_sft(chosen_logp, rejected_logratio)
        assert torch.allclose(out, torch.tensor(2.0), atol=1e-7)

    def test_sft_ignores_extra_keyword_args(self) -> None:
        """Extra keyword arguments (beta, label_smoothing, …) are silently ignored."""
        chosen_logp = torch.tensor(-4.0)
        out = dpo_sft(chosen_logp, beta=0.1, label_smoothing=0.05, discopop_tau=0.05)
        assert torch.allclose(out, torch.tensor(4.0), atol=1e-7)

    def test_sft_ignores_both_extra_args(self) -> None:
        """Can be called with full (chosen_logp, rejected, *, beta, ...) signature."""
        chosen_logp = torch.tensor(-1.5)
        rejected = torch.tensor(0.5)
        out = dpo_sft(chosen_logp, rejected, beta=0.2, label_smoothing=0.0)
        assert torch.allclose(out, torch.tensor(1.5), atol=1e-7)

    def test_sft_nan_propagates_only_to_nan_example(self) -> None:
        """NaN in chosen_logp propagates only to that output index."""
        chosen_logp = torch.tensor([-1.0, float("nan"), -2.0, -0.5])
        out = dpo_sft(chosen_logp)
        assert torch.isnan(out[1])
        assert torch.isfinite(out[0])
        assert torch.isfinite(out[2])
        assert torch.isfinite(out[3])


# ---------------------------------------------------------------------------
# dpo_squarechipo
# ---------------------------------------------------------------------------


class TestDpoSquarechipo:
    """Hand-computed reference cases for the SquareChiPO loss (arXiv:2505.21395)."""

    def test_squarechipo_delta_zero_beta_one(self) -> None:
        """Δ = 0, β = 1: 0.5 · (σ(0) − 1)² = 0.5 · (0.5 − 1)² = 0.125."""
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = dpo_squarechipo(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(0.125), atol=1e-6)

    def test_squarechipo_large_positive_delta(self) -> None:
        """Large positive Δ: σ(β·Δ) → 1, loss → 0."""
        chosen = torch.tensor(20.0)
        rejected = torch.tensor(0.0)
        out = dpo_squarechipo(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert out.item() < 1e-5

    def test_squarechipo_large_negative_delta(self) -> None:
        """Large negative Δ: σ(β·Δ) → 0, loss → 0.5 · (0 − 1)² = 0.5."""
        chosen = torch.tensor(-20.0)
        rejected = torch.tensor(0.0)
        out = dpo_squarechipo(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(0.5), atol=1e-4)

    def test_squarechipo_delta_one_beta_one(self) -> None:
        """Δ = 1, β = 1: 0.5 · (σ(1) − 1)²; σ(1) = 1/(1+e^{-1})."""
        chosen = torch.tensor(1.0)
        rejected = torch.tensor(0.0)
        out = dpo_squarechipo(chosen, rejected, beta=1.0)
        sig = torch.sigmoid(torch.tensor(1.0)).item()
        expected = 0.5 * (sig - 1.0) ** 2
        assert torch.allclose(out, torch.tensor(expected), atol=1e-6)

    def test_squarechipo_batched_shape(self) -> None:
        """Batched (B,) inputs: output shape matches input shape."""
        chosen = torch.randn(8)
        rejected = torch.randn(8)
        out = dpo_squarechipo(chosen, rejected, beta=0.5)
        assert out.shape == (8,)

    def test_squarechipo_per_example_matches_batched(self) -> None:
        """Per-example (0-dim) calls match the corresponding batched row."""
        torch.manual_seed(4)
        chosen = torch.randn(6)
        rejected = torch.randn(6)
        batched = dpo_squarechipo(chosen, rejected, beta=0.3)
        per_example = torch.stack(
            [dpo_squarechipo(chosen[i], rejected[i], beta=0.3) for i in range(6)]
        )
        assert torch.allclose(batched, per_example, atol=1e-6)

    def test_squarechipo_nonnegative(self) -> None:
        """Loss is always non-negative (squared term)."""
        torch.manual_seed(5)
        chosen = torch.randn(20)
        rejected = torch.randn(20)
        out = dpo_squarechipo(chosen, rejected, beta=1.0)
        assert (out >= 0.0).all()

    def test_squarechipo_vmap_grad_finite(self) -> None:
        """vmap(grad(dpo_squarechipo)) over a (4,) batch yields finite gradients."""
        torch.manual_seed(6)
        chosen = torch.randn(4)
        rejected = torch.randn(4)

        def per_example(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return dpo_squarechipo(c, r, beta=0.1)

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
        out = dpo_squarechipo(chosen, rejected, beta=1.0)
        assert out.shape == (4,)
        assert torch.isnan(out[2])
        assert torch.isfinite(out[0])
        assert torch.isfinite(out[1])
        assert torch.isfinite(out[3])


# ---------------------------------------------------------------------------
# Cross-variant: vmap(grad) composed over dpo_ipo + dpo_squarechipo
# ---------------------------------------------------------------------------


class TestVmapGradComposed:
    """Additional vmap(grad) safety tests over a (4,) synthetic batch."""

    def test_ipo_vmap_grad_finite_larger_batch(self) -> None:
        """vmap(grad(dpo_ipo)) over a (4,) batch: grads w.r.t. chosen finite."""
        torch.manual_seed(7)
        chosen = torch.randn(4, requires_grad=False)
        rejected = torch.randn(4, requires_grad=False)

        def fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return dpo_ipo(c, r, beta=0.5)

        grads = vmap(grad(fn))(chosen, rejected)
        assert grads.shape == (4,)
        assert torch.isfinite(grads).all()

    def test_squarechipo_vmap_grad_finite_with_grad_wrt_rejected(self) -> None:
        """vmap(grad(dpo_squarechipo, argnums=1)) over a (4,) batch: finite."""
        torch.manual_seed(8)
        chosen = torch.randn(4)
        rejected = torch.randn(4)

        def fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return dpo_squarechipo(c, r, beta=0.2)

        # Gradient w.r.t. rejected (argnums=1)
        grads_r = vmap(grad(fn, argnums=1))(chosen, rejected)
        assert grads_r.shape == (4,)
        assert torch.isfinite(grads_r).all()
