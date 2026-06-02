"""Tests for the DPO length-normalised sigmoid loss variant.

Covers work-unit γ.3 of the opaque-alignment plan (§10, §11.2, §11.3).

Hand-computed reference cases for :func:`sigmoid_norm_loss` (the "norm"
variant), checked against the smoothed sigmoid formula.

Imports target concrete implementation paths because the public façade
`__init__.py` is wired in the separate γ.W wire-up unit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from opaque.api.alignment.dpo.loss._sigmoid_norm import sigmoid_norm_loss

# ---------------------------------------------------------------------------
# sigmoid_norm_loss
# ---------------------------------------------------------------------------


class TestDpoSigmoidNorm:
    """Hand-computed reference cases for the length-normalised sigmoid loss."""

    def test_sigmoid_norm_delta_zero_no_smoothing(self) -> None:
        """Δ = 0, β = 1, ε = 0: -log σ(0) = log 2."""
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = sigmoid_norm_loss(chosen, rejected, beta=1.0)
        expected = math.log(2.0)  # -log(sigmoid(0)) = -log(0.5) = log 2
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(expected), atol=1e-6)

    def test_sigmoid_norm_large_positive_delta(self) -> None:
        """Large positive Δ → loss near 0 (chosen clearly preferred)."""
        chosen = torch.tensor(10.0)
        rejected = torch.tensor(0.0)
        out = sigmoid_norm_loss(chosen, rejected, beta=1.0)
        # -log σ(10) ≈ 0 (sigmoid(10) ≈ 1)
        assert out.shape == ()
        assert out.item() < 1e-3

    def test_sigmoid_norm_label_smoothing(self) -> None:
        """With ε > 0, output equals the smoothed sigmoid formula."""
        chosen = torch.tensor(1.0)
        rejected = torch.tensor(0.0)
        beta = 0.5
        eps = 0.1
        out = sigmoid_norm_loss(chosen, rejected, beta=beta, label_smoothing=eps)
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
        out = sigmoid_norm_loss(chosen, rejected, beta=beta, label_smoothing=eps)
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
        out = sigmoid_norm_loss(chosen, rejected, beta=0.2)
        assert out.shape == (6,)

    def test_sigmoid_norm_no_smoothing_equals_sigmoid(self) -> None:
        """With ε = 0, sigmoid_norm equals standard sigmoid formula."""
        torch.manual_seed(2)
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        beta = 0.5
        out_norm = sigmoid_norm_loss(chosen, rejected, beta=beta, label_smoothing=0.0)
        # Reference: -logsigmoid(beta * (chosen - rejected))
        delta = chosen - rejected
        ref = -F.logsigmoid(beta * delta)
        assert torch.allclose(out_norm, ref, atol=1e-6)
