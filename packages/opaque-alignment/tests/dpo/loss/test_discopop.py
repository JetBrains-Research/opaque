"""Tests for the DPO DiscoPOP blended loss variant.

Covers work-unit γ.3 of the opaque-alignment plan (§10, §11.2, §11.3).

Hand-computed reference cases plus a NaN-injection DP-purity contract test
(§11.3) for :func:`discopop_loss`.

Imports target concrete implementation paths because the public façade
`__init__.py` is wired in the separate γ.W wire-up unit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from opaque.api.alignment.dpo.loss._discopop import discopop_loss

# ---------------------------------------------------------------------------
# discopop_loss
# ---------------------------------------------------------------------------


class TestDpoDiscopop:
    """Hand-computed reference cases for the DiscoPOP blended loss."""

    def test_discopop_delta_zero(self) -> None:
        """Δ = 0: logits = 0, gate = 0.5, logistic = log 2, exp = 1.
        L = log(2) * 0.5 + 1 * 0.5 = (log(2) + 1) / 2 ≈ 0.8466.
        """
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = discopop_loss(chosen, rejected, beta=1.0)
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
        out = discopop_loss(chosen, rejected, beta=1.0)
        # logits = 10; gate = sigmoid(10/0.05) ≈ 1; exp(-10) ≈ 4.5e-5
        assert out.shape == ()
        assert 0.0 <= out.item() < 1e-3

    def test_discopop_large_negative_logits_finite(self) -> None:
        """Large negative logits: output is finite (exp clamp prevents overflow)."""
        chosen = torch.tensor(-50.0)
        rejected = torch.tensor(0.0)
        out = discopop_loss(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.isfinite(out)
        # logits = -50; clamped to -80; exp(-clamped) = exp(80) ≈ 5.5e34 but gate ≈ 0
        # so the contribution is nearly zero. Logistic component dominates.
        assert out.item() > 0.0

    def test_discopop_batched_shape(self) -> None:
        """Batched (B,) inputs: output shape matches input shape."""
        chosen = torch.randn(5)
        rejected = torch.randn(5)
        out = discopop_loss(chosen, rejected, beta=0.5)
        assert out.shape == (5,)

    def test_discopop_per_example_matches_batched(self) -> None:
        """Per-example (0-dim) calls match the corresponding batched row."""
        torch.manual_seed(3)
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        batched = discopop_loss(chosen, rejected, beta=0.5)
        per_example = torch.stack(
            [discopop_loss(chosen[i], rejected[i], beta=0.5) for i in range(4)]
        )
        assert torch.allclose(batched, per_example, atol=1e-6)

    def test_discopop_custom_tau(self) -> None:
        """Custom tau=1.0 changes the modulation gate but keeps output finite."""
        chosen = torch.tensor(1.0)
        rejected = torch.tensor(0.0)
        out_default = discopop_loss(chosen, rejected, beta=1.0, discopop_tau=0.05)
        out_tau1 = discopop_loss(chosen, rejected, beta=1.0, discopop_tau=1.0)
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
        out = discopop_loss(chosen, rejected, beta=1.0)
        assert out.shape == (4,)
        assert torch.isnan(out[1])
        assert torch.isfinite(out[0])
        assert torch.isfinite(out[2])
        assert torch.isfinite(out[3])
