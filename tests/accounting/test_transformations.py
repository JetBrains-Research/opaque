"""Tests for opaque.accounting.transformations — AdaClip."""

import pytest

import opaque.accounting as acc
from opaque.accounting.mechanisms import Gaussian

# ── Constructor function tests ───────────────────────────────────────


class TestAdaclipConstructor:
    """acc.adaclip() returns Gaussian with effective noise multiplier."""

    def test_returns_gaussian(self):
        result = acc.adaclip(acc.gaussian(0.8), 50.0)
        assert isinstance(result, Gaussian)

    def test_effective_noise_differs_from_base(self):
        result = acc.adaclip(acc.gaussian(0.8), 50.0)
        # effective noise should be lower than base (more privacy cost)
        assert result.noise_multiplier != pytest.approx(0.8)

    def test_large_quantile_noise_approaches_base(self):
        """Very large σ_b → z_eff ≈ z (quantile adds negligible cost)."""
        result = acc.adaclip(acc.gaussian(1.0), 1e10)
        assert result.noise_multiplier == pytest.approx(1.0, rel=1e-6)

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            acc.adaclip(acc.eps_delta(1.0), 50.0)  # type: ignore[arg-type]

    def test_more_privacy_cost_than_base(self):
        """AdaClip ε ≥ base ε (extra cost from quantile noise)."""
        base = acc.gaussian(0.8)
        ac = acc.adaclip(acc.gaussian(0.8), 50.0)
        eps_base = base.epsilon_at(1e-5)
        eps_ac = ac.epsilon_at(1e-5)
        assert eps_ac >= eps_base - 1e-6
