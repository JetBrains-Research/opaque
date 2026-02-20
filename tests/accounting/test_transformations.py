"""Tests for opaque.accounting.transformations — AdaClip."""

import pytest

import opaque.accounting as acc
from opaque.accounting.transformations import AdaClip

# ── Constructor function tests ───────────────────────────────────────


class TestAdaclipConstructor:
    """acc.adaclip() returns AdaClip with effective noise multiplier."""

    def test_returns_adaclip(self):
        result = acc.adaclip(acc.gaussian(0.8), 50.0)
        assert isinstance(result, AdaClip)

    def test_effective_noise_differs_from_base(self):
        base = acc.gaussian(0.8)
        result = acc.adaclip(base, 50.0)
        # Effective noise should reduce, so epsilon should increase.
        assert result.epsilon_at(1e-5) > base.epsilon_at(1e-5)

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
