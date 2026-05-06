"""Tests for AdaClip — now in opaque.dpsgd.accounting.mechanisms."""

import math

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.dpsgd.accounting.mechanisms.types import AdaClip

# ── Constructor function tests ───────────────────────────────────────


class TestAdaclipConstructor:
    """dpsgd_acc.adaclip() returns AdaClip with effective noise multiplier."""

    def test_returns_adaclip(self):
        result = dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000)
        assert isinstance(result, AdaClip)

    def test_effective_noise_differs_from_base(self):
        base = dpsgd_acc.gaussian(0.8)
        result = dpsgd_acc.adaclip(base, expected_batch_size=1000)
        # Effective noise should reduce, so epsilon should increase.
        assert result.epsilon_at(1e-5) > base.epsilon_at(1e-5)

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            dpsgd_acc.adaclip(acc.eps_delta(1.0), expected_batch_size=100)  # type: ignore[arg-type]

    def test_more_privacy_cost_than_base(self):
        """AdaClip ε ≥ base ε (extra cost from quantile noise)."""
        base = dpsgd_acc.gaussian(0.8)
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000)
        eps_base = base.epsilon_at(1e-5)
        eps_ac = ac.epsilon_at(1e-5)
        assert eps_ac >= eps_base - 1e-6


class TestEffectiveNoiseMultiplier:
    """Tests for AdaClip.effective_noise_multiplier property."""

    def test_effective_noise_multiplier_is_less_than_base(self):
        """z_eff < z because extra sensitivity from quantile estimator."""
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(1.0), expected_batch_size=100)
        assert ac.effective_noise_multiplier < 1.0
        assert ac.effective_noise_multiplier > 0.0

    def test_effective_noise_multiplier_formula(self):
        z = 1.1
        multiplier = 0.05
        batch_size = 200
        sigma_b = batch_size * multiplier

        ac = dpsgd_acc.adaclip(
            dpsgd_acc.gaussian(z),
            fraction_noise_std=multiplier,
            expected_batch_size=batch_size,
        )
        expected = 1.0 / math.sqrt(1.0 / z**2 + 1.0 / (4.0 * sigma_b**2))
        assert abs(ac.effective_noise_multiplier - expected) < 1e-10

    def test_large_batch_size_negligible_cost(self):
        z = 1.0
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(z), expected_batch_size=100_000)
        assert abs(ac.effective_noise_multiplier - z) < 1e-4

    def test_small_batch_size_significant_cost(self):
        z = 1.0
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(z), expected_batch_size=5)
        assert ac.effective_noise_multiplier < 0.5 * z

    def test_poisson_uses_effective_noise_multiplier(self):
        """Poisson wrapper should use effective_noise_multiplier, not inner.noise_multiplier."""
        base = dpsgd_acc.gaussian(1.0)
        ac = dpsgd_acc.adaclip(base, expected_batch_size=100)

        # Poisson wrapping AdaClip should produce higher epsilon than Poisson wrapping Gaussian(z_eff)
        step_ac = dpsgd_acc.poisson(ac, sample_rate=0.01)
        step_eff = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(ac.effective_noise_multiplier), sample_rate=0.01
        )

        eps_ac = step_ac.epsilon_at(1e-5)
        eps_eff = step_eff.epsilon_at(1e-5)
        # Should match: both use the same z_eff
        assert abs(eps_ac - eps_eff) < 1e-6
