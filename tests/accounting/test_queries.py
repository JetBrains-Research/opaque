"""Tests for query functions.

Tests for get_epsilon, get_delta, get_beta, get_advantage, and get_privacy_curve.
"""

import numpy as np
import pytest

import opaque.accounting as acc


class TestGetEpsilon:
    """Test get_epsilon function."""

    def test_identity_epsilon(self):
        """Test that identity state has zero epsilon."""
        state = acc.create()
        epsilon = acc.get_epsilon(state, delta=1e-5)
        assert epsilon == pytest.approx(0.0, abs=1e-10)

    def test_epsilon_increases_with_steps(self):
        """Test that epsilon increases with more composition steps."""
        state1 = acc.create()
        state1 = acc.compose_poisson_gaussian(
            state1, noise_multiplier=1.0, sample_rate=0.01, count=10
        )
        epsilon1 = acc.get_epsilon(state1, delta=1e-5)

        state2 = acc.create()
        state2 = acc.compose_poisson_gaussian(
            state2, noise_multiplier=1.0, sample_rate=0.01, count=100
        )
        epsilon2 = acc.get_epsilon(state2, delta=1e-5)

        assert epsilon2 > epsilon1

    def test_invalid_delta(self):
        """Test that invalid delta values raise errors."""
        state = acc.create()

        with pytest.raises(ValueError):
            acc.get_epsilon(state, delta=0.0)

        with pytest.raises(ValueError):
            acc.get_epsilon(state, delta=1.0)

        with pytest.raises(ValueError):
            acc.get_epsilon(state, delta=-0.5)


class TestGetDelta:
    """Test get_delta function."""

    def test_identity_delta(self):
        """Test that identity state has zero delta."""
        state = acc.create()
        delta = acc.get_delta(state, epsilon=1.0)
        assert delta == pytest.approx(0.0, abs=1e-10)

    def test_invalid_epsilon(self):
        """Test that invalid epsilon values raise errors."""
        state = acc.create()

        with pytest.raises(ValueError):
            acc.get_delta(state, epsilon=-1.0)


class TestGetBeta:
    """Test get_beta function."""

    def test_get_beta_scalar(self):
        """Test beta query with scalar alpha."""
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=100
        )

        # Query beta for single FPR
        beta = acc.get_beta(state, alpha=0.01)

        assert isinstance(beta, float)
        assert 0.0 <= beta <= 1.0

    def test_get_beta_array(self):
        """Test beta query with array of alphas."""
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=100
        )

        # Query beta for multiple FPRs
        alphas = np.array([0.001, 0.01, 0.05, 0.1])
        betas = acc.get_beta(state, alpha=alphas)

        assert isinstance(betas, np.ndarray)
        assert len(betas) == len(alphas)
        assert np.all(betas >= 0.0)
        assert np.all(betas <= 1.0)

    def test_beta_monotonicity(self):
        """Test that beta decreases as alpha increases (tradeoff property)."""
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=100
        )

        alphas = np.linspace(0.01, 0.99, 50)
        betas = acc.get_beta(state, alpha=alphas)

        # Beta should generally decrease as alpha increases
        # (allowing some numerical tolerance for discrete approximations)
        diff = np.diff(betas)
        assert np.sum(diff < 0) > np.sum(diff > 0)  # Mostly decreasing

    def test_invalid_alpha(self):
        """Test that invalid alpha values raise errors."""
        state = acc.create()

        with pytest.raises(ValueError):
            acc.get_beta(state, alpha=-0.1)

        with pytest.raises(ValueError):
            acc.get_beta(state, alpha=1.5)


class TestGetAdvantage:
    """Test get_advantage function."""

    def test_identity_advantage(self):
        """Test that identity state has zero advantage."""
        state = acc.create()
        advantage = acc.get_advantage(state)
        assert advantage == pytest.approx(0.0, abs=1e-10)

    def test_get_advantage(self):
        """Test advantage computation."""
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=100
        )

        advantage = acc.get_advantage(state)

        assert isinstance(advantage, float)
        assert 0.0 <= advantage <= 1.0

        # Higher noise should give lower advantage
        state_high_noise = acc.create()
        state_high_noise = acc.compose_poisson_gaussian(
            state_high_noise, noise_multiplier=2.0, sample_rate=0.01, count=100
        )
        advantage_high_noise = acc.get_advantage(state_high_noise)

        assert advantage_high_noise < advantage


class TestGetPrivacyCurve:
    """Test get_privacy_curve function."""

    def test_privacy_curve(self):
        """Test full privacy curve computation."""
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=100
        )

        alphas = np.linspace(0, 1, 100)
        alphas_out, betas = acc.get_privacy_curve(state, alphas)

        assert np.array_equal(alphas_out, alphas)
        assert len(betas) == len(alphas)
        assert np.all(betas >= 0.0)
        assert np.all(betas <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
