"""Tests for per_group_gaussian — correct PLD accounting for per-group noise."""

import math

import pytest

import opaque_accounting as acc
from opaque_accounting.mechanisms import Gaussian, per_group_gaussian


class TestPerGroupGaussian:
    """per_group_gaussian returns a Gaussian with adjusted noise multiplier."""

    def test_single_group_returns_standard_gaussian(self):
        g = per_group_gaussian(1.1, num_groups=1)
        assert isinstance(g, Gaussian)
        assert g.noise_multiplier == pytest.approx(1.1)

    def test_multiple_groups_adjusts_noise_multiplier(self):
        g = per_group_gaussian(1.1, num_groups=7)
        assert isinstance(g, Gaussian)
        assert g.noise_multiplier == pytest.approx(1.1 / math.sqrt(7))

    def test_invalid_num_groups(self):
        with pytest.raises(ValueError, match="num_groups must be >= 1"):
            per_group_gaussian(1.1, num_groups=0)

    def test_zero_noise_multiplier(self):
        g = per_group_gaussian(0, num_groups=5)
        assert g.noise_multiplier == 0.0


class TestPldEquivalence:
    """PLD of per_group_gaussian(nm, K) must equal K-fold composition of gaussian(nm)."""

    @pytest.mark.parametrize("nm", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("K", [2, 3, 5, 10])
    def test_pld_matches_composition(self, nm, K):
        """PLD_Gaussian(nm)^{*K} == PLD_Gaussian(nm/sqrt(K))."""
        # K-fold composition
        composed = acc.gaussian(nm) * K
        eps_composed = composed.epsilon_at(1e-5)

        # Our shortcut
        pg = per_group_gaussian(nm, num_groups=K)
        eps_pg = pg.epsilon_at(1e-5)

        assert eps_pg == pytest.approx(eps_composed, rel=1e-3), (
            f"PLD mismatch for nm={nm}, K={K}: "
            f"per_group={eps_pg:.6f} vs composed={eps_composed:.6f}"
        )

    def test_more_groups_more_epsilon(self):
        """More groups at the same per-group nm should give higher epsilon."""
        nm = 1.0
        eps_values = []
        for K in [1, 2, 5, 10]:
            eps = per_group_gaussian(nm, K).epsilon_at(1e-5)
            eps_values.append(eps)
        # Strictly increasing
        for i in range(len(eps_values) - 1):
            assert eps_values[i] < eps_values[i + 1]


class TestWithAmplification:
    """per_group_gaussian works correctly in amplified pipelines."""

    def test_poisson_amplified(self):
        nm, K, q = 1.0, 5, 0.01
        step = acc.poisson(per_group_gaussian(nm, K), sample_rate=q)
        eps = step.epsilon_at(1e-5)
        assert eps > 0
        # Should be worse than single-group
        eps_single = acc.poisson(acc.gaussian(nm), sample_rate=q).epsilon_at(1e-5)
        assert eps > eps_single

    def test_with_adaclip(self):
        nm, K = 1.0, 3
        step = acc.poisson(
            acc.adaclip(
                per_group_gaussian(nm, K),
                expected_batch_size=128,
                num_groups=K,
            ),
            sample_rate=0.01,
        )
        eps = step.epsilon_at(1e-5)
        assert eps > 0
