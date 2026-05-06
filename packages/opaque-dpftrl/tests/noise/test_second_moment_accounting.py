"""Tests for acc.second_moment() accounting transformation."""

import math

import pytest

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting.transformations.types import SecondMoment


class TestSecondMomentAccounting:
    def test_wraps_band_mf(self):
        mech = ftrl_acc.band_mf(1.0, sensitivity=1.5, num_groups=10)
        wrapped = acc.second_moment(mech, sensitivity=0.1)
        assert isinstance(wrapped, SecondMoment)

    def test_sensitivity_formula(self):
        mech = ftrl_acc.band_mf(1.0, sensitivity=1.5, num_groups=10)
        wrapped = acc.second_moment(mech, sensitivity=0.1)
        expected = 0.1 * 1.5 * math.sqrt(1.5)
        assert wrapped.sensitivity == pytest.approx(expected, rel=1e-10)

    def test_noise_multiplier_passthrough(self):
        mech = ftrl_acc.blt(2.0, sensitivity=1.0)
        wrapped = acc.second_moment(mech, sensitivity=0.5)
        assert wrapped.noise_multiplier == 2.0

    def test_gram_matrix_passthrough(self):
        gram = (1.0, 0.5, 0.5, 1.0)
        mech = ftrl_acc.lambda_cgd(1.0, sensitivity=1.2, gram_matrix=gram)
        wrapped = acc.second_moment(mech, sensitivity=0.1)
        assert wrapped.gram_matrix == gram

    def test_num_groups_passthrough(self):
        mech = ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=42)
        wrapped = acc.second_moment(mech, sensitivity=0.1)
        assert wrapped.num_groups == 42

    def test_max_column_norm_override(self):
        mech = ftrl_acc.blt(1.0, sensitivity=3.0)
        wrapped = acc.second_moment(mech, sensitivity=0.1, max_column_norm=1.5)
        expected = 0.1 * 1.5 * math.sqrt(1.5)
        assert wrapped.sensitivity == pytest.approx(expected, rel=1e-10)

    def test_max_column_norm_fallback(self):
        """Without max_column_norm, falls back to inner.sensitivity."""
        mech = ftrl_acc.blt(1.0, sensitivity=2.0)
        wrapped = acc.second_moment(mech, sensitivity=0.1)
        expected = 0.1 * 2.0 * math.sqrt(1.5)
        assert wrapped.sensitivity == pytest.approx(expected, rel=1e-10)

    def test_cyclic_poisson_accepts_second_moment(self):
        mech = ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=10)
        wrapped = acc.second_moment(mech, sensitivity=0.1)
        proc = ftrl_acc.cyclic_poisson(wrapped, sample_rate=0.01)
        eps = proc.epsilon_at(1e-5)
        assert eps > 0

    def test_balls_in_bins_accepts_second_moment_lambda_cgd(self):
        # Gram matrix must be num_bins × num_bins.
        num_bins = 2
        gram = (1.0, 0.5, 0.5, 1.0)
        mech = ftrl_acc.lambda_cgd(1.0, sensitivity=1.2, gram_matrix=gram)
        wrapped = acc.second_moment(mech, sensitivity=0.1)
        proc = ftrl_acc.balls_in_bins(wrapped, num_bins=num_bins, num_epochs=3)
        eps = proc.epsilon_at(1e-5)
        assert eps > 0

    def test_second_moment_sensitivity_larger_than_base(self):
        """Default sensitivity is input_sensitivity * S * sqrt(3/2)."""
        mech = ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=10)
        wrapped = acc.second_moment(mech, sensitivity=1.0)
        assert wrapped.sensitivity > mech.sensitivity

    def test_pld_computes(self):
        mech = ftrl_acc.blt(1.0, sensitivity=1.5)
        wrapped = acc.second_moment(mech, sensitivity=0.1)
        pld = wrapped.pld()
        eps = pld.epsilon_at(1e-5)
        assert eps > 0

    def test_accepts_gaussian_inner(self):
        """DP-SGD second-moment: the joint sensitivity collapses to
        ``input_sensitivity · overhead`` (c1 = 1.0 for identity strategy)."""
        mech = dpsgd_acc.gaussian(1.1)
        wrapped = acc.second_moment(mech, sensitivity=0.05)
        assert wrapped.sensitivity > 0
        eps = wrapped.epsilon_at(1e-5)
        assert eps > 0

    def test_rejects_non_gaussian_family(self):
        """Inner must be a Gaussian-family mechanism."""
        with pytest.raises(TypeError):
            acc.second_moment(acc.nonprivate(), sensitivity=0.1)

    def test_rejects_nonpositive_sensitivity(self):
        mech = ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=1)
        with pytest.raises(ValueError):
            acc.second_moment(mech, sensitivity=0.0)
