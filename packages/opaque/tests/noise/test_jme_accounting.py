"""Tests for acc.jme() accounting transformation."""

import math

import pytest

import opaque_accounting as acc


class TestJmeAccounting:
    def test_jme_wraps_band_mf(self):
        mech = acc.band_mf(1.0, sensitivity=1.5, num_groups=10)
        wrapped = acc.jme(mech, zeta=0.1)
        assert isinstance(wrapped, acc.transformations.Jme)

    def test_sensitivity_formula(self):
        mech = acc.band_mf(1.0, sensitivity=1.5, num_groups=10)
        wrapped = acc.jme(mech, zeta=0.1)
        expected = 0.1 * 1.5 * math.sqrt(1.5)
        assert wrapped.sensitivity == pytest.approx(expected, rel=1e-10)

    def test_noise_multiplier_passthrough(self):
        mech = acc.blt(2.0, sensitivity=1.0)
        wrapped = acc.jme(mech, zeta=0.5)
        assert wrapped.noise_multiplier == 2.0

    def test_gram_matrix_passthrough(self):
        gram = (1.0, 0.5, 0.5, 1.0)
        mech = acc.lambda_cgd(1.0, sensitivity=1.2, gram_matrix=gram)
        wrapped = acc.jme(mech, zeta=0.1)
        assert wrapped.gram_matrix == gram

    def test_num_groups_passthrough(self):
        mech = acc.band_mf(1.0, sensitivity=1.0, num_groups=42)
        wrapped = acc.jme(mech, zeta=0.1)
        assert wrapped.num_groups == 42

    def test_max_column_norm_override(self):
        mech = acc.blt(1.0, sensitivity=3.0)
        wrapped = acc.jme(mech, zeta=0.1, max_column_norm=1.5)
        expected = 0.1 * 1.5 * math.sqrt(1.5)
        assert wrapped.sensitivity == pytest.approx(expected, rel=1e-10)

    def test_max_column_norm_fallback(self):
        """Without max_column_norm, falls back to inner.sensitivity."""
        mech = acc.blt(1.0, sensitivity=2.0)
        wrapped = acc.jme(mech, zeta=0.1)
        expected = 0.1 * 2.0 * math.sqrt(1.5)
        assert wrapped.sensitivity == pytest.approx(expected, rel=1e-10)

    def test_cyclic_poisson_accepts_jme(self):
        mech = acc.band_mf(1.0, sensitivity=1.0, num_groups=10)
        wrapped = acc.jme(mech, zeta=0.1)
        proc = acc.cyclic_poisson(wrapped, sample_rate=0.01)
        eps = proc.epsilon_at(1e-5)
        assert eps > 0

    def test_balls_in_bins_accepts_jme_lambda_cgd(self):
        # Gram matrix must be num_bins × num_bins
        num_bins = 2
        gram = (1.0, 0.5, 0.5, 1.0)  # 2×2
        mech = acc.lambda_cgd(1.0, sensitivity=1.2, gram_matrix=gram)
        wrapped = acc.jme(mech, zeta=0.1)
        proc = acc.balls_in_bins(wrapped, num_bins=num_bins, num_epochs=3)
        eps = proc.epsilon_at(1e-5)
        assert eps > 0

    def test_jme_sensitivity_larger_than_base(self):
        """JME sensitivity = ζ·S·√(3/2), which is larger than ζ·S
        (first-moment-only) but may be smaller or larger than S depending
        on ζ. The key property: for the SAME noise multiplier, JME gives
        a higher effective_nm (better privacy per σ) when ζ < 1/√(3/2)."""
        mech = acc.band_mf(1.0, sensitivity=1.0, num_groups=10)
        wrapped = acc.jme(mech, zeta=1.0)

        # With ζ=1: JME sens = 1.0 * √1.5 ≈ 1.22 > base sens 1.0
        assert wrapped.sensitivity > mech.sensitivity

    def test_pld_computes(self):
        mech = acc.blt(1.0, sensitivity=1.5)
        wrapped = acc.jme(mech, zeta=0.1)
        pld = wrapped.pld()
        eps = pld.epsilon_at(1e-5)
        assert eps > 0

    def test_rejects_non_mf(self):
        with pytest.raises(TypeError):
            acc.jme(acc.gaussian(1.0), zeta=0.1)

    def test_rejects_nonpositive_zeta(self):
        mech = acc.band_mf(1.0, sensitivity=1.0, num_groups=1)
        with pytest.raises(ValueError):
            acc.jme(mech, zeta=0.0)
