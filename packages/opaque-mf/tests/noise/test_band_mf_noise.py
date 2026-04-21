"""Tests for BandMfStrategy factory and accounting equivalence."""

import pytest
import torch

import opaque.accounting as acc
from opaque.mf.noise.band_mf import BandMfStrategy, band_mf_strategy


class TestBandMfStrategy:
    def test_returns_correct_type(self):
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert isinstance(s, BandMfStrategy)

    def test_sensitivity_is_one(self):
        """Optimized Toeplitz coefficients are L2-normalized."""
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert s.sensitivity == pytest.approx(1.0, abs=1e-6)

    def test_gram_matrix_is_none(self):
        """BandMF uses cyclic_poisson, not BnB."""
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert s.gram_matrix is None

    def test_coefficients_length(self):
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert len(s.coefficients) == 10

    def test_streaming_matrix_present(self):
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert s._streaming_matrix is not None

    def test_matches_old_sensitivity(self):
        acc.band_mf(1.0, sensitivity=1.0, num_groups=10)
        new = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert new.sensitivity == pytest.approx(1.0, abs=1e-6)

    def test_rejects_bad_n_steps(self):
        with pytest.raises(ValueError):
            band_mf_strategy(n_steps=0, bands=5)

    def test_rejects_bad_bands(self):
        with pytest.raises(ValueError):
            band_mf_strategy(n_steps=100, bands=0)
        with pytest.raises(ValueError):
            band_mf_strategy(n_steps=100, bands=101)

    def test_with_lr_schedule(self):
        lr = torch.ones(100, dtype=torch.float64) * 0.01
        lr[:10] = torch.linspace(0.001, 0.01, 10)
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95, lr_schedule=lr)
        assert s.sensitivity == pytest.approx(1.0, abs=1e-6)


class TestBandMfPld:
    delta = 1e-5

    def test_band_mf_pld(self):
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        eps = acc.band_mf(1.0, sensitivity=s.sensitivity).epsilon_at(self.delta)
        assert eps > 0

    def test_cyclic_poisson_matches_manual(self):
        """Cyclic Poisson via band_mf matches manual poisson composition."""
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        sample_rate = 0.05

        eps_new = acc.cyclic_poisson(
            acc.band_mf(1.0, sensitivity=s.sensitivity, num_groups=s.num_groups),
            sample_rate=sample_rate,
        ).epsilon_at(self.delta)
        eps_manual = (
            acc.poisson(acc.gaussian(1.0 / s.sensitivity), sample_rate) * s.num_groups
        ).epsilon_at(self.delta)
        assert eps_new == pytest.approx(eps_manual, abs=1e-10)
