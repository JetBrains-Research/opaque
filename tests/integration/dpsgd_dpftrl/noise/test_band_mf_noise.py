"""Tests for BandMfStrategy factory and accounting equivalence."""

import math

import pytest
import torch

import opaque.dpftrl.accounting as ftrl_acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.dpftrl.noise._band_mf import BandMfStrategy, band_mf_strategy


_N_STEPS = 100
_BANDS = 10


def _full_part() -> dict:
    return dict(n_steps=_N_STEPS, min_sep=1, max_participations=_N_STEPS)


class TestBandMfStrategy:
    def test_returns_correct_type(self):
        assert isinstance(band_mf_strategy(bands=_BANDS, momentum=0.95), BandMfStrategy)

    def test_sensitivity_is_one(self):
        """Optimized Toeplitz coefficients are L2-normalized."""
        s = band_mf_strategy(bands=_BANDS, momentum=0.95)
        assert s.sensitivity(**_full_part()) == pytest.approx(1.0, abs=1e-6)

    def test_no_gram_matrix(self):
        """BandMF uses Poisson amplification, not BnB — no Gram needed."""
        s = band_mf_strategy(bands=_BANDS, momentum=0.95)
        with pytest.raises(NotImplementedError):
            s.gram_matrix(n_steps=_N_STEPS, min_sep=_BANDS, max_participations=10)

    def test_coefficients_length(self):
        s = band_mf_strategy(bands=_BANDS, momentum=0.95)
        assert len(s.coefficients(n_steps=_N_STEPS)) == _BANDS

    def test_streaming_matrix_present(self):
        s = band_mf_strategy(bands=_BANDS, momentum=0.95)
        assert s.streaming_matrix(n_steps=_N_STEPS) is not None

    def test_rejects_bad_bands(self):
        with pytest.raises(ValueError):
            band_mf_strategy(bands=0)

    def test_with_lr_schedule(self):
        lr = torch.ones(_N_STEPS, dtype=torch.float64) * 0.01
        lr[:10] = torch.linspace(0.001, 0.01, 10)
        s = band_mf_strategy(bands=_BANDS, momentum=0.95, lr_schedule=lr)
        assert s.sensitivity(**_full_part()) == pytest.approx(1.0, abs=1e-6)


class TestBandMfPld:
    delta = 1e-5

    def test_band_mf_pld(self):
        s = band_mf_strategy(bands=_BANDS, momentum=0.95)
        eps = ftrl_acc.mf_gaussian(1.0, s, **_full_part()).epsilon_at(self.delta)
        assert eps > 0

    def test_poisson_matches_manual(self):
        """Poisson via band_mf matches manual poisson composition."""
        s = band_mf_strategy(bands=_BANDS, momentum=0.95)
        sample_rate = 0.05

        eps_new = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, s),
            sample_rate=sample_rate,
            n_steps=_N_STEPS,
        ).epsilon_at(self.delta)
        sens = s.sensitivity(n_steps=_N_STEPS)
        num_groups = math.ceil(_N_STEPS / _BANDS)
        eps_manual = (
            dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0 / sens), sample_rate) * num_groups
        ).epsilon_at(self.delta)
        assert eps_new == pytest.approx(eps_manual, abs=1e-10)
