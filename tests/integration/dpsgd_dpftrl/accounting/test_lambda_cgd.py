"""Tests for MF mechanism types — LambdaCgd, Bisr, and BnB amplification."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.dpsgd.accounting as dpsgd_acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core._base import DpProcess
from opaque.dpftrl.noise import bisr_strategy, lambda_cgd_strategy


def _lambda_cgd_mech(noise_multiplier: float = 1.0, **kw):
    s = lambda_cgd_strategy(
        kw.pop("lambda_", 0.5),
        n_steps=kw.pop("n_steps", 10),
        min_sep=kw.pop("min_sep", 1),
        max_participations=kw.pop("max_participations", 1),
    )
    return s.as_mechanism(noise_multiplier)


def _bisr_mech(noise_multiplier: float = 1.0, **kw):
    s = bisr_strategy(
        bandwidth=kw.pop("bandwidth", 2),
        n_steps=kw.pop("n_steps", 10),
        min_sep=kw.pop("min_sep", 1),
        max_participations=kw.pop("max_participations", 1),
    )
    return s.as_mechanism(noise_multiplier)


# ── LambdaCgd dataclass tests ──────────────────────────────────────


class TestLambdaCgdDataclass:
    def test_fields_via_strategy(self):
        proc = _lambda_cgd_mech(1.0, lambda_=0.5)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity > 0
        assert proc.lambda_ == pytest.approx(0.5)
        assert isinstance(proc.gram_matrix, tuple)

    def test_frozen(self):
        proc = _lambda_cgd_mech()
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(_lambda_cgd_mech(), DpProcess)

    def test_equality(self):
        a = _lambda_cgd_mech(1.0)
        b = _lambda_cgd_mech(1.0)
        assert a == b
        assert _lambda_cgd_mech(1.0) != _lambda_cgd_mech(2.0)


# ── Bisr dataclass tests ───────────────────────────────────────────


class TestBisrDataclass:
    def test_fields_via_strategy(self):
        proc = _bisr_mech(1.0)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity > 0
        assert isinstance(proc.gram_matrix, tuple)

    def test_frozen(self):
        proc = _bisr_mech()
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(_bisr_mech(), DpProcess)


# ── PLD / epsilon tests ────────────────────────────────────────────


class TestMfGaussianPld:
    @pytest.mark.slow
    def test_epsilon_is_finite_positive(self):
        eps = _lambda_cgd_mech(1.0).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    @pytest.mark.slow
    def test_more_noise_lowers_epsilon(self):
        # Higher nm → lower ε (more noise = better privacy)
        eps_low = _lambda_cgd_mech(0.5).epsilon_at(1e-5)
        eps_high = _lambda_cgd_mech(2.0).epsilon_at(1e-5)
        assert eps_high < eps_low

    @pytest.mark.slow
    def test_single_step_matches_gaussian(self):
        """λCGD at n_steps=1 (lambda=0) reduces to a single Gaussian."""
        nm = 1.0
        s = lambda_cgd_strategy(0.0, n_steps=1, min_sep=1, max_participations=1)
        eps_mf = s.as_mechanism(nm).epsilon_at(1e-5)
        eps_gauss = dpsgd_acc.gaussian(nm).epsilon_at(1e-5)
        assert abs(eps_mf - eps_gauss) / eps_gauss < 0.01

    @pytest.mark.slow
    def test_bisr_pld_valid(self):
        eps = _bisr_mech(1.0).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ── BnB amplification tests ────────────────────────────────────────


class TestBnbAmplification:
    """BnB amplification with MF Gram matrix types."""

    @pytest.mark.slow
    def test_bnb_rejects_non_accepted_type(self):
        """BnB rejects BandMf (should use poisson)."""
        with pytest.raises(TypeError):
            ftrl_acc.balls_in_bins(
                ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0,)),
                num_bins=50,
                n_steps=150,
            )

    @pytest.mark.slow
    def test_composition(self):
        """Can compose with * operator."""
        proc = _lambda_cgd_mech(1.0)
        eps = (proc * 3).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0
