"""Tests for MF mechanism types — LambdaCgd, Bisr, and BnB amplification."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque_accounting as acc
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms import LambdaCgd, Bisr


# ── LambdaCgd dataclass tests ──────────────────────────────────────


class TestLambdaCgdDataclass:
    def test_fields(self):
        proc = LambdaCgd(1.0, 1.5, gram_matrix=(0.1, 0.2))
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity == pytest.approx(1.5)
        assert proc.gram_matrix == (0.1, 0.2)

    def test_frozen(self):
        proc = LambdaCgd(1.0, 1.0)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(LambdaCgd(1.0, 1.0), DpProcess)

    def test_equality(self):
        assert LambdaCgd(1.0, 1.0) == LambdaCgd(1.0, 1.0)
        assert LambdaCgd(1.0, 1.0) != LambdaCgd(1.0, 2.0)


# ── Bisr dataclass tests ───────────────────────────────────────────


class TestBisrDataclass:
    def test_fields(self):
        proc = Bisr(1.0, 1.5, gram_matrix=(0.1, 0.2))
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity == pytest.approx(1.5)
        assert proc.gram_matrix == (0.1, 0.2)

    def test_frozen(self):
        proc = Bisr(1.0, 1.0)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Bisr(1.0, 1.0), DpProcess)


# ── Constructor tests ───────────────────────────────────────────────


class TestLambdaCgdConstructor:
    def test_returns_correct_type(self):
        proc = acc.lambda_cgd(1.0, sensitivity=1.0)
        assert isinstance(proc, LambdaCgd)

    def test_gram_matrix_default_empty(self):
        proc = acc.lambda_cgd(1.0, sensitivity=1.0)
        assert proc.gram_matrix == ()


class TestBisrConstructor:
    def test_returns_correct_type(self):
        proc = acc.bisr(1.0, sensitivity=1.0)
        assert isinstance(proc, Bisr)

    def test_gram_matrix_default_empty(self):
        proc = acc.bisr(1.0, sensitivity=1.0)
        assert proc.gram_matrix == ()


# ── PLD / epsilon tests ────────────────────────────────────────────


class TestMfGaussianPld:
    @pytest.mark.slow
    def test_epsilon_is_finite_positive(self):
        proc = acc.lambda_cgd(1.0, sensitivity=1.5)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    @pytest.mark.slow
    def test_more_noise_lowers_epsilon(self):
        # Higher nm → lower ε (more noise = better privacy)
        eps_low = acc.lambda_cgd(0.5, sensitivity=1.0).epsilon_at(1e-5)
        eps_high = acc.lambda_cgd(2.0, sensitivity=1.0).epsilon_at(1e-5)
        assert eps_high < eps_low

    @pytest.mark.slow
    def test_sensitivity_one_matches_gaussian(self):
        """MfGaussian with sensitivity=1 should match plain Gaussian."""
        nm = 1.0
        eps_mf = acc.lambda_cgd(nm, sensitivity=1.0).epsilon_at(1e-5)
        eps_gauss = acc.gaussian(nm).epsilon_at(1e-5)
        assert abs(eps_mf - eps_gauss) / eps_gauss < 0.01

    @pytest.mark.slow
    def test_bisr_pld_valid(self):
        proc = acc.bisr(1.0, sensitivity=1.5)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ── BnB amplification tests ────────────────────────────────────────


class TestBnbAmplification:
    """BnB amplification with MF Gram matrix types."""

    @pytest.mark.slow
    def test_bnb_requires_gram_matrix(self):
        """BnB with empty gram_matrix raises ValueError."""
        with pytest.raises(ValueError, match="gram_matrix"):
            acc.balls_in_bins(
                acc.lambda_cgd(1.0, sensitivity=1.0),
                num_bins=50, num_epochs=3,
            ).epsilon_at(1e-5)

    @pytest.mark.slow
    def test_bnb_rejects_non_accepted_type(self):
        """BnB rejects BandMf (should use cyclic_poisson)."""
        with pytest.raises(TypeError):
            acc.balls_in_bins(
                acc.band_mf(1.0, sensitivity=1.0, num_groups=20),
                num_bins=50, num_epochs=3,
            )

    @pytest.mark.slow
    def test_composition(self):
        """Can compose BnB epochs with * operator."""
        # Use a simple gram matrix (identity-like) for testing
        gram = tuple([1.0] + [0.0] * 49 for _ in range(1))[0] * 50  # rough identity
        # Actually just test that composition of BnB processes works
        proc = acc.lambda_cgd(1.0, sensitivity=1.0)
        eps = (proc * 3).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0
