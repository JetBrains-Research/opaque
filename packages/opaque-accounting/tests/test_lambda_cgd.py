"""Tests for DP-λCGD accounting — LambdaCgd mechanism and BnB amplification."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque_accounting as acc
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms import LambdaCgd


# ── LambdaCgd dataclass tests ──────────────────────────────────────


class TestLambdaCgdDataclass:
    def test_fields(self):
        proc = LambdaCgd(1.0, 0.9, 100, 10, 5)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.lambda_ == pytest.approx(0.9)
        assert proc.n_steps == 100
        assert proc.min_sep == 10
        assert proc.max_participations == 5
        assert proc.normalized is True  # default

    def test_fields_unnormalized(self):
        proc = LambdaCgd(1.0, 0.9, 100, 10, 5, normalized=False)
        assert proc.normalized is False

    def test_frozen(self):
        proc = LambdaCgd(1.0, 0.9, 100, 10, 5)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(LambdaCgd(1.0, 0.9, 100, 10, 1), DpProcess)

    def test_equality(self):
        assert LambdaCgd(1.0, 0.9, 100, 10, 5) == LambdaCgd(1.0, 0.9, 100, 10, 5)
        assert LambdaCgd(1.0, 0.9, 100, 10, 5) != LambdaCgd(1.0, 0.5, 100, 10, 5)


# ── Constructor validation ──────────────────────────────────────────


class TestLambdaCgdConstructor:
    def test_returns_correct_type(self):
        proc = acc.lambda_cgd(1.0, lambda_=0.9, n_steps=100)
        assert isinstance(proc, LambdaCgd)

    def test_rejects_non_positive_noise(self):
        with pytest.raises(ValueError):
            acc.lambda_cgd(0.0, lambda_=0.9, n_steps=100)

    def test_rejects_bad_lambda(self):
        with pytest.raises(ValueError):
            acc.lambda_cgd(1.0, lambda_=-0.1, n_steps=100)
        with pytest.raises(ValueError):
            acc.lambda_cgd(1.0, lambda_=1.0, n_steps=100)

    def test_rejects_bad_n_steps(self):
        with pytest.raises(ValueError):
            acc.lambda_cgd(1.0, lambda_=0.9, n_steps=0)

    def test_rejects_bad_min_sep(self):
        with pytest.raises(ValueError):
            acc.lambda_cgd(1.0, lambda_=0.9, n_steps=100, min_sep=0)

    def test_rejects_bad_max_participations(self):
        with pytest.raises(ValueError):
            acc.lambda_cgd(1.0, lambda_=0.9, n_steps=100, max_participations=0)


# ── Sensitivity tests ───────────────────────────────────────────────


class TestLambdaCgdSensitivity:
    def test_lambda_zero_is_dpsgd(self):
        """λ=0 → C_λ = I, sensitivity = sqrt(k) for both normalized and unnormalized."""
        proc_norm = acc.lambda_cgd(
            1.0, lambda_=0.0, n_steps=100, min_sep=10, max_participations=3
        )
        proc_unnorm = acc.lambda_cgd(
            1.0,
            lambda_=0.0,
            n_steps=100,
            min_sep=10,
            max_participations=3,
            normalized=False,
        )
        assert proc_norm.sensitivity() == pytest.approx(math.sqrt(3), rel=1e-6)
        assert proc_unnorm.sensitivity() == pytest.approx(math.sqrt(3), rel=1e-6)

    def test_single_participation_unnormalized_column_norm(self):
        """Unnormalized k=1: sens² = (1 - λ^{2n}) / (1 - λ²)."""
        lam = 0.5
        n = 20
        proc = acc.lambda_cgd(
            1.0,
            lambda_=lam,
            n_steps=n,
            min_sep=n,
            max_participations=1,
            normalized=False,
        )
        expected = sum(lam ** (2 * r) for r in range(n)) ** 0.5
        assert proc.sensitivity() == pytest.approx(expected, rel=1e-6)

    def test_single_participation_normalized_is_one(self):
        """Column-normalized k=1: sensitivity = 1.0 for any λ > 0."""
        for lam in [0.3, 0.5, 0.7, 0.9, 0.99]:
            proc = acc.lambda_cgd(
                1.0, lambda_=lam, n_steps=100, min_sep=100, max_participations=1
            )
            assert proc.sensitivity() == pytest.approx(1.0, abs=1e-8), (
                f"λ={lam}: normalized sens should be 1.0, got {proc.sensitivity()}"
            )

    def test_normalized_leq_unnormalized(self):
        """Normalized sensitivity ≤ unnormalized for any configuration."""
        for lam in [0.3, 0.7, 0.9]:
            for k in [1, 2, 3]:
                norm = acc.lambda_cgd(
                    1.0, lambda_=lam, n_steps=100, min_sep=10, max_participations=k
                ).sensitivity()
                unnorm = acc.lambda_cgd(
                    1.0,
                    lambda_=lam,
                    n_steps=100,
                    min_sep=10,
                    max_participations=k,
                    normalized=False,
                ).sensitivity()
                assert norm <= unnorm + 1e-10, (
                    f"λ={lam}, k={k}: norm={norm} > unnorm={unnorm}"
                )

    def test_sensitivity_increases_with_lambda(self):
        """Higher λ → higher sensitivity for unnormalized (normalized k=1 is always 1)."""
        prev = acc.lambda_cgd(
            1.0,
            lambda_=0.0,
            n_steps=100,
            min_sep=100,
            max_participations=1,
            normalized=False,
        ).sensitivity()
        for lam in [0.3, 0.5, 0.7, 0.9]:
            curr = acc.lambda_cgd(
                1.0,
                lambda_=lam,
                n_steps=100,
                min_sep=100,
                max_participations=1,
                normalized=False,
            ).sensitivity()
            assert curr > prev, f"λ={lam}: {curr} should be > {prev}"
            prev = curr

    def test_sensitivity_increases_with_participations(self):
        """More participations → higher sensitivity."""
        for k in [1, 2, 3, 5]:
            if k == 1:
                prev = acc.lambda_cgd(
                    1.0, lambda_=0.9, n_steps=100, min_sep=10, max_participations=k
                ).sensitivity()
            else:
                curr = acc.lambda_cgd(
                    1.0, lambda_=0.9, n_steps=100, min_sep=10, max_participations=k
                ).sensitivity()
                assert curr > prev

    def test_sensitivity_cached(self):
        proc = acc.lambda_cgd(
            1.0, lambda_=0.9, n_steps=100, min_sep=10, max_participations=3
        )
        s1 = proc.sensitivity()
        s2 = proc.sensitivity()
        assert s1 == s2


# ── PLD / epsilon tests ────────────────────────────────────────────


class TestLambdaCgdPld:
    @pytest.mark.slow
    def test_epsilon_is_finite_positive(self):
        proc = acc.lambda_cgd(
            1.0, lambda_=0.9, n_steps=100, min_sep=10, max_participations=3
        )
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    @pytest.mark.slow
    def test_more_noise_lowers_epsilon(self):
        eps_low = acc.lambda_cgd(
            0.5, lambda_=0.9, n_steps=100, min_sep=10, max_participations=3
        ).epsilon_at(1e-5)
        eps_high = acc.lambda_cgd(
            2.0, lambda_=0.9, n_steps=100, min_sep=10, max_participations=3
        ).epsilon_at(1e-5)
        assert eps_high < eps_low

    @pytest.mark.slow
    def test_lambda_zero_matches_gaussian(self):
        """λ=0, single participation should be close to gaussian(nm)."""
        nm = 1.0
        proc_lcgd = acc.lambda_cgd(
            nm, lambda_=0.0, n_steps=1, min_sep=1, max_participations=1
        )
        proc_gauss = acc.gaussian(nm)
        eps_lcgd = proc_lcgd.epsilon_at(1e-5)
        eps_gauss = proc_gauss.epsilon_at(1e-5)
        assert abs(eps_lcgd - eps_gauss) / eps_gauss < 0.01


# ── BnB amplification tests ────────────────────────────────────────


class TestLambdaCgdBnb:
    @pytest.mark.slow
    def test_bnb_gives_amplification(self):
        """MC BnB should have lower epsilon than unamplified λCGD."""
        nm = 1.0
        lam = 0.9
        n_per_epoch = 100
        n_epochs = 5
        n_total = n_per_epoch * n_epochs

        # No amplification
        proc_no_amp = acc.lambda_cgd(
            nm,
            lambda_=lam,
            n_steps=n_total,
            min_sep=n_per_epoch,
            max_participations=n_epochs,
        )
        eps_no_amp = proc_no_amp.epsilon_at(1e-5)

        # MC BnB amplification (unified multi-epoch path)
        training = acc.balls_in_bins(
            acc.lambda_cgd(
                nm,
                lambda_=lam,
                n_steps=n_total,
                min_sep=n_per_epoch,
                max_participations=n_epochs,
            ),
            num_bins=n_per_epoch,
            num_epochs=n_epochs,
        )
        eps_bnb = training.epsilon_at(1e-5)

        assert eps_bnb < eps_no_amp, (
            f"BnB eps={eps_bnb} should be < no-amp eps={eps_no_amp}"
        )

    @pytest.mark.slow
    def test_bnb_more_bins_reduces_epsilon(self):
        """More bins → smaller per-step sampling rate → better privacy."""
        nm = 1.0
        lam = 0.9
        eps_prev = float("inf")
        for bins in [10, 50, 100]:
            epoch = acc.balls_in_bins(
                acc.lambda_cgd(
                    nm, lambda_=lam, n_steps=bins, min_sep=bins, max_participations=1
                ),
                num_bins=bins,
            )
            eps = (epoch * 5).epsilon_at(1e-5)
            assert eps < eps_prev, f"bins={bins}: eps={eps} should be < {eps_prev}"
            eps_prev = eps

    @pytest.mark.slow
    def test_composition(self):
        """Can compose BnB epochs with * operator."""
        epoch = acc.balls_in_bins(
            acc.lambda_cgd(
                1.0, lambda_=0.9, n_steps=50, min_sep=50, max_participations=1
            ),
            num_bins=50,
        )
        total = epoch * 3
        eps = total.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    @pytest.mark.slow
    def test_normalized_bnb_tighter_than_unnormalized(self):
        """Column-normalized MC BnB should give equal or lower epsilon
        than unnormalized MC BnB (exact vs conservative analysis)."""
        nm = 1.0
        lam = 0.9
        bins = 100
        n_epochs = 5
        n_total = bins * n_epochs

        training_norm = acc.balls_in_bins(
            acc.lambda_cgd(
                nm,
                lambda_=lam,
                n_steps=n_total,
                min_sep=bins,
                max_participations=n_epochs,
                normalized=True,
            ),
            num_bins=bins,
            num_epochs=n_epochs,
        )
        training_unnorm = acc.balls_in_bins(
            acc.lambda_cgd(
                nm,
                lambda_=lam,
                n_steps=n_total,
                min_sep=bins,
                max_participations=n_epochs,
                normalized=False,
            ),
            num_bins=bins,
            num_epochs=n_epochs,
        )

        eps_norm = training_norm.epsilon_at(1e-5)
        eps_unnorm = training_unnorm.epsilon_at(1e-5)

        # Normalized gives lower epsilon (tighter analysis)
        assert eps_norm < eps_unnorm, (
            f"normalized eps={eps_norm} should be < unnormalized eps={eps_unnorm}"
        )

    @pytest.mark.slow
    def test_mc_bnb_more_noise_lowers_epsilon(self):
        """MC BnB: more noise → lower epsilon (monotonicity for calibration)."""
        lam = 0.9
        n_per_epoch = 50
        n_epochs = 3
        n_total = n_per_epoch * n_epochs

        eps_low_noise = acc.balls_in_bins(
            acc.lambda_cgd(
                0.5,
                lambda_=lam,
                n_steps=n_total,
                min_sep=n_per_epoch,
                max_participations=n_epochs,
            ),
            num_bins=n_per_epoch,
            num_epochs=n_epochs,
        ).epsilon_at(1e-5)

        eps_high_noise = acc.balls_in_bins(
            acc.lambda_cgd(
                2.0,
                lambda_=lam,
                n_steps=n_total,
                min_sep=n_per_epoch,
                max_participations=n_epochs,
            ),
            num_bins=n_per_epoch,
            num_epochs=n_epochs,
        ).epsilon_at(1e-5)

        assert eps_high_noise < eps_low_noise, (
            f"high-noise eps={eps_high_noise} should be < low-noise eps={eps_low_noise}"
        )

    @pytest.mark.slow
    def test_mc_bnb_deterministic_with_seed(self):
        """MC BnB gives reproducible epsilon (fixed seed=42 in implementation)."""
        lam = 0.9
        n_per_epoch = 50
        n_epochs = 3
        n_total = n_per_epoch * n_epochs

        def compute():
            return acc.balls_in_bins(
                acc.lambda_cgd(
                    1.0,
                    lambda_=lam,
                    n_steps=n_total,
                    min_sep=n_per_epoch,
                    max_participations=n_epochs,
                ),
                num_bins=n_per_epoch,
                num_epochs=n_epochs,
            ).epsilon_at(1e-5)

        eps1 = compute()
        eps2 = compute()
        assert eps1 == pytest.approx(eps2, rel=1e-10)

    def test_num_epochs_validation(self):
        """num_epochs conflicting with max_participations raises ValueError."""
        with pytest.raises(ValueError, match="conflicts"):
            acc.balls_in_bins(
                acc.lambda_cgd(
                    1.0, lambda_=0.9, n_steps=100, min_sep=20, max_participations=5
                ),
                num_bins=20,
                num_epochs=3,  # conflicts with max_participations=5
            )
