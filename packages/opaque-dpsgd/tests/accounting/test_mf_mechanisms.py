"""Tests for MF accounting mechanisms — BandMf, Blt, Poisson (FTRL)."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.dpsgd.accounting as dpsgd_acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.dpftrl.accounting.amplification.types import MfPoisson
from opaque.accounting._base import DpProcess
from opaque.dpftrl.accounting.mechanisms.types import BandMf, Blt


# ── BandMf dataclass tests ──────────────────────────────────────────


class TestBandMfDataclass:
    """BandMf frozen dataclass."""

    def test_fields(self):
        proc = BandMf(1.0, 1.0, coefficients=(0.9, 0.1))
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity == pytest.approx(1.0)
        assert proc.coefficients == (0.9, 0.1)
        assert proc.bands == 2

    def test_frozen(self):
        proc = BandMf(1.0, 1.0, coefficients=(0.9, 0.1))
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(BandMf(1.0, 1.0, coefficients=(1.0,)), DpProcess)

    def test_equality(self):
        assert BandMf(1.0, 1.0, (0.9, 0.1)) == BandMf(1.0, 1.0, (0.9, 0.1))
        assert BandMf(1.0, 1.0, (0.9, 0.1)) != BandMf(1.0, 1.0, (0.5, 0.5))

    @pytest.mark.slow
    def test_pld_returns_valid(self):
        proc = BandMf(1.0, 1.0, coefficients=(1.0,))
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestBandMfConstructor:
    """ftrl_acc.band_mf() returns BandMf."""

    def test_returns_correct_type(self):
        proc = ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(0.9, 0.1))
        assert isinstance(proc, BandMf)
        assert proc.bands == 2


# ── Poisson (FTRL) tests ────────────────────────────────────────────


class TestFtrlPoissonDataclass:
    """opaque.dpftrl.accounting MfPoisson frozen dataclass."""

    def test_fields(self):
        inner = BandMf(1.0, 1.0, coefficients=(0.9, 0.1))
        proc = MfPoisson(inner, 0.01, n_steps=100)
        assert proc.inner is inner
        assert proc.sample_rate == pytest.approx(0.01)
        assert proc.n_steps == 100

    def test_frozen(self):
        proc = MfPoisson(BandMf(1.0, 1.0, coefficients=(1.0,)), 0.01, n_steps=20)
        with pytest.raises(FrozenInstanceError):
            proc.sample_rate = 0.5  # type: ignore[misc]

    def test_is_dp_process(self):
        proc = MfPoisson(BandMf(1.0, 1.0, coefficients=(1.0,)), 0.01, n_steps=20)
        assert isinstance(proc, DpProcess)

    def test_pld_returns_valid(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0,)),
            sample_rate=0.01,
            n_steps=20,
        )
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_matches_manual_poisson_composition(self):
        """poisson(BandMf(bands=1)) should match poisson(gaussian(nm/S)) * n_steps."""
        nm, sensitivity, n_steps, rate = 1.0, 1.0, 20, 0.01
        proc = ftrl_acc.poisson(
            ftrl_acc.band_mf(nm, sensitivity=sensitivity, coefficients=(1.0,)),
            sample_rate=rate,
            n_steps=n_steps,
        )

        manual = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm / sensitivity), rate) * n_steps

        eps_proc = proc.epsilon_at(1e-5)
        eps_manual = manual.epsilon_at(1e-5)
        assert eps_proc == pytest.approx(eps_manual, rel=1e-6)

    def test_more_steps_higher_epsilon(self):
        """More n_steps → more composition → higher epsilon."""
        eps_small = ftrl_acc.poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0,)),
            sample_rate=0.01,
            n_steps=2,
        ).epsilon_at(1e-5)
        eps_large = ftrl_acc.poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0,)),
            sample_rate=0.01,
            n_steps=20,
        ).epsilon_at(1e-5)
        assert eps_small < eps_large


class TestFtrlPoissonConstructor:
    """ftrl_acc.poisson() validates and returns Poisson."""

    def test_returns_correct_type(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0,)),
            sample_rate=0.01,
            n_steps=20,
        )
        assert isinstance(proc, MfPoisson)

    def test_rejects_unsupported_inner(self):
        with pytest.raises(TypeError, match="BandMf or IdentityMf"):
            ftrl_acc.poisson(dpsgd_acc.gaussian(1.0), 0.01, n_steps=20)  # type: ignore[arg-type]

    def test_rejects_bad_sample_rate(self):
        inner = ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0,))
        with pytest.raises(ValueError):
            ftrl_acc.poisson(inner, 0.0, n_steps=10)
        with pytest.raises(ValueError):
            ftrl_acc.poisson(inner, 1.5, n_steps=10)


# ── Blt tests ────────────────────────────────────────────────────────


class TestBltDataclass:
    """Blt frozen dataclass."""

    def test_fields(self):
        proc = Blt(1.0, 1.0, gram_matrix=(0.1, 0.2))
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity == pytest.approx(1.0)
        assert proc.gram_matrix == (0.1, 0.2)

    def test_frozen(self):
        proc = Blt(1.0, 1.0)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Blt(1.0, 1.0), DpProcess)

    def test_pld_returns_valid(self):
        proc = ftrl_acc.blt(1.0, sensitivity=1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestBltConstructor:
    """ftrl_acc.blt() returns Blt."""

    def test_returns_correct_type(self):
        proc = ftrl_acc.blt(1.0, sensitivity=1.0)
        assert isinstance(proc, Blt)

    def test_gram_matrix_default_empty(self):
        proc = ftrl_acc.blt(1.0, sensitivity=1.0)
        assert proc.gram_matrix == ()


# ── Composition tests ───────────────────────────────────────────────


class TestMfComposition:
    """MF mechanisms compose with other DpProcess nodes."""

    def test_band_mf_composes_with_gaussian(self):
        """BandMf | Gaussian works."""
        proc = ftrl_acc.band_mf(
            1.0, sensitivity=1.0, coefficients=(1.0,)
        ) | dpsgd_acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_poisson_composes_with_gaussian(self):
        """Poisson | Gaussian works."""
        inner = ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0,))
        proc = ftrl_acc.poisson(inner, 0.01, n_steps=20) | dpsgd_acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_class_tree_visible(self):
        """The composition tree preserves class types for debugging."""
        inner = ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0,))
        proc = ftrl_acc.poisson(inner, 0.01, n_steps=20)

        assert isinstance(proc, MfPoisson)
        assert isinstance(proc.inner, BandMf)
        assert proc.inner.bands == 1
