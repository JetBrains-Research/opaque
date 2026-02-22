"""Tests for opaque.accounting.mechanisms — Gaussian, EpsDelta, Identity, RectifiedGaussian, TruncatedGaussian, BandMfAmplified."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
from opaque.accounting.base import DpProcess
from opaque.accounting.discretization import DiscretizationConfig
from opaque.accounting.mechanisms import (
    BandMfAmplified,
    EpsDelta,
    Gaussian,
    Identity,
    RectifiedGaussian,
    TruncatedGaussian,
)

# ── Mechanism dataclass tests ────────────────────────────────────────


class TestGaussianDataclass:
    """Gaussian frozen dataclass."""

    def test_fields(self):
        g = Gaussian(1.1)
        assert g.noise_multiplier == pytest.approx(1.1)

    def test_frozen(self):
        g = Gaussian(1.1)
        with pytest.raises(FrozenInstanceError):
            g.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Gaussian(1.0), DpProcess)

    def test_equality(self):
        assert Gaussian(1.0) == Gaussian(1.0)
        assert Gaussian(1.0) != Gaussian(1.1)

    def test_pld_with_query_config(self):
        """Config is now query-time - test pld() with different discretization."""
        g = Gaussian(1.0)
        # Both should compute successfully with different query configs
        pld1 = g.pld(discretization=1e-3)
        pld2 = g.pld(discretization=1e-4)
        eps1 = pld1.epsilon_at(1e-5)
        eps2 = pld2.epsilon_at(1e-5)
        assert math.isfinite(eps1) and eps1 > 0
        assert math.isfinite(eps2) and eps2 > 0

    def test_pld_returns_valid(self):
        pld = Gaussian(0.8).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestEpsDeltaDataclass:
    """EpsDelta frozen dataclass."""

    def test_fields(self):
        e = EpsDelta(1.0, 1e-5)
        assert e.epsilon == pytest.approx(1.0)
        assert e.delta == pytest.approx(1e-5)

    def test_frozen(self):
        e = EpsDelta(1.0, 1e-5)
        with pytest.raises(FrozenInstanceError):
            e.epsilon = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(EpsDelta(1.0, 1e-5), DpProcess)

    def test_equality(self):
        assert EpsDelta(1.0, 1e-5) == EpsDelta(1.0, 1e-5)
        assert EpsDelta(1.0, 1e-5) != EpsDelta(2.0, 1e-5)

    def test_pld_returns_valid(self):
        pld = EpsDelta(1.0, 1e-5).pld()
        d = pld.delta_at(1.0)
        assert math.isfinite(d)


# ── Constructor function tests ───────────────────────────────────────


class TestGaussianConstructor:
    """acc.gaussian() returns Gaussian with correct config."""

    def test_returns_gaussian(self):
        g = acc.gaussian(1.1)
        assert isinstance(g, Gaussian)
        assert g.noise_multiplier == pytest.approx(1.1)

    def test_default_config_none(self):
        g = acc.gaussian(1.1)
        # config is None or module default, either way pld works
        eps = g.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestEpsDeltaConstructor:
    """acc.eps_delta() returns EpsDelta."""

    def test_returns_eps_delta(self):
        e = acc.eps_delta(1.0, 1e-5)
        assert isinstance(e, EpsDelta)
        assert e.epsilon == pytest.approx(1.0)
        assert e.delta == pytest.approx(1e-5)

    def test_pure_dp(self):
        """Default delta=0 → pure ε-DP."""
        e = acc.eps_delta(1.0)
        assert e.delta == pytest.approx(0.0)


class TestIdentityConstructor:
    """acc.identity() returns Identity."""

    def test_returns_identity(self):
        i = acc.identity()
        assert isinstance(i, Identity)

    def test_zero_epsilon(self):
        i = acc.identity()
        eps = i.epsilon_at(1e-5)
        assert eps == pytest.approx(0.0, abs=1e-10)


class TestIdentityDataclass:
    """Identity node — zero privacy loss."""

    def test_is_dp_process(self):
        assert isinstance(Identity(), DpProcess)

    def test_is_dataclass(self):
        """Identity is a dataclass."""
        i = Identity()
        import dataclasses

        assert dataclasses.is_dataclass(i)

    def test_zero_epsilon(self):
        eps = Identity().epsilon_at(1e-5)
        assert eps == pytest.approx(0.0, abs=1e-10)

    def test_zero_advantage(self):
        adv = Identity().advantage()
        assert adv == pytest.approx(0.0, abs=1e-10)

    def test_equality(self):
        assert Identity() == Identity()


# ── RectifiedGaussian tests ──────────────────────────────────────────


class TestRectifiedGaussianDataclass:
    """RectifiedGaussian frozen dataclass."""

    def test_fields(self):
        g = RectifiedGaussian(1.1, 5.0)
        assert g.noise_multiplier == pytest.approx(1.1)
        assert g.radius == pytest.approx(5.0)

    def test_frozen(self):
        g = RectifiedGaussian(1.1, 5.0)
        with pytest.raises(FrozenInstanceError):
            g.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(RectifiedGaussian(1.0, 5.0), DpProcess)

    def test_equality(self):
        assert RectifiedGaussian(1.0, 5.0) == RectifiedGaussian(1.0, 5.0)
        assert RectifiedGaussian(1.0, 5.0) != RectifiedGaussian(1.1, 5.0)
        assert RectifiedGaussian(1.0, 5.0) != RectifiedGaussian(1.0, 3.0)

    def test_pld_returns_valid(self):
        pld = RectifiedGaussian(0.8, 5.0).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_pld_with_query_config(self):
        g = RectifiedGaussian(1.0, 5.0)
        pld1 = g.pld(discretization=1e-3)
        pld2 = g.pld(discretization=1e-4)
        eps1 = pld1.epsilon_at(1e-5)
        eps2 = pld2.epsilon_at(1e-5)
        assert math.isfinite(eps1) and eps1 > 0
        assert math.isfinite(eps2) and eps2 > 0

    def test_epsilon_le_gaussian(self):
        """Rectified Gaussian should give ε ≤ standard Gaussian (tighter)."""
        nm, R = 1.0, 5.0
        eps_gauss = Gaussian(nm).epsilon_at(1e-5)
        eps_rect = RectifiedGaussian(nm, R).epsilon_at(1e-5)
        assert eps_rect <= eps_gauss + 1e-6


class TestRectifiedGaussianConstructor:
    """acc.rectified_gaussian() returns RectifiedGaussian."""

    def test_returns_rectified_gaussian(self):
        g = acc.rectified_gaussian(1.1, 5.0)
        assert isinstance(g, RectifiedGaussian)
        assert g.noise_multiplier == pytest.approx(1.1)
        assert g.radius == pytest.approx(5.0)

    def test_epsilon_at_works(self):
        g = acc.rectified_gaussian(1.1, 5.0)
        eps = g.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ── TruncatedGaussian tests ─────────────────────────────────────────


class TestTruncatedGaussianDataclass:
    """TruncatedGaussian frozen dataclass."""

    def test_fields(self):
        g = TruncatedGaussian(1.1, 5.0)
        assert g.noise_multiplier == pytest.approx(1.1)
        assert g.radius == pytest.approx(5.0)

    def test_frozen(self):
        g = TruncatedGaussian(1.1, 5.0)
        with pytest.raises(FrozenInstanceError):
            g.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(TruncatedGaussian(1.0, 5.0), DpProcess)

    def test_equality(self):
        assert TruncatedGaussian(1.0, 5.0) == TruncatedGaussian(1.0, 5.0)
        assert TruncatedGaussian(1.0, 5.0) != TruncatedGaussian(1.1, 5.0)
        assert TruncatedGaussian(1.0, 5.0) != TruncatedGaussian(1.0, 3.0)

    def test_pld_returns_valid(self):
        pld = TruncatedGaussian(0.8, 5.0).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_pld_with_query_config(self):
        g = TruncatedGaussian(1.0, 5.0)
        pld1 = g.pld(discretization=1e-3)
        pld2 = g.pld(discretization=1e-4)
        eps1 = pld1.epsilon_at(1e-5)
        eps2 = pld2.epsilon_at(1e-5)
        assert math.isfinite(eps1) and eps1 > 0
        assert math.isfinite(eps2) and eps2 > 0

    def test_epsilon_le_gaussian(self):
        """Truncated Gaussian should give ε ≤ standard Gaussian (tighter)."""
        nm, R = 1.0, 5.0
        eps_gauss = Gaussian(nm).epsilon_at(1e-5)
        eps_trunc = TruncatedGaussian(nm, R).epsilon_at(1e-5)
        assert eps_trunc <= eps_gauss + 1e-6

    def test_epsilon_le_rectified(self):
        """Truncated Gaussian should give ε ≤ rectified Gaussian (tightest)."""
        nm, R = 1.0, 5.0
        eps_rect = RectifiedGaussian(nm, R).epsilon_at(1e-5)
        eps_trunc = TruncatedGaussian(nm, R).epsilon_at(1e-5)
        assert eps_trunc <= eps_rect + 1e-6


class TestTruncatedGaussianConstructor:
    """acc.truncated_gaussian() returns TruncatedGaussian."""

    def test_returns_truncated_gaussian(self):
        g = acc.truncated_gaussian(1.1, 5.0)
        assert isinstance(g, TruncatedGaussian)
        assert g.noise_multiplier == pytest.approx(1.1)
        assert g.radius == pytest.approx(5.0)

    def test_epsilon_at_works(self):
        g = acc.truncated_gaussian(1.1, 5.0)
        eps = g.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ── BandMfAmplified dataclass tests ─────────────────────────────────


class TestBandMfAmplifiedDataclass:
    """BandMfAmplified frozen dataclass."""

    def test_fields(self):
        proc = BandMfAmplified(1.0, 2.5, 0.01, 200)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity == pytest.approx(2.5)
        assert proc.sample_rate == pytest.approx(0.01)
        assert proc.num_groups == 200

    def test_frozen(self):
        proc = BandMfAmplified(1.0, 2.5, 0.01, 200)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(BandMfAmplified(1.0, 2.5, 0.01, 200), DpProcess)

    def test_pld_returns_valid(self):
        proc = BandMfAmplified(1.0, 1.0, 0.01, 10)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_matches_manual_composition(self):
        """BandMfAmplified should match poisson(gaussian(...)) * k."""
        nm, sens, rate, k = 1.0, 2.0, 0.01, 50
        proc = acc.band_mf_amplified(nm, sens, rate, k)
        manual = acc.poisson(acc.gaussian(nm / sens), rate) * k
        eps_proc = proc.epsilon_at(1e-5)
        eps_manual = manual.epsilon_at(1e-5)
        assert eps_proc == pytest.approx(eps_manual, rel=1e-6)

    def test_diagonal_matches_standard_poisson(self):
        """With sensitivity=1 (diagonal strategy), matches standard poisson * n."""
        n = 100
        nm, rate = 1.0, 0.01
        proc = acc.band_mf_amplified(nm, 1.0, rate, n)
        standard = acc.poisson(acc.gaussian(nm), rate) * n
        eps_proc = proc.epsilon_at(1e-5)
        eps_std = standard.epsilon_at(1e-5)
        assert eps_proc == pytest.approx(eps_std, rel=1e-6)

    def test_more_groups_higher_epsilon(self):
        """More groups should give higher epsilon (more composition)."""
        eps_10 = acc.band_mf_amplified(1.0, 1.0, 0.01, 10).epsilon_at(1e-5)
        eps_50 = acc.band_mf_amplified(1.0, 1.0, 0.01, 50).epsilon_at(1e-5)
        assert eps_10 < eps_50


class TestBandMfAmplifiedConstructor:
    """acc.band_mf_amplified() validates and returns BandMfAmplified."""

    def test_returns_correct_type(self):
        proc = acc.band_mf_amplified(1.0, 2.5, 0.01, 200)
        assert isinstance(proc, BandMfAmplified)

    def test_rejects_non_positive_noise(self):
        with pytest.raises(ValueError):
            acc.band_mf_amplified(0.0, 1.0, 0.01, 10)

    def test_rejects_non_positive_sensitivity(self):
        with pytest.raises(ValueError):
            acc.band_mf_amplified(1.0, 0.0, 0.01, 10)

    def test_rejects_bad_sample_rate(self):
        with pytest.raises(ValueError):
            acc.band_mf_amplified(1.0, 1.0, 0.0, 10)
        with pytest.raises(ValueError):
            acc.band_mf_amplified(1.0, 1.0, 1.5, 10)

    def test_rejects_bad_num_groups(self):
        with pytest.raises(ValueError):
            acc.band_mf_amplified(1.0, 1.0, 0.01, 0)
