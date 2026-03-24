"""Tests for opaque.accounting.mechanisms — Gaussian, EpsDelta, Identity, RectifiedGaussian, TruncatedGaussian."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque_accounting as acc
from opaque_accounting import DiscretizationConfig
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms import (
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

    def test_pmf_with_query_config(self):
        """Config is now query-time - test pmf() with different discretization."""
        g = Gaussian(1.0)
        # Both should compute successfully with different query configs
        pmf1 = g.pmf(DiscretizationConfig(discretization=1e-3))
        pmf2 = g.pmf(DiscretizationConfig(discretization=1e-4))
        eps1 = pmf1.epsilon_at(1e-5)
        eps2 = pmf2.epsilon_at(1e-5)
        assert math.isfinite(eps1) and eps1 > 0
        assert math.isfinite(eps2) and eps2 > 0

    def test_cgf_returns_valid(self):
        eps = Gaussian(0.8).cgf().epsilon_at(1e-5)
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

    def test_pmf_returns_valid(self):
        pmf = EpsDelta(1.0, 1e-5).pmf(DiscretizationConfig())
        d = pmf.delta_at(1.0)
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
        # config is None or module default, either way cgf works
        eps = g.cgf().epsilon_at(1e-5)
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
        eps = i.pmf(DiscretizationConfig()).epsilon_at(1e-5)
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
        eps = Identity().pmf(DiscretizationConfig()).epsilon_at(1e-5)
        assert eps == pytest.approx(0.0, abs=1e-10)

    def test_zero_advantage(self):
        adv = Identity().pmf(DiscretizationConfig()).advantage()
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

    def test_pmf_returns_valid(self):
        pmf = RectifiedGaussian(0.8, 5.0).pmf(DiscretizationConfig())
        eps = pmf.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_pmf_with_query_config(self):
        g = RectifiedGaussian(1.0, 5.0)
        pmf1 = g.pmf(DiscretizationConfig(discretization=1e-3))
        pmf2 = g.pmf(DiscretizationConfig(discretization=1e-4))
        eps1 = pmf1.epsilon_at(1e-5)
        eps2 = pmf2.epsilon_at(1e-5)
        assert math.isfinite(eps1) and eps1 > 0
        assert math.isfinite(eps2) and eps2 > 0

    def test_epsilon_le_gaussian(self):
        """Rectified Gaussian should give ε ≤ standard Gaussian (tighter)."""
        nm, R = 1.0, 5.0
        eps_gauss = Gaussian(nm).cgf().epsilon_at(1e-5)
        eps_rect = RectifiedGaussian(nm, R).pmf(DiscretizationConfig()).epsilon_at(1e-5)
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
        eps = g.pmf(DiscretizationConfig()).epsilon_at(1e-5)
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

    def test_pmf_returns_valid(self):
        pmf = TruncatedGaussian(0.8, 5.0).pmf(DiscretizationConfig())
        eps = pmf.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_pmf_with_query_config(self):
        g = TruncatedGaussian(1.0, 5.0)
        pmf1 = g.pmf(DiscretizationConfig(discretization=1e-3))
        pmf2 = g.pmf(DiscretizationConfig(discretization=1e-4))
        eps1 = pmf1.epsilon_at(1e-5)
        eps2 = pmf2.epsilon_at(1e-5)
        assert math.isfinite(eps1) and eps1 > 0
        assert math.isfinite(eps2) and eps2 > 0

    def test_epsilon_le_gaussian(self):
        """Truncated Gaussian should give ε ≤ standard Gaussian (tighter)."""
        nm, R = 1.0, 5.0
        eps_gauss = Gaussian(nm).cgf().epsilon_at(1e-5)
        eps_trunc = TruncatedGaussian(nm, R).pmf(DiscretizationConfig()).epsilon_at(1e-5)
        assert eps_trunc <= eps_gauss + 1e-6

    def test_epsilon_le_rectified(self):
        """Truncated Gaussian should give ε ≤ rectified Gaussian (tightest)."""
        nm, R = 1.0, 5.0
        eps_rect = RectifiedGaussian(nm, R).pmf(DiscretizationConfig()).epsilon_at(1e-5)
        eps_trunc = TruncatedGaussian(nm, R).pmf(DiscretizationConfig()).epsilon_at(1e-5)
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
        eps = g.pmf(DiscretizationConfig()).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0
