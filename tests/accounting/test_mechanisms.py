"""Tests for opaque.accounting.mechanisms — Gaussian, EpsDelta, Identity, BoundedGaussian."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
from opaque.accounting.base import DiscretizationConfig, DpProcess
from opaque.accounting.mechanisms import (
    BoundedGaussian,
    EpsDelta,
    Gaussian,
    Identity,
)

# ── Mechanism dataclass tests ────────────────────────────────────────


class TestGaussianDataclass:
    """Gaussian frozen dataclass."""

    def test_fields(self):
        g = Gaussian(1.1)
        assert g.noise_multiplier == pytest.approx(1.1)
        assert g.config is None

    def test_frozen(self):
        g = Gaussian(1.1)
        with pytest.raises(FrozenInstanceError):
            g.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Gaussian(1.0), DpProcess)

    def test_equality(self):
        assert Gaussian(1.0) == Gaussian(1.0)
        assert Gaussian(1.0) != Gaussian(1.1)

    def test_config_participates_in_equality_and_hash(self):
        """Config participates in both __eq__ and __hash__."""
        a = Gaussian(1.0, config=None)
        b = Gaussian(1.0, config=DiscretizationConfig(discretization=1e-3))
        # same nm + different config → NOT equal
        assert a != b
        # same nm + same config → equal and same hash
        c = Gaussian(1.0, config=None)
        assert a == c
        assert hash(a) == hash(c)

    def test_config_excluded_from_repr(self):
        """Config field has repr=False."""
        g = Gaussian(1.0, config=DiscretizationConfig(discretization=1e-3))
        assert "config" not in repr(g)

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

    def test_float_discretization(self):
        g = acc.gaussian(0.8, discretization=1e-3)
        assert g.config is not None
        assert g.config.discretization == pytest.approx(1e-3)

    def test_config_discretization(self):
        cfg = DiscretizationConfig(discretization=1e-3)
        g = acc.gaussian(0.8, discretization=cfg)
        assert g.config is cfg


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

    def test_float_discretization(self):
        e = acc.eps_delta(1.0, discretization=1e-3)
        assert e.config is not None


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

    def test_frozen(self):
        i = Identity()
        with pytest.raises(FrozenInstanceError):
            i.config = None  # type: ignore[misc]

    def test_zero_epsilon(self):
        eps = Identity().epsilon_at(1e-5)
        assert eps == pytest.approx(0.0, abs=1e-10)

    def test_zero_advantage(self):
        adv = Identity().advantage()
        assert adv == pytest.approx(0.0, abs=1e-10)

    def test_equality(self):
        assert Identity() == Identity()


# ── BoundedGaussian dataclass and constructor tests ──────────────────


class TestBoundedGaussianDataclass:
    """BoundedGaussian frozen dataclass."""

    def test_fields(self):
        bg = BoundedGaussian(1.1)
        assert bg.noise_multiplier == pytest.approx(1.1)
        assert bg.config is None

    def test_frozen(self):
        bg = BoundedGaussian(1.1)
        with pytest.raises(FrozenInstanceError):
            bg.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(BoundedGaussian(1.0), DpProcess)

    def test_equality(self):
        assert BoundedGaussian(1.0) == BoundedGaussian(1.0)
        assert BoundedGaussian(1.0) != BoundedGaussian(1.1)

    def test_config_participates_in_equality_and_hash(self):
        a = BoundedGaussian(1.0, config=None)
        b = BoundedGaussian(1.0, config=DiscretizationConfig(discretization=1e-3))
        assert a != b
        c = BoundedGaussian(1.0, config=None)
        assert a == c
        assert hash(a) == hash(c)

    def test_config_excluded_from_repr(self):
        bg = BoundedGaussian(1.0, config=DiscretizationConfig(discretization=1e-3))
        assert "config" not in repr(bg)

    def test_pld_returns_valid(self):
        pld = BoundedGaussian(1.1).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_higher_epsilon_than_gaussian(self):
        """Replace adjacency doubles sensitivity → higher epsilon than Gaussian."""
        bg_eps = BoundedGaussian(1.1).epsilon_at(1e-5)
        g_eps = Gaussian(1.1).epsilon_at(1e-5)
        assert bg_eps > g_eps

    def test_equals_gaussian_with_half_noise_multiplier(self):
        """bounded_gaussian(σ) has the same PLD as gaussian(σ/2)."""
        bg_eps = BoundedGaussian(1.1).epsilon_at(1e-5)
        g_half_eps = Gaussian(1.1 / 2.0).epsilon_at(1e-5)
        assert bg_eps == pytest.approx(g_half_eps, rel=1e-6)

    def test_rejects_out_of_range(self):
        """noise_multiplier=0.05 is below minimum — should raise."""
        with pytest.raises(ValueError):
            BoundedGaussian(0.05).pld()

    def test_rejects_above_max(self):
        """noise_multiplier=1.21 is above maximum — should raise."""
        with pytest.raises(ValueError):
            BoundedGaussian(1.21).pld()


class TestBoundedGaussianConstructor:
    """acc.bounded_gaussian() returns BoundedGaussian with correct config."""

    def test_returns_bounded_gaussian(self):
        bg = acc.bounded_gaussian(1.1)
        assert isinstance(bg, BoundedGaussian)
        assert bg.noise_multiplier == pytest.approx(1.1)

    def test_default_config_none(self):
        bg = acc.bounded_gaussian(1.1)
        eps = bg.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_float_discretization(self):
        bg = acc.bounded_gaussian(1.1, discretization=1e-3)
        assert bg.config is not None
        assert bg.config.discretization == pytest.approx(1e-3)

    def test_config_discretization(self):
        cfg = DiscretizationConfig(discretization=1e-3)
        bg = acc.bounded_gaussian(1.1, discretization=cfg)
        assert bg.config is cfg

    def test_composition(self):
        """BoundedGaussian can be composed like any DpProcess."""
        training = acc.bounded_gaussian(1.1) * 1000
        eps = training.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0
