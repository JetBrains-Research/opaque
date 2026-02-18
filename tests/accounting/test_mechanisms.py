"""Tests for opaque.accounting.mechanisms — constructor functions."""

import math

import pytest

import opaque.accounting as acc
from opaque.accounting.base import PldConfig
from opaque.accounting.nodes import Identity
from opaque.accounting.types import (
    Accumulated,
    EpsDelta,
    Gaussian,
    Poisson,
    TruncatedPoisson,
)


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

    def test_pldconfig_discretization(self):
        cfg = PldConfig(discretization=1e-3)
        g = acc.gaussian(0.8, discretization=cfg)
        assert g.config is cfg


class TestPoissonConstructor:
    """acc.poisson() validates inner type and returns Poisson."""

    def test_returns_poisson(self):
        p = acc.poisson(acc.gaussian(0.8), 0.01)
        assert isinstance(p, Poisson)
        assert p.noise_multiplier == pytest.approx(0.8)
        assert p.sample_rate == pytest.approx(0.01)

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            acc.poisson(acc.eps_delta(1.0, 1e-5), 0.01)  # type: ignore[arg-type]

    def test_propagates_config(self):
        cfg = PldConfig(discretization=1e-3)
        g = acc.gaussian(0.8, discretization=cfg)
        p = acc.poisson(g, 0.01)
        assert p.config is cfg


class TestTruncatedPoissonConstructor:
    """acc.truncated_poisson() validates inner type."""

    def test_returns_truncated_poisson(self):
        t = acc.truncated_poisson(acc.gaussian(0.8), 0.01, 128, 10_000)
        assert isinstance(t, TruncatedPoisson)
        assert t.batch_size_cap == 128
        assert t.dataset_size == 10_000

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            acc.truncated_poisson(acc.eps_delta(1.0), 0.01, 128, 10_000)  # type: ignore[arg-type]


class TestAccumulateConstructor:
    """acc.accumulate() validates inner type (must be Poisson)."""

    def test_returns_accumulated(self):
        p = acc.poisson(acc.gaussian(0.8), 0.01)
        a = acc.accumulate(p, 4)
        assert isinstance(a, Accumulated)
        assert a.microbatches == 4

    def test_rejects_non_poisson(self):
        with pytest.raises(TypeError, match="Poisson"):
            acc.accumulate(acc.gaussian(0.8), 4)  # type: ignore[arg-type]


class TestAdaclipConstructor:
    """acc.adaclip() returns Gaussian with effective noise multiplier."""

    def test_returns_gaussian(self):
        result = acc.adaclip(acc.gaussian(0.8), 50.0)
        assert isinstance(result, Gaussian)

    def test_effective_noise_differs_from_base(self):
        result = acc.adaclip(acc.gaussian(0.8), 50.0)
        # effective noise should be lower than base (more privacy cost)
        assert result.noise_multiplier != pytest.approx(0.8)

    def test_large_quantile_noise_approaches_base(self):
        """Very large σ_b → z_eff ≈ z (quantile adds negligible cost)."""
        result = acc.adaclip(acc.gaussian(1.0), 1e10)
        assert result.noise_multiplier == pytest.approx(1.0, rel=1e-6)

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            acc.adaclip(acc.eps_delta(1.0), 50.0)  # type: ignore[arg-type]

    def test_more_privacy_cost_than_base(self):
        """AdaClip ε ≥ base ε (extra cost from quantile noise)."""
        base = acc.gaussian(0.8)
        ac = acc.adaclip(acc.gaussian(0.8), 50.0)
        eps_base = base.epsilon_at(1e-5)
        eps_ac = ac.epsilon_at(1e-5)
        assert eps_ac >= eps_base - 1e-6


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
