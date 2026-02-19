"""Tests for opaque.accounting.types — frozen mechanism dataclasses."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
from opaque.accounting.base import DiscretizationConfig, DpProcess
from opaque.accounting.types import (
    Accumulated,
    EpsDelta,
    Gaussian,
    Poisson,
    TruncatedPoisson,
)


class TestGaussian:
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

    def test_config_excluded_from_hash(self):
        """Config has hash=False — excluded from __hash__ but still in __eq__."""
        a = Gaussian(1.0, config=None)
        b = Gaussian(1.0, config=DiscretizationConfig(discretization=1e-3))
        # config is NOT excluded from __eq__ (only from __hash__)
        # so same nm + different config → NOT equal
        assert a != b
        # But same nm + same config → equal and same hash
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


class TestPoisson:
    """Poisson frozen dataclass."""

    def test_fields(self):
        p = Poisson(0.8, 0.01)
        assert p.noise_multiplier == pytest.approx(0.8)
        assert p.sample_rate == pytest.approx(0.01)

    def test_frozen(self):
        p = Poisson(0.8, 0.01)
        with pytest.raises(FrozenInstanceError):
            p.sample_rate = 0.1  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Poisson(0.8, 0.01), DpProcess)

    def test_equality(self):
        assert Poisson(0.8, 0.01) == Poisson(0.8, 0.01)
        assert Poisson(0.8, 0.01) != Poisson(0.8, 0.02)

    def test_pld_returns_valid(self):
        pld = Poisson(0.8, 0.01).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestTruncatedPoisson:
    """TruncatedPoisson frozen dataclass."""

    def test_fields(self):
        t = TruncatedPoisson(0.8, 0.01, 128, 10_000)
        assert t.noise_multiplier == pytest.approx(0.8)
        assert t.sample_rate == pytest.approx(0.01)
        assert t.batch_size_cap == 128
        assert t.dataset_size == 10_000

    def test_frozen(self):
        t = TruncatedPoisson(0.8, 0.01, 128, 10_000)
        with pytest.raises(FrozenInstanceError):
            t.batch_size_cap = 256  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(TruncatedPoisson(0.8, 0.01, 128, 10_000), DpProcess)

    def test_pld_returns_valid(self):
        pld = TruncatedPoisson(0.8, 0.01, 128, 10_000).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestAccumulated:
    """Accumulated frozen dataclass."""

    def test_fields(self):
        a = Accumulated(0.8, 0.01, 4)
        assert a.noise_multiplier == pytest.approx(0.8)
        assert a.sample_rate == pytest.approx(0.01)
        assert a.microbatches == 4

    def test_frozen(self):
        a = Accumulated(0.8, 0.01, 4)
        with pytest.raises(FrozenInstanceError):
            a.microbatches = 8  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Accumulated(0.8, 0.01, 4), DpProcess)

    def test_pld_returns_valid(self):
        pld = Accumulated(0.8, 0.01, 4).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestEpsDelta:
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
