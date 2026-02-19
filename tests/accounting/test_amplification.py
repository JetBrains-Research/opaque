"""Tests for opaque.accounting.amplification — Poisson, TruncatedPoisson, Accumulated."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
from opaque.accounting.amplification import (
    Accumulated,
    Poisson,
    TruncatedPoisson,
)
from opaque.accounting.base import DiscretizationConfig, DpProcess

# ── Amplification dataclass tests ────────────────────────────────────


class TestPoissonDataclass:
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


class TestTruncatedPoissonDataclass:
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


class TestAccumulatedDataclass:
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


# ── Constructor function tests ───────────────────────────────────────


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
        cfg = DiscretizationConfig(discretization=1e-3)
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
