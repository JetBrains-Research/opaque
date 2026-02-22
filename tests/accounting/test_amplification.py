"""Tests for opaque.accounting.amplification — Poisson, TruncatedPoisson, ParallelPoisson."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
from opaque.accounting.amplification import (
    ParallelPoisson,
    Poisson,
    TruncatedPoisson,
)
from opaque.accounting.base import DpProcess
from opaque.accounting.discretization import DiscretizationConfig
from opaque.accounting.mechanisms import Gaussian

# ── Amplification dataclass tests ────────────────────────────────────


class TestPoissonDataclass:
    """Poisson frozen dataclass."""

    def test_fields(self):
        g = Gaussian(0.8)
        p = Poisson(g, 0.01)
        assert p.inner is g
        assert p.sample_rate == pytest.approx(0.01)

    def test_frozen(self):
        p = Poisson(Gaussian(0.8), 0.01)
        with pytest.raises(FrozenInstanceError):
            p.sample_rate = 0.1  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Poisson(Gaussian(0.8), 0.01), DpProcess)

    def test_equality(self):
        assert Poisson(Gaussian(0.8), 0.01) == Poisson(Gaussian(0.8), 0.01)
        assert Poisson(Gaussian(0.8), 0.01) != Poisson(Gaussian(0.8), 0.02)

    def test_pld_returns_valid(self):
        pld = Poisson(Gaussian(0.8), 0.01).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestTruncatedPoissonDataclass:
    """TruncatedPoisson frozen dataclass."""

    def test_fields(self):
        g = Gaussian(0.8)
        t = TruncatedPoisson(g, 0.01, 128, 10_000)
        assert t.inner is g
        assert t.sample_rate == pytest.approx(0.01)
        assert t.batch_size_cap == 128
        assert t.dataset_size == 10_000

    def test_frozen(self):
        t = TruncatedPoisson(Gaussian(0.8), 0.01, 128, 10_000)
        with pytest.raises(FrozenInstanceError):
            t.batch_size_cap = 256  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(TruncatedPoisson(Gaussian(0.8), 0.01, 128, 10_000), DpProcess)

    def test_pld_returns_valid(self):
        pld = TruncatedPoisson(Gaussian(0.8), 0.01, 128, 10_000).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestParallelPoissonDataclass:
    """ParallelPoisson frozen dataclass."""

    def test_fields(self):
        inner = Poisson(Gaussian(0.8), 0.01)
        a = ParallelPoisson(inner, 4)
        assert a.inner is inner
        assert a.num_workers == 4

    def test_frozen(self):
        a = ParallelPoisson(Poisson(Gaussian(0.8), 0.01), 4)
        with pytest.raises(FrozenInstanceError):
            a.num_workers = 8  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(ParallelPoisson(Poisson(Gaussian(0.8), 0.01), 4), DpProcess)

    def test_pld_returns_valid(self):
        pld = ParallelPoisson(Poisson(Gaussian(0.8), 0.01), 4).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ── Constructor function tests ───────────────────────────────────────


class TestPoissonConstructor:
    """acc.poisson() validates inner type and returns Poisson."""

    def test_returns_poisson(self):
        p = acc.poisson(acc.gaussian(0.8), 0.01)
        assert isinstance(p, Poisson)
        assert isinstance(p.inner, Gaussian)
        assert p.inner.noise_multiplier == pytest.approx(0.8)
        assert p.sample_rate == pytest.approx(0.01)

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian|AdaClip"):
            acc.poisson(acc.eps_delta(1.0, 1e-5), 0.01)  # type: ignore[arg-type]

    def test_accepts_adaclip(self):
        step = acc.poisson(acc.adaclip(acc.gaussian(0.8), batch_size=1000), 0.01)
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_propagates_config(self):
        """Config is now query-time, so this test verifies pld() accepts discretization."""
        g = acc.gaussian(0.8)
        p = acc.poisson(g, 0.01)
        # Config is query-time - verify pld() accepts discretization parameter
        pld1 = p.pld(discretization=1e-3)
        pld2 = p.pld(discretization=1e-4)
        # Both should compute successfully (different discretizations)
        eps1 = pld1.epsilon_at(1e-5)
        eps2 = pld2.epsilon_at(1e-5)
        assert math.isfinite(eps1) and eps1 > 0
        assert math.isfinite(eps2) and eps2 > 0


class TestTruncatedPoissonConstructor:
    """acc.truncated_poisson() validates inner type."""

    def test_returns_truncated_poisson(self):
        t = acc.truncated_poisson(acc.gaussian(0.8), 0.01, 128, 10_000)
        assert isinstance(t, TruncatedPoisson)
        assert t.batch_size_cap == 128
        assert t.dataset_size == 10_000

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian|AdaClip"):
            acc.truncated_poisson(acc.eps_delta(1.0), 0.01, 128, 10_000)  # type: ignore[arg-type]

    def test_accepts_adaclip(self):
        step = acc.truncated_poisson(
            acc.adaclip(acc.gaussian(0.8), batch_size=1000),
            0.01,
            128,
            10_000,
        )
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestParallelPoissonConstructor:
    """acc.parallel_poisson() takes (Gaussian, sample_rate, num_workers)."""

    def test_returns_parallel_poisson(self):
        a = acc.parallel_poisson(acc.gaussian(0.8), sample_rate=0.01, num_workers=4)
        assert isinstance(a, ParallelPoisson)
        assert a.num_workers == 4

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            acc.parallel_poisson("bad", sample_rate=0.01, num_workers=4)  # type: ignore[arg-type]
