"""Tests for DP-SGD amplification — Poisson (with optional truncation), ParallelPoisson."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.accounting.core._base import DpProcess
from opaque.dpsgd.accounting.amplification.types import (
    ParallelPoisson,
    Poisson,
)
from opaque.dpsgd.accounting.mechanisms.types import Gaussian

# ── Amplification dataclass tests ────────────────────────────────────


class TestPoissonDataclass:
    """Poisson frozen dataclass (plain Poisson)."""

    def test_fields(self):
        g = Gaussian(0.8)
        p = Poisson(g, 0.01)
        assert p.inner is g
        assert p.sample_rate == pytest.approx(0.01)
        assert p.truncated_batch_size is None
        assert p.dataset_size is None

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


class TestPoissonTruncatedDataclass:
    """Poisson with truncation switched on."""

    def test_fields(self):
        g = Gaussian(0.8)
        t = Poisson(g, 0.01, truncated_batch_size=128, dataset_size=10_000)
        assert t.inner is g
        assert t.sample_rate == pytest.approx(0.01)
        assert t.truncated_batch_size == 128
        assert t.dataset_size == 10_000

    def test_pld_returns_valid(self):
        pld = Poisson(
            Gaussian(0.8), 0.01, truncated_batch_size=128, dataset_size=10_000
        ).pld()
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
    """dpsgd_acc.poisson() validates inner type and returns Poisson."""

    def test_returns_poisson(self):
        p = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
        assert isinstance(p, Poisson)
        assert isinstance(p.inner, Gaussian)
        assert p.inner.noise_multiplier == pytest.approx(0.8)
        assert p.sample_rate == pytest.approx(0.01)
        assert p.truncated_batch_size is None
        assert p.dataset_size is None

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian|AdaClip"):
            dpsgd_acc.poisson(acc.eps_delta(1.0, 1e-5), 0.01)  # type: ignore[arg-type]

    def test_accepts_adaclip(self):
        step = dpsgd_acc.poisson(
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000), 0.01
        )
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_propagates_config(self):
        """Config is now query-time, so this test verifies pld() accepts discretization."""
        g = dpsgd_acc.gaussian(0.8)
        p = dpsgd_acc.poisson(g, 0.01)
        pld1 = p.pld(discretization=1e-3)
        pld2 = p.pld(discretization=1e-4)
        eps1 = pld1.epsilon_at(1e-5)
        eps2 = pld2.epsilon_at(1e-5)
        assert math.isfinite(eps1) and eps1 > 0
        assert math.isfinite(eps2) and eps2 > 0


class TestPoissonTruncatedConstructor:
    """dpsgd_acc.poisson(..., truncated_batch_size=..., dataset_size=...)."""

    def test_returns_truncated_poisson(self):
        t = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(0.8),
            0.01,
            truncated_batch_size=128,
            dataset_size=10_000,
        )
        assert isinstance(t, Poisson)
        assert t.truncated_batch_size == 128
        assert t.dataset_size == 10_000

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian|AdaClip"):
            dpsgd_acc.poisson(
                acc.eps_delta(1.0),
                0.01,
                truncated_batch_size=128,
                dataset_size=10_000,
            )  # type: ignore[arg-type]

    def test_accepts_adaclip(self):
        step = dpsgd_acc.poisson(
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000),
            0.01,
            truncated_batch_size=128,
            dataset_size=10_000,
        )
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_requires_both_truncation_args(self):
        with pytest.raises(ValueError, match="truncated_batch_size and dataset_size"):
            dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01, truncated_batch_size=128)
        with pytest.raises(ValueError, match="truncated_batch_size and dataset_size"):
            dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01, dataset_size=10_000)


class TestParallelPoissonConstructor:
    """dpsgd_acc.parallel_poisson() takes (Gaussian, sample_rate, num_workers)."""

    def test_returns_parallel_poisson(self):
        a = dpsgd_acc.parallel_poisson(
            dpsgd_acc.gaussian(0.8), sample_rate=0.01, num_workers=4
        )
        assert isinstance(a, ParallelPoisson)
        assert a.num_workers == 4

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            dpsgd_acc.parallel_poisson("bad", sample_rate=0.01, num_workers=4)  # type: ignore[arg-type]


# ── Bounded Gaussian amplification tests ─────────────────────────────


class TestParallelPoissonAutoTruncation:
    """Automatic truncation from query-time discretization settings."""

    def test_auto_respects_query_time_discretization_overrides(self):
        nm = 0.8
        q = 0.0032
        m = 8
        delta = 1e-8
        auto = dpsgd_acc.parallel_poisson(
            dpsgd_acc.gaussian(nm), sample_rate=q, num_workers=m
        )

        eps_tight = auto.epsilon_at(delta, log_x_mass_truncation_bound=-50.0)
        eps_loose = auto.epsilon_at(delta, log_x_mass_truncation_bound=-15.0)

        assert eps_loose >= eps_tight - 1e-10
