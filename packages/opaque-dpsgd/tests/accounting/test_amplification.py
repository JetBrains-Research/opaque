"""DP-SGD amplification — Poisson (optionally truncated), ParallelPoisson,
RandomAllocation."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.accounting.core._base import DpProcess
from opaque.dpsgd.accounting.amplification.types import (
    ParallelPoisson,
    Poisson,
    RandomAllocation,
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
        assert math.isfinite(eps)
        assert eps > 0


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
        assert math.isfinite(eps)
        assert eps > 0


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
        assert math.isfinite(eps)
        assert eps > 0


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
        with pytest.raises(TypeError, match=r"Gaussian|AdaClip"):
            dpsgd_acc.poisson(acc.eps_delta(1.0, 1e-5), 0.01)  # type: ignore[arg-type]

    def test_accepts_adaclip(self):
        step = dpsgd_acc.poisson(
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000), 0.01
        )
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_propagates_config(self):
        """Config is now query-time, so this test verifies pld() accepts discretization."""
        g = dpsgd_acc.gaussian(0.8)
        p = dpsgd_acc.poisson(g, 0.01)
        pld1 = p.pld(discretization=1e-3)
        pld2 = p.pld(discretization=1e-4)
        eps1 = pld1.epsilon_at(1e-5)
        eps2 = pld2.epsilon_at(1e-5)
        assert math.isfinite(eps1)
        assert eps1 > 0
        assert math.isfinite(eps2)
        assert eps2 > 0


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
        with pytest.raises(TypeError, match=r"Gaussian|AdaClip"):
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
        assert math.isfinite(eps)
        assert eps > 0

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


# ── Random allocation (per-epoch atom) ───────────────────────────────


class TestRandomAllocationDataclass:
    """RandomAllocation frozen dataclass."""

    def test_fields(self):
        g = Gaussian(0.8)
        r = RandomAllocation(g, 16)
        assert r.inner is g
        assert r.num_bins == 16

    def test_frozen(self):
        r = RandomAllocation(Gaussian(0.8), 16)
        with pytest.raises(FrozenInstanceError):
            r.num_bins = 32  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(RandomAllocation(Gaussian(0.8), 16), DpProcess)

    def test_equality(self):
        assert RandomAllocation(Gaussian(0.8), 16) == RandomAllocation(
            Gaussian(0.8), 16
        )
        assert RandomAllocation(Gaussian(0.8), 16) != RandomAllocation(
            Gaussian(0.8), 32
        )

    def test_steps_per_epoch_is_num_bins(self):
        """The conversion factor users need to turn steps into epochs."""
        assert RandomAllocation(Gaussian(0.8), 16).steps_per_epoch == 16

    def test_validates_on_direct_construction(self):
        """Deserialization calls ``cls(**kwargs)``, bypassing the factory, so
        the bound has to live in ``__post_init__``."""
        with pytest.raises(ValueError, match="num_bins"):
            RandomAllocation(Gaussian(0.8), 1)

    @pytest.mark.slow
    def test_pld_returns_valid(self):
        eps = RandomAllocation(Gaussian(1.0), 8).pld().epsilon_at(1e-8)
        assert math.isfinite(eps)
        assert eps > 0


class TestRandomAllocationConstructor:
    """dpsgd_acc.random_allocation() takes (inner, num_bins=...)."""

    def test_returns_random_allocation(self):
        r = dpsgd_acc.random_allocation(dpsgd_acc.gaussian(0.8), num_bins=16)
        assert isinstance(r, RandomAllocation)
        assert r.num_bins == 16

    def test_num_bins_is_keyword_only(self):
        with pytest.raises(TypeError):
            dpsgd_acc.random_allocation(dpsgd_acc.gaussian(0.8), 16)  # type: ignore[misc]

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            dpsgd_acc.random_allocation("bad", num_bins=16)  # type: ignore[arg-type]

    def test_rejects_num_bins_below_two(self):
        with pytest.raises(ValueError, match="num_bins"):
            dpsgd_acc.random_allocation(dpsgd_acc.gaussian(0.8), num_bins=1)

    @pytest.mark.slow
    def test_accepts_adaclip(self):
        r = dpsgd_acc.random_allocation(
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(1.0), expected_batch_size=256),
            num_bins=8,
        )
        eps = r.epsilon_at(1e-8)
        assert math.isfinite(eps)
        assert eps > 0

    def test_nonprivate_inner_is_infinite(self):
        r = dpsgd_acc.random_allocation(acc.nonprivate(), num_bins=8)
        assert math.isinf(r.epsilon_at(1e-8))

    def test_zero_noise_gaussian_is_infinite(self):
        """``Gaussian(0)`` short-circuits before reaching the native primitive,
        which requires ``σ > 0``."""
        r = dpsgd_acc.random_allocation(dpsgd_acc.gaussian(0.0), num_bins=8)
        assert math.isinf(r.epsilon_at(1e-8))


class TestRandomAllocationTightness:
    """The reason the process exists: it beats Poisson at the matched rate."""

    @pytest.mark.slow
    def test_below_poisson_at_matched_rate(self):
        b, sigma, delta = 16, 1.0, 1e-8
        ra = dpsgd_acc.random_allocation(dpsgd_acc.gaussian(sigma), num_bins=b)
        po = dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), 1.0 / b)
        assert ra.epsilon_at(delta) < (po * b).epsilon_at(delta)

    @pytest.mark.slow
    def test_monotone_in_num_bins(self):
        """More bins to hide among, less privacy loss per epoch."""
        sigma, delta = 1.0, 1e-8
        eps = [
            dpsgd_acc.random_allocation(
                dpsgd_acc.gaussian(sigma), num_bins=b
            ).epsilon_at(delta)
            for b in (4, 8, 16)
        ]
        assert eps[0] > eps[1] > eps[2]
