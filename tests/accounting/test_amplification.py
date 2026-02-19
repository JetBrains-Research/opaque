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


# ── BoundedGaussian + amplification tests ───────────────────────────


class TestPoissonWithBoundedGaussian:
    """poisson() accepts BoundedGaussian and computes correct PLD."""

    def test_accepts_bounded_gaussian(self):
        """poisson(bounded_gaussian(nm), q) should not raise."""
        import math

        step = acc.poisson(acc.bounded_gaussian(1.1), 0.01)
        assert isinstance(step, Poisson)
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_equivalent_to_poisson_gaussian_half_nm(self):
        """poisson(bounded_gaussian(nm), q) == poisson(gaussian(nm/2), q)."""
        nm = 1.0
        q = 0.01
        bg_step = acc.poisson(acc.bounded_gaussian(nm), q)
        g_step = acc.poisson(acc.gaussian(nm / 2.0), q)
        assert bg_step.epsilon_at(1e-5) == pytest.approx(g_step.epsilon_at(1e-5), rel=1e-6)

    def test_propagates_config(self):
        """Config is propagated from BoundedGaussian to Poisson."""
        cfg = DiscretizationConfig(discretization=1e-3)
        bg = acc.bounded_gaussian(0.8, discretization=cfg)
        step = acc.poisson(bg, 0.01)
        assert step.config is cfg

    def test_stored_nm_is_halved(self):
        """Poisson.noise_multiplier stores nm/2 (effective Remove-adjacency nm)."""
        bg = acc.bounded_gaussian(1.0)
        step = acc.poisson(bg, 0.01)
        assert step.noise_multiplier == pytest.approx(0.5)

    def test_rejects_eps_delta(self):
        """Non-Gaussian, non-BoundedGaussian inner still raises TypeError."""
        with pytest.raises(TypeError, match="Gaussian"):
            acc.poisson(acc.eps_delta(1.0, 1e-5), 0.01)  # type: ignore[arg-type]


class TestTruncatedPoissonWithBoundedGaussian:
    """truncated_poisson() accepts BoundedGaussian and computes correct PLD."""

    def test_accepts_bounded_gaussian(self):
        """truncated_poisson(bounded_gaussian(nm), ...) should not raise."""
        import math

        step = acc.truncated_poisson(acc.bounded_gaussian(1.0), 0.01, 128, 10_000)
        assert isinstance(step, TruncatedPoisson)
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_equivalent_to_truncated_poisson_gaussian_half_nm(self):
        """truncated_poisson(bounded_gaussian(nm), ...) == truncated_poisson(gaussian(nm/2), ...)."""
        nm = 1.0
        bg_step = acc.truncated_poisson(acc.bounded_gaussian(nm), 0.01, 128, 10_000)
        g_step = acc.truncated_poisson(acc.gaussian(nm / 2.0), 0.01, 128, 10_000)
        assert bg_step.epsilon_at(1e-5) == pytest.approx(g_step.epsilon_at(1e-5), rel=1e-6)

    def test_stored_nm_is_halved(self):
        """TruncatedPoisson.noise_multiplier stores nm/2 (effective nm)."""
        bg = acc.bounded_gaussian(1.0)
        step = acc.truncated_poisson(bg, 0.01, 128, 10_000)
        assert step.noise_multiplier == pytest.approx(0.5)

    def test_rejects_eps_delta(self):
        """Non-Gaussian, non-BoundedGaussian inner still raises TypeError."""
        with pytest.raises(TypeError, match="Gaussian"):
            acc.truncated_poisson(acc.eps_delta(1.0), 0.01, 128, 10_000)  # type: ignore[arg-type]
