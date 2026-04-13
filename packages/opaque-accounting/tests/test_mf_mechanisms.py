"""Tests for MF accounting mechanisms — BandMf, Blt, CyclicPoisson."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque_accounting as acc
from opaque_accounting.amplification import (
    CyclicPoisson,
)
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms import (
    BandMf,
    Blt,
)


# ── BandMf dataclass tests ──────────────────────────────────────────


class TestBandMfDataclass:
    """BandMf frozen dataclass."""

    def test_fields(self):
        proc = BandMf(1.0, 1.0, num_groups=20)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity == pytest.approx(1.0)
        assert proc.num_groups == 20

    def test_frozen(self):
        proc = BandMf(1.0, 1.0, num_groups=20)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(BandMf(1.0, 1.0, num_groups=20), DpProcess)

    def test_equality(self):
        assert BandMf(1.0, 1.0, 20) == BandMf(1.0, 1.0, 20)
        assert BandMf(1.0, 1.0, 20) != BandMf(1.0, 1.0, 10)

    @pytest.mark.slow
    def test_pld_returns_valid(self):
        proc = BandMf(1.0, 1.0, num_groups=20)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestBandMfConstructor:
    """acc.band_mf() returns BandMf."""

    def test_returns_correct_type(self):
        proc = acc.band_mf(1.0, sensitivity=1.0, num_groups=20)
        assert isinstance(proc, BandMf)

    def test_num_groups_default(self):
        proc = acc.band_mf(1.0, sensitivity=1.0)
        assert proc.num_groups == 1


# ── CyclicPoisson tests ─────────────────────────────────────────────


class TestCyclicPoissonDataclass:
    """CyclicPoisson frozen dataclass."""

    def test_fields(self):
        inner = BandMf(1.0, 1.0, num_groups=20)
        proc = CyclicPoisson(inner, 0.01)
        assert proc.inner is inner
        assert proc.sample_rate == pytest.approx(0.01)

    def test_frozen(self):
        proc = CyclicPoisson(BandMf(1.0, 1.0, num_groups=20), 0.01)
        with pytest.raises(FrozenInstanceError):
            proc.sample_rate = 0.5  # type: ignore[misc]

    def test_is_dp_process(self):
        proc = CyclicPoisson(BandMf(1.0, 1.0, num_groups=20), 0.01)
        assert isinstance(proc, DpProcess)

    def test_pld_returns_valid(self):
        proc = acc.cyclic_poisson(
            acc.band_mf(1.0, sensitivity=1.0, num_groups=20),
            0.01,
        )
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_matches_manual_poisson_composition(self):
        """CyclicPoisson(BandMf) should match poisson(gaussian(nm/S)) * k."""
        nm, sensitivity, num_groups, rate = 1.0, 1.0, 20, 0.01
        proc = acc.cyclic_poisson(
            acc.band_mf(nm, sensitivity=sensitivity, num_groups=num_groups),
            rate,
        )

        manual = acc.poisson(acc.gaussian(nm / sensitivity), rate) * num_groups

        eps_proc = proc.epsilon_at(1e-5)
        eps_manual = manual.epsilon_at(1e-5)
        assert eps_proc == pytest.approx(eps_manual, rel=1e-6)

    def test_more_groups_higher_epsilon(self):
        """More groups → more composition → higher epsilon."""
        eps_small = acc.cyclic_poisson(
            acc.band_mf(1.0, sensitivity=1.0, num_groups=2),
            0.01,
        ).epsilon_at(1e-5)
        eps_large = acc.cyclic_poisson(
            acc.band_mf(1.0, sensitivity=1.0, num_groups=20),
            0.01,
        ).epsilon_at(1e-5)
        assert eps_small < eps_large


class TestCyclicPoissonConstructor:
    """acc.cyclic_poisson() validates and returns CyclicPoisson."""

    def test_returns_correct_type(self):
        proc = acc.cyclic_poisson(
            acc.band_mf(1.0, sensitivity=1.0, num_groups=20),
            0.01,
        )
        assert isinstance(proc, CyclicPoisson)

    def test_rejects_non_band_mf(self):
        """cyclic_poisson only accepts BandMf."""
        with pytest.raises(TypeError, match="BandMf"):
            acc.cyclic_poisson(acc.gaussian(1.0), 0.01)  # type: ignore[arg-type]

    def test_rejects_bad_sample_rate(self):
        inner = acc.band_mf(1.0, sensitivity=1.0, num_groups=20)
        with pytest.raises(ValueError):
            acc.cyclic_poisson(inner, 0.0)
        with pytest.raises(ValueError):
            acc.cyclic_poisson(inner, 1.5)


# ── Blt tests ────────────────────────────────────────────────────────


class TestBltDataclass:
    """Blt frozen dataclass."""

    def test_fields(self):
        proc = Blt(1.0, 1.0, gram_matrix=(0.1, 0.2))
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity == pytest.approx(1.0)
        assert proc.gram_matrix == (0.1, 0.2)

    def test_frozen(self):
        proc = Blt(1.0, 1.0)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Blt(1.0, 1.0), DpProcess)

    def test_pld_returns_valid(self):
        proc = acc.blt(1.0, sensitivity=1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestBltConstructor:
    """acc.blt() returns Blt."""

    def test_returns_correct_type(self):
        proc = acc.blt(1.0, sensitivity=1.0)
        assert isinstance(proc, Blt)

    def test_gram_matrix_default_empty(self):
        proc = acc.blt(1.0, sensitivity=1.0)
        assert proc.gram_matrix == ()


# ── Composition tests ───────────────────────────────────────────────


class TestMfComposition:
    """MF mechanisms compose with other DpProcess nodes."""

    def test_band_mf_composes_with_gaussian(self):
        """BandMf | Gaussian works."""
        proc = acc.band_mf(1.0, sensitivity=1.0) | acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_cyclic_poisson_composes_with_gaussian(self):
        """CyclicPoisson | Gaussian works."""
        inner = acc.band_mf(1.0, sensitivity=1.0, num_groups=20)
        proc = acc.cyclic_poisson(inner, 0.01) | acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_class_tree_visible(self):
        """The composition tree preserves class types for debugging."""
        inner = acc.band_mf(1.0, sensitivity=1.0, num_groups=20)
        proc = acc.cyclic_poisson(inner, 0.01)

        assert isinstance(proc, CyclicPoisson)
        assert isinstance(proc.inner, BandMf)
        assert proc.inner.num_groups == 20
