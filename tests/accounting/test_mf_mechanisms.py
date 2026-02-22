"""Tests for MF accounting mechanisms — BandMf, BltMf, DenseMf, CyclicPoisson."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
from opaque.accounting.base import DpProcess
from opaque.accounting.mechanisms import (
    BandMf,
    BltMf,
    DenseMf,
)
from opaque.accounting.amplification import (
    CyclicPoisson,
)


# ── BandMf dataclass tests ──────────────────────────────────────────


class TestBandMfDataclass:
    """BandMf frozen dataclass."""

    def test_fields(self):
        proc = BandMf(1.0, 100, 5)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.n_steps == 100
        assert proc.bands == 5

    def test_frozen(self):
        proc = BandMf(1.0, 100, 5)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(BandMf(1.0, 100, 5), DpProcess)

    def test_equality(self):
        assert BandMf(1.0, 100, 5) == BandMf(1.0, 100, 5)
        assert BandMf(1.0, 100, 5) != BandMf(1.0, 100, 10)

    def test_pld_returns_valid(self):
        proc = BandMf(1.0, 100, 5)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_sensitivity_is_one_for_normalized_toeplitz(self):
        """Optimized Toeplitz coefficients have L2 norm 1, so sensitivity = 1."""
        proc = BandMf(1.0, 100, 5)
        assert proc.sensitivity() == pytest.approx(1.0, abs=1e-6)

    def test_sensitivity_cached(self):
        """Sensitivity is computed once and cached."""
        proc = BandMf(1.0, 100, 5)
        s1 = proc.sensitivity()
        s2 = proc.sensitivity()
        assert s1 == s2


class TestBandMfConstructor:
    """acc.band_mf() validates and returns BandMf."""

    def test_returns_correct_type(self):
        proc = acc.band_mf(1.0, 100, 5)
        assert isinstance(proc, BandMf)

    def test_rejects_non_positive_noise(self):
        with pytest.raises(ValueError):
            acc.band_mf(0.0, 100, 5)

    def test_rejects_bad_n_steps(self):
        with pytest.raises(ValueError):
            acc.band_mf(1.0, 0, 5)

    def test_rejects_bad_bands(self):
        with pytest.raises(ValueError):
            acc.band_mf(1.0, 100, 0)
        with pytest.raises(ValueError):
            acc.band_mf(1.0, 100, 101)  # bands > n_steps


# ── CyclicPoisson tests ─────────────────────────────────────────────


class TestCyclicPoissonDataclass:
    """CyclicPoisson frozen dataclass."""

    def test_fields(self):
        inner = BandMf(1.0, 100, 5)
        proc = CyclicPoisson(inner, 0.01)
        assert proc.inner is inner
        assert proc.sample_rate == pytest.approx(0.01)

    def test_frozen(self):
        proc = CyclicPoisson(BandMf(1.0, 100, 5), 0.01)
        with pytest.raises(FrozenInstanceError):
            proc.sample_rate = 0.5  # type: ignore[misc]

    def test_is_dp_process(self):
        proc = CyclicPoisson(BandMf(1.0, 100, 5), 0.01)
        assert isinstance(proc, DpProcess)

    def test_num_groups(self):
        """num_groups = ceil(n_steps / bands)."""
        proc = CyclicPoisson(BandMf(1.0, 100, 5), 0.01)
        assert proc.num_groups == 20  # ceil(100 / 5)

        proc2 = CyclicPoisson(BandMf(1.0, 103, 5), 0.01)
        assert proc2.num_groups == 21  # ceil(103 / 5)

    def test_pld_returns_valid(self):
        proc = acc.cyclic_poisson(acc.band_mf(1.0, 100, 5), 0.01)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_matches_manual_poisson_composition(self):
        """CyclicPoisson(BandMf) should match poisson(gaussian(nm/S)) * k."""
        nm, n_steps, bands, rate = 1.0, 100, 5, 0.01
        proc = acc.cyclic_poisson(acc.band_mf(nm, n_steps, bands), rate)

        # BandMf has sensitivity ~1.0 for normalized Toeplitz
        sensitivity = acc.band_mf(nm, n_steps, bands).sensitivity()
        num_groups = math.ceil(n_steps / bands)
        manual = acc.poisson(acc.gaussian(nm / sensitivity), rate) * num_groups

        eps_proc = proc.epsilon_at(1e-5)
        eps_manual = manual.epsilon_at(1e-5)
        assert eps_proc == pytest.approx(eps_manual, rel=1e-6)

    def test_more_groups_higher_epsilon(self):
        """More groups → more composition → higher epsilon."""
        # Fewer groups (larger bands)
        eps_small = acc.cyclic_poisson(
            acc.band_mf(1.0, 100, 50), 0.01
        ).epsilon_at(1e-5)
        # More groups (smaller bands)
        eps_large = acc.cyclic_poisson(
            acc.band_mf(1.0, 100, 5), 0.01
        ).epsilon_at(1e-5)
        assert eps_small < eps_large

    def test_transparent_to_inner(self):
        """CyclicPoisson accesses BandMf's internal parameters."""
        inner = acc.band_mf(1.0, 100, 5)
        proc = acc.cyclic_poisson(inner, 0.01)
        assert proc.inner.bands == 5
        assert proc.inner.n_steps == 100
        assert proc.inner.noise_multiplier == 1.0


class TestCyclicPoissonConstructor:
    """acc.cyclic_poisson() validates and returns CyclicPoisson."""

    def test_returns_correct_type(self):
        proc = acc.cyclic_poisson(acc.band_mf(1.0, 100, 5), 0.01)
        assert isinstance(proc, CyclicPoisson)

    def test_rejects_non_band_mf(self):
        """cyclic_poisson only accepts BandMf."""
        with pytest.raises(TypeError, match="BandMf"):
            acc.cyclic_poisson(acc.gaussian(1.0), 0.01)  # type: ignore[arg-type]

    def test_rejects_bad_sample_rate(self):
        with pytest.raises(ValueError):
            acc.cyclic_poisson(acc.band_mf(1.0, 100, 5), 0.0)
        with pytest.raises(ValueError):
            acc.cyclic_poisson(acc.band_mf(1.0, 100, 5), 1.5)


# ── BltMf tests ─────────────────────────────────────────────────────


class TestBltMfDataclass:
    """BltMf frozen dataclass."""

    def test_fields(self):
        proc = BltMf(1.0, 50, 1, 1, "max", 5)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.n_steps == 50
        assert proc.min_sep == 1
        assert proc.max_participations == 1
        assert proc.error == "max"
        assert proc.max_buffers == 5

    def test_frozen(self):
        proc = BltMf(1.0, 50, 1, 1, "max", 5)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(BltMf(1.0, 50, 1, 1, "max", 5), DpProcess)

    def test_pld_returns_valid(self):
        proc = acc.blt_mf(1.0, 50, max_buffers=3)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_sensitivity_positive(self):
        proc = acc.blt_mf(1.0, 50, max_buffers=3)
        assert proc.sensitivity() > 0


class TestBltMfConstructor:
    """acc.blt_mf() validates and returns BltMf."""

    def test_returns_correct_type(self):
        proc = acc.blt_mf(1.0, 50)
        assert isinstance(proc, BltMf)

    def test_rejects_non_positive_noise(self):
        with pytest.raises(ValueError):
            acc.blt_mf(0.0, 50)

    def test_rejects_bad_n_steps(self):
        with pytest.raises(ValueError):
            acc.blt_mf(1.0, 0)

    def test_rejects_bad_min_sep(self):
        with pytest.raises(ValueError):
            acc.blt_mf(1.0, 50, min_sep=0)

    def test_rejects_bad_error(self):
        with pytest.raises(ValueError):
            acc.blt_mf(1.0, 50, error="invalid")


# ── DenseMf tests ───────────────────────────────────────────────────


class TestDenseMfDataclass:
    """DenseMf frozen dataclass."""

    def test_fields(self):
        proc = DenseMf(1.0, 20, 1, None, False)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.n_steps == 20
        assert proc.epochs == 1
        assert proc.bands is None
        assert proc.equal_norm is False

    def test_frozen(self):
        proc = DenseMf(1.0, 20, 1, None, False)
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(DenseMf(1.0, 20, 1, None, False), DpProcess)

    def test_pld_returns_valid(self):
        proc = acc.dense_mf(1.0, 20)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_sensitivity_positive(self):
        proc = acc.dense_mf(1.0, 20)
        assert proc.sensitivity() > 0

    def test_multi_epoch(self):
        """Multi-epoch dense MF computes successfully."""
        proc = acc.dense_mf(1.0, 20, epochs=2)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


class TestDenseMfConstructor:
    """acc.dense_mf() validates and returns DenseMf."""

    def test_returns_correct_type(self):
        proc = acc.dense_mf(1.0, 20)
        assert isinstance(proc, DenseMf)

    def test_rejects_non_positive_noise(self):
        with pytest.raises(ValueError):
            acc.dense_mf(0.0, 20)

    def test_rejects_bad_n_steps(self):
        with pytest.raises(ValueError):
            acc.dense_mf(1.0, 0)

    def test_rejects_bad_epochs(self):
        with pytest.raises(ValueError):
            acc.dense_mf(1.0, 20, epochs=0)

    def test_rejects_non_dividing_epochs(self):
        with pytest.raises(ValueError):
            acc.dense_mf(1.0, 20, epochs=3)  # 3 doesn't divide 20


# ── Composition tests ───────────────────────────────────────────────


class TestMfComposition:
    """MF mechanisms compose with other DpProcess nodes."""

    def test_band_mf_composes_with_gaussian(self):
        """BandMf | Gaussian works."""
        proc = acc.band_mf(1.0, 100, 5) | acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_cyclic_poisson_composes_with_gaussian(self):
        """CyclicPoisson | Gaussian works."""
        proc = (
            acc.cyclic_poisson(acc.band_mf(1.0, 100, 5), 0.01)
            | acc.gaussian(1.0)
        )
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_class_tree_visible(self):
        """The composition tree preserves class types for debugging."""
        inner = acc.band_mf(1.0, 100, 5)
        proc = acc.cyclic_poisson(inner, 0.01)

        assert isinstance(proc, CyclicPoisson)
        assert isinstance(proc.inner, BandMf)
        assert proc.inner.bands == 5
