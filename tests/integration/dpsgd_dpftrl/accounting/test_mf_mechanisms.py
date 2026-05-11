"""Tests for MF accounting — single ``MfGaussian(nm, strategy)`` mechanism.

The MF accounting layer collapses to one DpProcess class, ``MfGaussian``,
parameterised by ``noise_multiplier`` and a strategy from
``opaque.dpftrl.noise``.  These tests cover the dataclass surface,
amplification dispatch on ``type(strategy)``, and composition with
other DpProcess nodes.
"""

import math
from dataclasses import FrozenInstanceError

import pytest

import opaque.dpsgd.accounting as dpsgd_acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core._base import DpProcess
from opaque.dpftrl.accounting.amplification.types import CyclicPoisson
from opaque.dpftrl.accounting.types import MfGaussian
from opaque.dpftrl.noise import blt_strategy, identity_strategy
from opaque.dpftrl.noise.types import BandMfStrategy


# ── MfGaussian(BandMfStrategy) — banded-Toeplitz path ────────────────────


def _band(nm: float = 1.0, sens: float = 1.0, coefs=(0.9, 0.1)) -> MfGaussian:
    return ftrl_acc.mf_gaussian(
        nm, BandMfStrategy(sensitivity=sens, coefficients=coefs)
    )


class TestBandMfGaussian:
    """``MfGaussian`` wrapping a ``BandMfStrategy`` behaves like a frozen DP process."""

    def test_fields_via_strategy(self):
        proc = _band(1.0, 1.0, (0.9, 0.1))
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity == pytest.approx(1.0)
        assert proc.strategy.coefficients == (0.9, 0.1)
        assert proc.strategy.bands == 2

    def test_frozen(self):
        proc = _band()
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(_band(), DpProcess)

    def test_equality(self):
        assert _band(1.0, 1.0, (0.9, 0.1)) == _band(1.0, 1.0, (0.9, 0.1))
        assert _band(1.0, 1.0, (0.9, 0.1)) != _band(1.0, 1.0, (0.5, 0.5))

    @pytest.mark.slow
    def test_pld_returns_valid(self):
        proc = _band(1.0, 1.0, (1.0,))
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ── CyclicPoisson(MfGaussian(BandMfStrategy)) ────────────────────────────


class TestFtrlPoissonDataclass:
    """``CyclicPoisson`` over ``MfGaussian`` frozen dataclass."""

    def test_fields(self):
        inner = _band(1.0, 1.0, (0.9, 0.1))
        proc = CyclicPoisson(inner=inner, sample_rate=0.01, n_steps=100)
        assert proc.inner is inner
        assert proc.sample_rate == pytest.approx(0.01)
        assert proc.n_steps == 100

    def test_frozen(self):
        proc = CyclicPoisson(inner=_band(1.0, 1.0, (1.0,)), sample_rate=0.01, n_steps=20)
        with pytest.raises(FrozenInstanceError):
            proc.sample_rate = 0.5  # type: ignore[misc]

    def test_is_dp_process(self):
        proc = CyclicPoisson(inner=_band(1.0, 1.0, (1.0,)), sample_rate=0.01, n_steps=20)
        assert isinstance(proc, DpProcess)

    def test_pld_returns_valid(self):
        proc = ftrl_acc.poisson(_band(1.0, 1.0, (1.0,)), sample_rate=0.01, n_steps=20)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_matches_manual_poisson_composition(self):
        """poisson(BandMf(bands=1)) should match poisson(gaussian(nm/S)) * n_steps."""
        nm, sensitivity, n_steps, rate = 1.0, 1.0, 20, 0.01
        proc = ftrl_acc.poisson(
            _band(nm, sensitivity, (1.0,)),
            sample_rate=rate,
            n_steps=n_steps,
        )

        manual = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm / sensitivity), rate) * n_steps

        assert proc.epsilon_at(1e-5) == pytest.approx(manual.epsilon_at(1e-5), rel=1e-6)

    def test_more_steps_higher_epsilon(self):
        eps_small = ftrl_acc.poisson(
            _band(1.0, 1.0, (1.0,)), sample_rate=0.01, n_steps=2
        ).epsilon_at(1e-5)
        eps_large = ftrl_acc.poisson(
            _band(1.0, 1.0, (1.0,)), sample_rate=0.01, n_steps=20
        ).epsilon_at(1e-5)
        assert eps_small < eps_large


class TestFtrlPoissonConstructor:
    """ftrl_acc.poisson() validates and returns CyclicPoisson."""

    def test_returns_correct_type(self):
        proc = ftrl_acc.poisson(
            _band(1.0, 1.0, (1.0,)), sample_rate=0.01, n_steps=20
        )
        assert isinstance(proc, CyclicPoisson)

    def test_rejects_unsupported_inner(self):
        with pytest.raises(TypeError):
            ftrl_acc.poisson(dpsgd_acc.gaussian(1.0), 0.01, n_steps=20)  # type: ignore[arg-type]

    def test_rejects_bad_sample_rate(self):
        inner = _band(1.0, 1.0, (1.0,))
        with pytest.raises(ValueError):
            ftrl_acc.poisson(inner, 0.0, n_steps=10)
        with pytest.raises(ValueError):
            ftrl_acc.poisson(inner, 1.5, n_steps=10)


# ── MfGaussian(BltStrategy) — built via blt_strategy ──────────────────


class TestBltMfGaussian:
    """``MfGaussian`` wrapping a ``BltStrategy``."""

    def _blt(self, noise_multiplier: float = 1.0):
        s = blt_strategy(n_steps=10, min_sep=10, max_participations=1, momentum=1.0)
        return ftrl_acc.mf_gaussian(noise_multiplier, s)

    def test_fields(self):
        proc = self._blt(1.0)
        assert proc.noise_multiplier == pytest.approx(1.0)
        assert proc.sensitivity > 0
        assert isinstance(proc.strategy.gram_matrix, tuple)
        assert isinstance(proc.strategy.coefficients, tuple)

    def test_frozen(self):
        proc = self._blt()
        with pytest.raises(FrozenInstanceError):
            proc.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(self._blt(), DpProcess)

    def test_pld_returns_valid(self):
        eps = self._blt().epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ── Composition tests ───────────────────────────────────────────────


class TestMfComposition:
    """MfGaussian mechanisms compose with other DpProcess nodes."""

    def test_band_mf_composes_with_gaussian(self):
        proc = _band(1.0, 1.0, (1.0,)) | dpsgd_acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_poisson_composes_with_gaussian(self):
        inner = _band(1.0, 1.0, (1.0,))
        proc = ftrl_acc.poisson(inner, 0.01, n_steps=20) | dpsgd_acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_identity_strategy_composes_with_gaussian(self):
        proc = ftrl_acc.mf_gaussian(1.0, identity_strategy()) | dpsgd_acc.gaussian(1.0)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0
