"""Tests for :class:`~opaque.dpftrl.accounting.types.IdentityMf` and the FTRL
amplifications dispatching on it (``cyclic_poisson``, ``balls_in_bins``)."""

import math

import pytest

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.accounting import _native
from opaque.accounting.discretization import get_discretization
from opaque.dpftrl.accounting.types import IdentityMf


_DELTA = 1e-5


# ---------------------------------------------------------------------------
# Mechanism: opaque.dpftrl.accounting.mf_identity
# ---------------------------------------------------------------------------


class TestMfIdentityMechanism:
    def test_factory_returns_identity_mf(self):
        proc = ftrl_acc.mf_identity(1.0)
        assert isinstance(proc, IdentityMf)
        assert proc.noise_multiplier == 1.0

    def test_pld_matches_unsubsampled_gaussian(self):
        nm = 1.5
        proc = ftrl_acc.mf_identity(nm)

        cfg = get_discretization()
        ref = _native.gaussian_pld(nm, cfg.to_native())
        assert math.isclose(
            proc.pld().epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_zero_noise_is_non_private(self):
        assert math.isinf(ftrl_acc.mf_identity(0.0).epsilon_at(_DELTA))

    def test_self_compose_matches_repeated_gaussian(self):
        nm = 2.0
        T = 50
        proc = ftrl_acc.mf_identity(nm) * T

        cfg = get_discretization()
        ref = _native.gaussian_pld(nm, cfg.to_native()).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )


# ---------------------------------------------------------------------------
# cyclic_poisson(IdentityMf(...), sample_rate, num_steps)
# ---------------------------------------------------------------------------


class TestCyclicPoissonIdentity:
    def test_pld_matches_self_composed_poisson_gaussian(self):
        nm, p, T = 1.1, 0.01, 500
        proc = ftrl_acc.cyclic_poisson(
            ftrl_acc.mf_identity(nm), sample_rate=p, num_steps=T
        )
        cfg = get_discretization()
        ref = _native.poisson_gaussian_pld(nm, p, cfg.to_native()).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_requires_num_steps(self):
        with pytest.raises(ValueError, match="num_steps"):
            ftrl_acc.cyclic_poisson(ftrl_acc.mf_identity(1.0), sample_rate=0.1)

    def test_rejects_invalid_num_steps(self):
        with pytest.raises(ValueError, match="num_steps"):
            ftrl_acc.cyclic_poisson(
                ftrl_acc.mf_identity(1.0), sample_rate=0.1, num_steps=0
            )

    def test_rejects_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ftrl_acc.cyclic_poisson(
                ftrl_acc.mf_identity(1.0), sample_rate=1.5, num_steps=10
            )

    def test_band_mf_unaffected_by_num_steps_passthrough(self):
        # Existing BandMF behaviour: num_steps None reads from inner.num_groups.
        T = 100
        proc = ftrl_acc.cyclic_poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=T), sample_rate=0.01
        )
        explicit = ftrl_acc.cyclic_poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=T),
            sample_rate=0.01,
            num_steps=T,
        )
        assert math.isclose(
            proc.epsilon_at(_DELTA), explicit.epsilon_at(_DELTA), rel_tol=1e-12
        )

    def test_band_mf_num_steps_mismatch_raises(self):
        with pytest.raises(ValueError, match="num_groups"):
            ftrl_acc.cyclic_poisson(
                ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=100),
                sample_rate=0.01,
                num_steps=200,
            )


# ---------------------------------------------------------------------------
# balls_in_bins(IdentityMf(...), num_bins, num_epochs)  — tight reduction
# ---------------------------------------------------------------------------


class TestBallsInBinsIdentity:
    def test_pld_matches_per_step_poisson_reduction(self):
        nm, k, E = 1.5, 32, 4
        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(nm), num_bins=k, num_epochs=E
        )
        cfg = get_discretization()
        ref = _native.poisson_gaussian_pld(
            nm, 1.0 / k, cfg.to_native()
        ).self_compose(k * E)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_zero_noise_non_private(self):
        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(0.0), num_bins=10, num_epochs=2
        )
        assert math.isinf(proc.epsilon_at(_DELTA))

    def test_rejects_invalid_num_bins(self):
        with pytest.raises(ValueError, match="num_bins"):
            ftrl_acc.balls_in_bins(
                ftrl_acc.mf_identity(1.0), num_bins=1, num_epochs=2
            )

    def test_rejects_invalid_num_epochs(self):
        with pytest.raises(ValueError, match="num_epochs"):
            ftrl_acc.balls_in_bins(
                ftrl_acc.mf_identity(1.0), num_bins=10, num_epochs=0
            )


# ---------------------------------------------------------------------------
# Calibration smoke
# ---------------------------------------------------------------------------


def test_mf_identity_calibrates_through_cyclic_poisson():
    cal = acc.calibrate(
        acc.epsilon_budget(3.0, delta=_DELTA),
        lambda nm: ftrl_acc.cyclic_poisson(
            ftrl_acc.mf_identity(nm), sample_rate=0.01, num_steps=500
        ),
        param_min=0.1,
        param_max=10.0,
    )
    assert cal.param > 0
    assert cal.achieved <= 3.0 + 1e-6
