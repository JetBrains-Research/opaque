"""Tests for :class:`~opaque.dpftrl.accounting.types.IdentityMf` and the FTRL
amplifications dispatching on it (``poisson``, ``balls_in_bins``)."""

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

    def test_negative_noise_multiplier_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ftrl_acc.mf_identity(-0.1)

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
# poisson(IdentityMf(...), sample_rate, n_steps)
# ---------------------------------------------------------------------------


class TestPoissonIdentity:
    def test_pld_matches_self_composed_poisson_gaussian(self):
        nm, p, T = 1.1, 0.01, 500
        proc = ftrl_acc.poisson(ftrl_acc.mf_identity(nm), sample_rate=p, n_steps=T)
        cfg = get_discretization()
        ref = _native.poisson_gaussian_pld(nm, p, cfg.to_native()).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_requires_n_steps(self):
        with pytest.raises(TypeError):
            ftrl_acc.poisson(ftrl_acc.mf_identity(1.0), sample_rate=0.1)

    def test_rejects_invalid_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            ftrl_acc.poisson(ftrl_acc.mf_identity(1.0), sample_rate=0.1, n_steps=0)

    def test_rejects_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ftrl_acc.poisson(ftrl_acc.mf_identity(1.0), sample_rate=1.5, n_steps=10)


class TestPoissonBandMf:
    def test_pld_matches_self_composed_with_bands(self):
        """For BandMf: num_groups = ceil(n_steps / bands)."""
        nm, p = 1.1, 0.01
        coefs = (1.0, 0.5)  # bands = 2
        bands = len(coefs)
        n_steps = 100
        proc = ftrl_acc.poisson(
            ftrl_acc.band_mf(nm, sensitivity=1.0, coefficients=coefs),
            sample_rate=p,
            n_steps=n_steps,
        )
        cfg = get_discretization()
        num_groups = math.ceil(n_steps / bands)
        ref = _native.poisson_gaussian_pld(nm, p, cfg.to_native()).self_compose(
            num_groups
        )
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )


# ---------------------------------------------------------------------------
# balls_in_bins(IdentityMf(...), num_bins, n_steps)  — tight reduction
# ---------------------------------------------------------------------------


class TestBallsInBinsIdentity:
    def test_pld_matches_lemma_3_2_dominating_pair(self):
        """For identity C=I, Lemma 3.2 of CC2024 gives Gram = E * I_b."""
        nm, k, E = 1.5, 32, 4
        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(nm), num_bins=k, n_steps=k * E
        )
        cfg = get_discretization()
        gram = [E if i == j else 0.0 for i in range(k) for j in range(k)]
        ref = _native.bnb_mc_pld(gram, k, nm, cfg.to_native())
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_strictly_tighter_than_unamplified_composition(self):
        """Lemma 3.2 amplification (factor ~1/num_bins) must beat the unamplified
        Gaussian composition over all n_steps rounds."""
        nm, k, E = 1.5, 32, 4
        amplified = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(nm), num_bins=k, n_steps=k * E
        ).epsilon_at(_DELTA)
        cfg = get_discretization()
        unamplified = (
            _native.gaussian_pld(nm, cfg.to_native())
            .self_compose(k * E)
            .epsilon_at(_DELTA)
        )
        assert amplified < unamplified

    def test_zero_noise_non_private(self):
        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(0.0), num_bins=10, n_steps=20
        )
        assert math.isinf(proc.epsilon_at(_DELTA))

    def test_rejects_invalid_num_bins(self):
        with pytest.raises(ValueError, match="num_bins"):
            ftrl_acc.balls_in_bins(ftrl_acc.mf_identity(1.0), num_bins=1, n_steps=20)

    def test_rejects_invalid_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            ftrl_acc.balls_in_bins(ftrl_acc.mf_identity(1.0), num_bins=10, n_steps=0)

    def test_rejects_n_steps_not_multiple_of_num_bins(self):
        with pytest.raises(ValueError, match="multiple of"):
            ftrl_acc.balls_in_bins(ftrl_acc.mf_identity(1.0), num_bins=10, n_steps=15)


# ---------------------------------------------------------------------------
# Calibration smoke
# ---------------------------------------------------------------------------


def test_mf_identity_calibrates_through_poisson():
    cal = acc.calibrate(
        acc.epsilon_budget(3.0, delta=_DELTA),
        lambda nm: ftrl_acc.poisson(
            ftrl_acc.mf_identity(nm), sample_rate=0.01, n_steps=500
        ),
        param_min=0.1,
        param_max=10.0,
    )
    assert cal.param > 0
    assert cal.achieved <= 3.0 + 1e-6
