"""Tight accounting for ``second_moment(gaussian)`` + Poisson family.

When :func:`second_moment` wraps a :class:`Gaussian` (DP-SGD baseline),
the resulting joint mechanism is itself a Gaussian with effective noise
multiplier ``σ ÷ joint_sensitivity``.  Each Poisson-family amplification
(Poisson, TruncatedPoisson, ParallelPoisson, BallsInBins) therefore reduces
to amplification of an ordinary Gaussian at the effective noise multiplier
— the tight bound, not a conservative shortcut.

These tests pin the equivalence:

    poisson(second_moment(gaussian(σ), sensitivity=Δ), q)
        ≡ poisson(gaussian(σ/(Δ·√(3/2))), q)
"""

from __future__ import annotations

import math

import pytest

import opaque.accounting as acc
from opaque.accounting.amplification.types import (
    BallsInBins,
    ParallelPoisson,
    Poisson,
    TruncatedPoisson,
)
from opaque.accounting.transformations.types import SecondMoment


_OVERHEAD = math.sqrt(3.0 / 2.0)


def _effective_nm(noise_multiplier: float, sensitivity: float) -> float:
    """Effective noise multiplier for ``second_moment(gaussian(...), sensitivity=...)``.

    For a Gaussian inner with identity strategy (c1=1) and the default
    ``sqrt(3/2)`` overhead.
    """
    return noise_multiplier / (sensitivity * _OVERHEAD)


# ── Construction & dispatch ──────────────────────────────────────────


class TestPoissonAcceptsSecondMoment:
    def test_constructs(self):
        sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
        p = acc.poisson(sm, sample_rate=0.01)
        assert isinstance(p, Poisson)
        assert isinstance(p.inner, SecondMoment)

    def test_pld_returns_valid(self):
        sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
        eps = acc.poisson(sm, sample_rate=0.01).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_zero_noise_multiplier_is_non_private(self):
        sm = acc.second_moment(acc.gaussian(0.0), sensitivity=1.0)
        eps = acc.poisson(sm, sample_rate=0.01).epsilon_at(1e-5)
        assert math.isinf(eps) or eps > 1e10  # non-private PLD

    def test_rejects_mf_inner(self):
        # SecondMoment(MfGaussian) must redirect to cyclic_poisson / b_min_sep.
        sm = acc.second_moment(
            acc.band_mf(0.8, sensitivity=1.0, num_groups=10),
            sensitivity=1.0,
        )
        with pytest.raises(TypeError, match="MfGaussian|cyclic_poisson|b_min_sep"):
            acc.poisson(sm, sample_rate=0.01)


class TestTruncatedPoissonAcceptsSecondMoment:
    def test_constructs(self):
        sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
        t = acc.truncated_poisson(sm, 0.01, 128, 10_000)
        assert isinstance(t, TruncatedPoisson)

    def test_pld_returns_valid(self):
        sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
        eps = acc.truncated_poisson(sm, 0.01, 128, 10_000).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_rejects_mf_inner(self):
        sm = acc.second_moment(
            acc.band_mf(0.8, sensitivity=1.0, num_groups=10),
            sensitivity=1.0,
        )
        with pytest.raises(TypeError, match="MfGaussian|cyclic_poisson|b_min_sep"):
            acc.truncated_poisson(sm, 0.01, 128, 10_000)


class TestParallelPoissonAcceptsSecondMoment:
    def test_constructs(self):
        sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
        a = acc.parallel_poisson(sm, sample_rate=0.01, num_workers=4)
        assert isinstance(a, ParallelPoisson)

    def test_pld_returns_valid(self):
        sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
        eps = acc.parallel_poisson(sm, sample_rate=0.01, num_workers=4).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_rejects_mf_inner(self):
        sm = acc.second_moment(
            acc.band_mf(0.8, sensitivity=1.0, num_groups=10),
            sensitivity=1.0,
        )
        with pytest.raises(TypeError, match="MfGaussian|cyclic_poisson|b_min_sep"):
            acc.parallel_poisson(sm, sample_rate=0.01, num_workers=4)


class TestBallsInBinsAcceptsSecondMoment:
    def test_constructs(self):
        sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
        b = acc.balls_in_bins(sm, num_bins=100, num_epochs=10)
        assert isinstance(b, BallsInBins)

    def test_pld_returns_valid(self):
        sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
        eps = acc.balls_in_bins(sm, num_bins=100, num_epochs=10).epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ── Math equivalence: SM(G(σ), Δ) ≡ G(σ/(Δ·√(3/2))) under amplification ──


@pytest.mark.parametrize("sigma", [0.5, 0.8, 1.5])
@pytest.mark.parametrize("sensitivity", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("rate", [0.001, 0.01, 0.05])
def test_poisson_second_moment_matches_effective_gaussian(sigma, sensitivity, rate):
    """``poisson(SM(G(σ), Δ), q)`` must equal ``poisson(G(effective_nm), q)``."""
    sm = acc.second_moment(acc.gaussian(sigma), sensitivity=sensitivity)
    p_sm = acc.poisson(sm, sample_rate=rate)
    p_g = acc.poisson(acc.gaussian(_effective_nm(sigma, sensitivity)), sample_rate=rate)
    assert p_sm.epsilon_at(1e-5) == pytest.approx(p_g.epsilon_at(1e-5), rel=1e-6)


@pytest.mark.parametrize("sigma", [0.8, 1.5])
@pytest.mark.parametrize("sensitivity", [1.0, 2.0])
def test_truncated_poisson_second_moment_matches_effective_gaussian(sigma, sensitivity):
    sm = acc.second_moment(acc.gaussian(sigma), sensitivity=sensitivity)
    t_sm = acc.truncated_poisson(sm, 0.01, 128, 10_000)
    t_g = acc.truncated_poisson(
        acc.gaussian(_effective_nm(sigma, sensitivity)),
        0.01,
        128,
        10_000,
    )
    assert t_sm.epsilon_at(1e-5) == pytest.approx(t_g.epsilon_at(1e-5), rel=1e-6)


@pytest.mark.parametrize("sigma", [0.8, 1.5])
@pytest.mark.parametrize("sensitivity", [1.0, 2.0])
def test_parallel_poisson_second_moment_matches_effective_gaussian(sigma, sensitivity):
    sm = acc.second_moment(acc.gaussian(sigma), sensitivity=sensitivity)
    a_sm = acc.parallel_poisson(sm, sample_rate=0.01, num_workers=4)
    a_g = acc.parallel_poisson(
        acc.gaussian(_effective_nm(sigma, sensitivity)),
        sample_rate=0.01,
        num_workers=4,
    )
    assert a_sm.epsilon_at(1e-5) == pytest.approx(a_g.epsilon_at(1e-5), rel=1e-6)


@pytest.mark.parametrize("sigma", [0.8, 1.5])
@pytest.mark.parametrize("sensitivity", [1.0, 2.0])
def test_balls_in_bins_second_moment_matches_effective_gaussian(sigma, sensitivity):
    sm = acc.second_moment(acc.gaussian(sigma), sensitivity=sensitivity)
    b_sm = acc.balls_in_bins(sm, num_bins=100, num_epochs=10)
    b_g = acc.balls_in_bins(
        acc.gaussian(_effective_nm(sigma, sensitivity)),
        num_bins=100,
        num_epochs=10,
    )
    assert b_sm.epsilon_at(1e-5) == pytest.approx(b_g.epsilon_at(1e-5), rel=1e-6)


# ── Composition with repeat / *N ─────────────────────────────────────


def test_poisson_second_moment_composes_over_steps():
    sm = acc.second_moment(acc.gaussian(0.8), sensitivity=1.0)
    step = acc.poisson(sm, sample_rate=0.01)
    training = step * 1000
    eps = training.epsilon_at(1e-5)
    assert math.isfinite(eps) and eps > 0


# ── SecondMoment over AdaClip (adaptive clipping + private 2nd moments) ─

# When per-step adaptive clipping releases its quantile estimate alongside
# the joint first+second moment streams, the right composition is
# ``second_moment(adaclip(gaussian))``: AdaClip's z_eff folds the
# threshold-quantile noise into the gradient noise as a single effective
# Gaussian, and SecondMoment's joint-sensitivity overhead is then applied.
# The two releases use independent randomness, so the composition is exact.


class TestSecondMomentOverAdaClip:
    """``second_moment(adaclip(gaussian))`` is a valid Gaussian-family inner."""

    def test_constructs(self):
        from opaque.accounting.mechanisms.types import Gaussian as GaussianT
        from opaque.accounting.transformations.types import AdaClip as AdaClipT

        sm = acc.second_moment(
            acc.adaclip(acc.gaussian(1.1), expected_batch_size=128),
            sensitivity=1.0,
        )
        assert isinstance(sm.inner, AdaClipT)
        assert isinstance(sm.inner.inner, GaussianT)

    def test_pld_returns_valid(self):
        sm = acc.second_moment(
            acc.adaclip(acc.gaussian(1.1), expected_batch_size=128),
            sensitivity=1.0,
        )
        eps = sm.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_uses_adaclip_effective_nm(self):
        """SecondMoment over AdaClip uses AdaClip's z_eff, not the raw inner σ.

        For AdaClip(Gaussian(σ)), ``effective_noise_multiplier`` < σ (the
        threshold-quantile cost reduces the effective noise budget).  The
        SecondMoment-wrapped PLD must reflect that smaller effective σ.
        """
        ac = acc.adaclip(acc.gaussian(1.1), expected_batch_size=128)
        # Sanity: AdaClip's z_eff is strictly less than the raw inner σ.
        assert ac.effective_noise_multiplier < 1.1

        sm_over_adaclip = acc.second_moment(ac, sensitivity=1.0)
        # The equivalent direct-Gaussian construction at the AdaClip-folded σ.
        g_eff = acc.second_moment(
            acc.gaussian(ac.effective_noise_multiplier), sensitivity=1.0,
        )
        assert sm_over_adaclip.epsilon_at(1e-5) == pytest.approx(
            g_eff.epsilon_at(1e-5), rel=1e-6,
        )

    def test_rejects_non_gaussian_adaclip_inner(self):
        from opaque.accounting.transformations.types import AdaClip
        from opaque.accounting.mechanisms.types import NonPrivate

        # Build a non-Gaussian AdaClip directly (the factory accepts NonPrivate).
        ac_np = AdaClip(
            inner=NonPrivate(),
            fraction_noise_std=0.05,
            expected_batch_size=128.0,
        )
        with pytest.raises(TypeError, match="Gaussian inside the AdaClip"):
            acc.second_moment(ac_np, sensitivity=1.0)


@pytest.mark.parametrize(
    "amplifier", ["poisson", "truncated_poisson", "balls_in_bins"]
)
def test_amplification_accepts_second_moment_over_adaclip(amplifier):
    """Each Poisson-family amplification accepts ``second_moment(adaclip(gaussian))``."""
    sm = acc.second_moment(
        acc.adaclip(acc.gaussian(1.1), expected_batch_size=128),
        sensitivity=1.0,
    )
    if amplifier == "poisson":
        proc = acc.poisson(sm, sample_rate=0.01)
    elif amplifier == "truncated_poisson":
        proc = acc.truncated_poisson(sm, 0.01, 128, 10_000)
    else:
        proc = acc.balls_in_bins(sm, num_bins=100, num_epochs=10)
    eps = proc.epsilon_at(1e-5)
    assert math.isfinite(eps) and eps > 0


def test_parallel_poisson_accepts_second_moment_over_adaclip():
    """Parallel Poisson dispatches through ``Poisson(SecondMoment(AdaClip(Gaussian)))``."""
    sm = acc.second_moment(
        acc.adaclip(acc.gaussian(1.1), expected_batch_size=128),
        sensitivity=1.0,
    )
    proc = acc.parallel_poisson(sm, sample_rate=0.01, num_workers=4)
    eps = proc.epsilon_at(1e-5)
    assert math.isfinite(eps) and eps > 0
