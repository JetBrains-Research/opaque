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
    def test_pld_agrees_with_generic_bnb_mc(self):
        """Identity-specialised MC must agree with generic ``bnb_mc_pld`` at
        ``G = E·I_b`` up to MC noise.  At the high sample budget set here, the
        per-seed relative MC σ for both methods is < 2% and the gap between
        means is small — empirically <5% at a single seed × 1M samples
        (verified across many seeds at this config).  Bound at 10% to leave
        room for rare outliers without becoming a privacy-blind tolerance."""
        nm, k, E = 1.5, 32, 4
        # Use a higher sample budget than the package default for a tight check.
        from opaque.accounting.discretization import DiscretizationConfig

        tight_cfg = DiscretizationConfig(num_mc_samples=1_000_000, seed=2024)
        cfg_native = tight_cfg.to_native()

        ref_eps = _native.bnb_mc_pld(
            [E if i == j else 0.0 for i in range(k) for j in range(k)],
            k,
            nm,
            cfg_native,
        ).epsilon_at(_DELTA)
        # Identity-specialised path with τ = 0 (no IS) should match generic
        # at any tilt the user picks; the IS specialisation only changes
        # variance, not the mean.
        eps_id_no_is = _native.bnb_mc_pld_identity(
            k, E, nm, 0.0, cfg_native
        ).epsilon_at(_DELTA)
        assert abs(eps_id_no_is - ref_eps) < 0.10 * abs(ref_eps), (
            f"identity τ=0 vs generic gap too large: id={eps_id_no_is}, "
            f"ref={ref_eps}"
        )

    def test_default_tilt_path_finite(self):
        """Default importance_tilt=1.0 path produces a finite, positive ε."""
        nm, k, E = 1.5, 32, 4
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(nm),
            num_bins=k,
            n_steps=k * E,
        ).epsilon_at(_DELTA)
        assert math.isfinite(eps) and eps > 0

    def test_zero_tilt_path_finite(self):
        """importance_tilt=0 path produces a finite, positive ε."""
        nm, k, E = 1.5, 32, 4
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(nm),
            num_bins=k,
            n_steps=k * E,
            importance_tilt=0.0,
        ).epsilon_at(_DELTA)
        assert math.isfinite(eps) and eps > 0

    @pytest.mark.slow
    def test_is_reduces_variance_on_ensemble(self):
        """``importance_tilt=1.0`` reduces ε std vs ``τ=0`` over a large enough
        seed ensemble.  This is the *empirical* point of the IS specialisation;
        small-N seed ensembles can be unlucky (std varies as χ² with `df = N-1`),
        so we use 32 seeds and require ≥ 1.4× std reduction (squared variance
        ≥ 2×).  In practice IS gives 3-30× std reduction at this config.
        """
        import statistics
        from opaque.accounting.discretization import DiscretizationConfig

        nm, k, E = 1.5, 32, 4
        budget = 200_000
        seeds = list(range(32))
        eps_no_is, eps_is = [], []
        for s in seeds:
            cfg_native = DiscretizationConfig(
                num_mc_samples=budget, seed=s
            ).to_native()
            eps_no_is.append(
                _native.bnb_mc_pld_identity(k, E, nm, 0.0, cfg_native).epsilon_at(
                    _DELTA
                )
            )
            eps_is.append(
                _native.bnb_mc_pld_identity(k, E, nm, 1.0, cfg_native).epsilon_at(
                    _DELTA
                )
            )
        std_no_is = statistics.stdev(eps_no_is)
        std_is = statistics.stdev(eps_is)
        assert std_is < std_no_is / 1.4, (
            f"IS should reduce per-seed std vs τ=0 over 32 seeds: "
            f"std(τ=0)={std_no_is:.4f}, std(τ=1)={std_is:.4f}"
        )

    def test_strictly_tighter_than_unamplified_composition(self):
        """Amplification (factor ~1/num_bins) must beat unamplified composition."""
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
