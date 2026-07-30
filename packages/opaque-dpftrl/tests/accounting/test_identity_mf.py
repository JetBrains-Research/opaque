"""Tests for ``MfGaussian(IdentityStrategy)`` and the FTRL amplifications
dispatching on it (``poisson``, ``balls_in_bins``)."""

import math

import pytest

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core import _native
from opaque.api.accounting.core.discretization import get_discretization
from opaque.dpftrl.accounting.types import MfGaussian
from opaque.dpftrl.noise import band_mf_strategy, identity_strategy
from opaque.dpftrl.noise.types import IdentityStrategy

_DELTA = 1e-5


# ---------------------------------------------------------------------------
# Mechanism: opaque.dpftrl.accounting.identity_mf
# ---------------------------------------------------------------------------


class TestIdentityMfMechanism:
    def test_factory_returns_mf_gaussian_with_identity_strategy(self):
        proc = ftrl_acc.mf_gaussian(1.0, identity_strategy())
        assert isinstance(proc, MfGaussian)
        assert isinstance(proc.strategy, IdentityStrategy)
        assert proc.noise_multiplier == 1.0

    def test_pld_matches_unsubsampled_gaussian(self):
        nm = 1.5
        proc = ftrl_acc.mf_gaussian(nm, identity_strategy())

        cfg = get_discretization()
        ref = _native.gaussian_pld(nm, cfg.to_native())
        assert math.isclose(
            proc.pld().epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_zero_noise_is_non_private(self):
        assert math.isinf(
            ftrl_acc.mf_gaussian(0.0, identity_strategy()).epsilon_at(_DELTA)
        )

    def test_negative_noise_multiplier_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ftrl_acc.mf_gaussian(-0.1, identity_strategy())

    def test_self_compose_matches_repeated_gaussian(self):
        nm = 2.0
        T = 50
        proc = ftrl_acc.mf_gaussian(nm, identity_strategy()) * T

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
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()), sample_rate=p, n_steps=T
        )
        cfg = get_discretization()
        ref = _native.poisson_gaussian_pld(nm, p, cfg.to_native()).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_requires_n_steps(self):
        with pytest.raises(TypeError):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), sample_rate=0.1
            )

    def test_rejects_invalid_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.1,
                n_steps=0,
            )

    def test_rejects_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=1.5,
                n_steps=10,
            )


class TestPoissonBandMf:
    def test_pld_matches_self_composed_with_bands(self):
        """For BandMf: num_groups = ceil(n_steps / bands)."""
        nm, p = 1.1, 0.01
        bands = 2
        n_steps = 100
        strategy = band_mf_strategy(bands=bands)
        sens = strategy.sensitivity(n_steps=n_steps)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm / sens, strategy),
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
        from opaque.api.accounting.core.discretization import DiscretizationConfig

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
            f"identity τ=0 vs generic gap too large: id={eps_id_no_is}, ref={ref_eps}"
        )

    def test_factory_path_finite(self):
        """The default ``balls_in_bins`` factory produces a finite, positive ε."""
        nm, k, E = 1.5, 32, 4
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            num_bins=k,
            n_steps=k * E,
        ).epsilon_at(_DELTA)
        assert math.isfinite(eps)
        assert eps > 0

    @pytest.mark.slow
    def test_is_reduces_variance_on_ensemble(self):
        """The internal IS tilt reduces ε std vs no-IS over a large enough seed
        ensemble.  This is the *empirical* point of the IS specialisation;
        small-N seed ensembles can be unlucky (std varies as χ² with df=N-1),
        so we use 32 seeds and require ≥ 1.4× std reduction (squared variance
        ≥ 2×).  In practice the gain at this config is 3-30×.
        """
        import statistics

        from opaque.api.accounting.core.discretization import DiscretizationConfig
        from opaque.api.accounting.dpftrl.amplification._balls_in_bins import (
            _IDENTITY_IS_TILT,
        )

        nm, k, E = 1.5, 32, 4
        budget = 200_000
        seeds = list(range(32))
        eps_no_is, eps_with_is = [], []
        for s in seeds:
            cfg_native = DiscretizationConfig(num_mc_samples=budget, seed=s).to_native()
            eps_no_is.append(
                _native.bnb_mc_pld_identity(k, E, nm, 0.0, cfg_native).epsilon_at(
                    _DELTA
                )
            )
            eps_with_is.append(
                _native.bnb_mc_pld_identity(
                    k, E, nm, _IDENTITY_IS_TILT, cfg_native
                ).epsilon_at(_DELTA)
            )
        std_no_is = statistics.stdev(eps_no_is)
        std_with_is = statistics.stdev(eps_with_is)
        assert std_with_is < std_no_is / 1.4, (
            f"IS should reduce per-seed std vs τ=0 over 32 seeds: "
            f"std(τ=0)={std_no_is:.4f}, std(τ={_IDENTITY_IS_TILT})={std_with_is:.4f}"
        )

    def test_strictly_tighter_than_unamplified_composition(self):
        """Amplification (factor ~1/num_bins) must beat unamplified composition."""
        nm, k, E = 1.5, 32, 4
        amplified = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(nm, identity_strategy()), num_bins=k, n_steps=k * E
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
            ftrl_acc.mf_gaussian(0.0, identity_strategy()), num_bins=10, n_steps=20
        )
        assert math.isinf(proc.epsilon_at(_DELTA))

    def test_rejects_invalid_num_bins(self):
        with pytest.raises(ValueError, match="num_bins"):
            ftrl_acc.balls_in_bins(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), num_bins=1, n_steps=20
            )

    def test_rejects_invalid_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            ftrl_acc.balls_in_bins(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), num_bins=10, n_steps=0
            )

    def test_rejects_n_steps_not_multiple_of_num_bins(self):
        with pytest.raises(ValueError, match="multiple of"):
            ftrl_acc.balls_in_bins(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()), num_bins=10, n_steps=15
            )


# ---------------------------------------------------------------------------
# Calibration smoke
# ---------------------------------------------------------------------------


class TestTruncatedPoissonIdentity:
    """``ftrl_acc.poisson(identity_mf(...), ..., truncated_batch_size=, dataset_size=)``."""

    def test_rejects_unpaired_truncated_batch_size(self):
        with pytest.raises(ValueError, match="truncated_batch_size and dataset_size"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.01,
                n_steps=10,
                truncated_batch_size=64,
            )

    def test_rejects_unpaired_dataset_size(self):
        with pytest.raises(ValueError, match="truncated_batch_size and dataset_size"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.01,
                n_steps=10,
                dataset_size=10_000,
            )

    def test_rejects_truncated_batch_size_below_one(self):
        with pytest.raises(ValueError, match="truncated_batch_size must be >= 1"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.01,
                n_steps=10,
                truncated_batch_size=0,
                dataset_size=10_000,
            )

    def test_rejects_dataset_size_below_one(self):
        with pytest.raises(ValueError, match="dataset_size must be >= 1"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, identity_strategy()),
                sample_rate=0.01,
                n_steps=10,
                truncated_batch_size=64,
                dataset_size=0,
            )

    def test_rejects_band_mf_with_truncation(self):
        with pytest.raises(ValueError, match="IdentityStrategy"):
            ftrl_acc.poisson(
                ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
                sample_rate=0.01,
                n_steps=10,
                truncated_batch_size=64,
                dataset_size=10_000,
            )

    def test_pld_matches_self_composed_truncated_poisson_gaussian(self):
        """Truncated path is the per-step PLD composed ``n_steps`` times."""
        nm, p, T = 1.1, 0.01, 200
        cap, n = 64, 50_000
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            sample_rate=p,
            n_steps=T,
            truncated_batch_size=cap,
            dataset_size=n,
        )
        cfg = get_discretization()
        ref = _native.truncated_poisson_gaussian_pld(
            nm, p, cap, n, cfg.to_native()
        ).self_compose(T)
        assert math.isclose(
            proc.epsilon_at(_DELTA), ref.epsilon_at(_DELTA), rel_tol=1e-9
        )

    def test_truncated_finite_and_positive(self):
        eps = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.1, identity_strategy()),
            sample_rate=0.01,
            n_steps=200,
            truncated_batch_size=64,
            dataset_size=50_000,
        ).epsilon_at(_DELTA)
        assert math.isfinite(eps)
        assert eps > 0


def test_identity_mf_calibrates_through_poisson():
    cal = acc.calibrate(
        acc.epsilon_budget(3.0, delta=_DELTA),
        lambda nm: ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()), sample_rate=0.01, n_steps=500
        ),
        param_min=0.1,
        param_max=10.0,
    )
    assert cal.param > 0
    assert cal.achieved <= 3.0 + 1e-6
