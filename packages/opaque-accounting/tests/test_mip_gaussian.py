"""Tests for MipGaussian mechanism and its integration with Poisson/AdaClip."""

import math

import pytest

import opaque_accounting as acc
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms.mip_gaussian import (
    MipGaussian,
    mip_gaussian,
    _bin_norms,
)


# ---------------------------------------------------------------------------
# Constructor & binning
# ---------------------------------------------------------------------------


class TestBinNorms:
    def test_uniform_norms(self):
        norms = [0.5] * 100
        sensitivities, weights = _bin_norms(norms)
        assert len(sensitivities) == 1
        assert weights[0] == pytest.approx(1.0)

    def test_two_clusters(self):
        norms = [0.2] * 80 + [0.8] * 20
        sensitivities, weights = _bin_norms(norms, num_bins=100)
        assert len(sensitivities) == 2
        assert sum(weights) == pytest.approx(1.0)
        # First cluster should have weight ~0.8
        assert weights[0] == pytest.approx(0.8)

    def test_weights_sum_to_one(self):
        norms = [0.1 * i for i in range(1, 101)]
        sensitivities, weights = _bin_norms(norms)
        assert sum(weights) == pytest.approx(1.0)

    def test_sorted_sensitivities(self):
        norms = [1.0, 0.1, 0.5, 0.3]
        sensitivities, _ = _bin_norms(norms)
        assert list(sensitivities) == sorted(sensitivities)


class TestMipGaussianConstructor:
    def test_basic(self):
        proc = mip_gaussian(0.8, [0.1, 0.3, 0.5, 0.7, 1.0])
        assert isinstance(proc, MipGaussian)
        assert proc.noise_multiplier == 0.8

    def test_is_dp_process(self):
        proc = mip_gaussian(0.8, [0.5, 1.0])
        assert isinstance(proc, DpProcess)

    def test_frozen(self):
        proc = mip_gaussian(0.8, [0.5])
        with pytest.raises(AttributeError):
            proc.noise_multiplier = 1.0

    def test_rejects_empty_norms(self):
        with pytest.raises(ValueError, match="non-empty"):
            mip_gaussian(0.8, [])

    def test_rejects_negative_norms(self):
        with pytest.raises(ValueError, match="non-negative"):
            mip_gaussian(0.8, [-0.1, 0.5])

    def test_rejects_zero_noise(self):
        with pytest.raises(ValueError, match="positive"):
            mip_gaussian(0.0, [0.5])

    def test_all_zero_norms_has_zero_privacy_loss(self):
        proc = mip_gaussian(0.8, [0.0, 0.0, 0.0])
        assert isinstance(proc, MipGaussian)
        assert proc.epsilon_at(1e-5) == 0.0

    def test_all_zero_norms_composes_with_poisson(self):
        proc = mip_gaussian(0.8, [0.0, 0.0, 0.0])
        step = acc.poisson(proc, sample_rate=0.01)
        assert step.epsilon_at(1e-5) == 0.0


# ---------------------------------------------------------------------------
# PLD computation
# ---------------------------------------------------------------------------


class TestMipGaussianPld:
    def test_returns_finite_epsilon(self):
        proc = mip_gaussian(0.8, [0.5, 1.0] * 50)
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_single_sensitivity_matches_gaussian(self):
        """MipGaussian with uniform sensitivity=1 should match Gaussian."""
        mip = MipGaussian(noise_multiplier=0.5, sensitivities=(1.0,), weights=(1.0,))
        gauss = acc.gaussian(0.5)

        mip_eps = mip.epsilon_at(1e-5)
        gauss_eps = gauss.epsilon_at(1e-5)

        assert mip_eps == pytest.approx(gauss_eps, abs=1e-2)

    def test_lower_sensitivities_improve_epsilon(self):
        """Mostly-low sensitivities should give better (lower) epsilon."""
        worst = MipGaussian(noise_multiplier=0.5, sensitivities=(1.0,), weights=(1.0,))
        mixed = MipGaussian(
            noise_multiplier=0.5,
            sensitivities=(0.3, 1.0),
            weights=(0.9, 0.1),
        )
        assert mixed.epsilon_at(1e-5) < worst.epsilon_at(1e-5)

    def test_sensitivity_monotonicity(self):
        """Higher max sensitivity should give worse (higher) epsilon."""
        low = MipGaussian(noise_multiplier=0.5, sensitivities=(0.5,), weights=(1.0,))
        high = MipGaussian(noise_multiplier=0.5, sensitivities=(1.0,), weights=(1.0,))
        assert low.epsilon_at(1e-5) < high.epsilon_at(1e-5)


# ---------------------------------------------------------------------------
# Poisson amplification
# ---------------------------------------------------------------------------


class TestPoissonMipGaussian:
    def test_basic(self):
        step = acc.poisson(mip_gaussian(0.8, [0.5, 1.0] * 50), sample_rate=0.01)
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_reduces_epsilon_vs_standalone(self):
        norms = [0.5, 1.0] * 50
        standalone = mip_gaussian(0.5, norms)
        subsampled = acc.poisson(mip_gaussian(0.5, norms), sample_rate=0.01)
        assert subsampled.epsilon_at(1e-5) < standalone.epsilon_at(1e-5)

    def test_single_sensitivity_matches_poisson_gaussian(self):
        """Poisson(MipGaussian(s=1)) should match Poisson(Gaussian)."""
        mip_step = acc.poisson(
            MipGaussian(noise_multiplier=0.5, sensitivities=(1.0,), weights=(1.0,)),
            sample_rate=0.01,
        )
        std_step = acc.poisson(acc.gaussian(0.5), sample_rate=0.01)

        assert mip_step.epsilon_at(1e-5) == pytest.approx(
            std_step.epsilon_at(1e-5), abs=1e-2
        )

    def test_composition(self):
        norms = [0.3, 0.7, 1.0] * 100
        step = acc.poisson(mip_gaussian(0.8, norms), sample_rate=0.01)
        training = step * 100
        eps = training.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0


# ---------------------------------------------------------------------------
# AdaClip integration
# ---------------------------------------------------------------------------


class TestAdaClipMipGaussian:
    def test_adaclip_wrapping(self):
        inner = mip_gaussian(0.8, [0.5, 1.0] * 50)
        wrapped = acc.adaclip(inner, batch_size=256)
        eps = wrapped.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_poisson_adaclip_mip(self):
        inner = mip_gaussian(0.8, [0.5, 1.0] * 50)
        step = acc.poisson(acc.adaclip(inner, batch_size=256), sample_rate=0.01)
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps) and eps > 0

    def test_adaclip_increases_epsilon(self):
        """AdaClip adds quantile noise cost, so epsilon should increase."""
        inner = mip_gaussian(0.8, [0.5, 1.0] * 50)
        bare = acc.poisson(inner, sample_rate=0.01)
        with_adaclip = acc.poisson(acc.adaclip(inner, batch_size=256), sample_rate=0.01)
        assert with_adaclip.epsilon_at(1e-5) > bare.epsilon_at(1e-5)


# ---------------------------------------------------------------------------
# Truncated Poisson stub
# ---------------------------------------------------------------------------


class TestTruncatedPoissonMipGaussian:
    def test_rejects_mip_gaussian(self):
        inner = mip_gaussian(0.8, [0.5, 1.0])
        with pytest.raises(TypeError, match="MipGaussian"):
            acc.truncated_poisson(
                inner, sample_rate=0.01, batch_size_cap=256, dataset_size=10000
            )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestMipGaussianStateDicts:
    def test_round_trip(self):
        proc = MipGaussian(
            noise_multiplier=0.8,
            sensitivities=(0.3, 0.7, 1.0),
            weights=(0.5, 0.3, 0.2),
        )
        state = proc.state_dict()
        restored = DpProcess.from_state_dict(state)
        assert restored == proc

    def test_poisson_round_trip(self):
        proc = acc.poisson(
            MipGaussian(
                noise_multiplier=0.8,
                sensitivities=(0.5, 1.0),
                weights=(0.7, 0.3),
            ),
            sample_rate=0.01,
        )
        state = proc.state_dict()
        restored = DpProcess.from_state_dict(state)
        assert restored == proc
