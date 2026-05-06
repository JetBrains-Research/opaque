"""Tests for MF strategy factories and mf_noise()."""

import pytest
import torch

import opaque.dpftrl.accounting as ftrl_acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.types import clipped

from opaque.types import NoisedPytree

from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    lambda_cgd_strategy,
    mf_noise,
)
from opaque.dpftrl.noise.types import (
    BandMfStrategy,
    BisrStrategy,
    BltStrategy,
    LambdaCgdStrategy,
)
from opaque.random import key


# ── BandMfStrategy ──────────────────────────────────────────────────────


class TestBandMfStrategy:
    def test_returns_correct_type(self):
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert isinstance(s, BandMfStrategy)

    def test_sensitivity_is_one(self):
        """Optimized Toeplitz coefficients are L2-normalized."""
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert s.sensitivity == pytest.approx(1.0, abs=1e-6)

    def test_gram_matrix_is_none(self):
        """BandMF uses cyclic_poisson, not BnB."""
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert s.gram_matrix is None

    def test_coefficients_length(self):
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert len(s.coefficients) == 10

    def test_streaming_matrix_present(self):
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert s._streaming_matrix is not None

    def test_matches_old_sensitivity(self):
        ftrl_acc.band_mf(1.0, sensitivity=1.0, num_groups=10)
        new = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        assert new.sensitivity == pytest.approx(1.0, abs=1e-6)

    def test_rejects_bad_n_steps(self):
        with pytest.raises(ValueError):
            band_mf_strategy(n_steps=0, bands=5)

    def test_rejects_bad_bands(self):
        with pytest.raises(ValueError):
            band_mf_strategy(n_steps=100, bands=0)
        with pytest.raises(ValueError):
            band_mf_strategy(n_steps=100, bands=101)

    def test_with_lr_schedule(self):
        lr = torch.ones(100, dtype=torch.float64) * 0.01
        lr[:10] = torch.linspace(0.001, 0.01, 10)
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95, lr_schedule=lr)
        assert s.sensitivity == pytest.approx(1.0, abs=1e-6)


# ── BltStrategy ─────────────────────────────────────────────────────────


class TestBltStrategy:
    def test_returns_correct_type(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert isinstance(s, BltStrategy)

    def test_sensitivity_positive(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert s.sensitivity > 0

    def test_gram_matrix_present(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert s.gram_matrix is not None
        # Gram matrix is 25x25 flattened
        assert len(s.gram_matrix) == 25 * 25

    def test_coefficients_length(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert len(s.coefficients) == 100

    def test_streaming_matrix_present(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert s._streaming_matrix is not None

    def test_matches_old_sensitivity(self):
        new = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        assert new.sensitivity > 0

    def test_single_participation(self):
        s = blt_strategy(n_steps=50, min_sep=1, max_participations=1)
        assert s.sensitivity > 0


# ── LambdaCgdStrategy ──────────────────────────────────────────────────


class TestLambdaCgdStrategy:
    def test_returns_correct_type(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert isinstance(s, LambdaCgdStrategy)

    def test_sensitivity_positive(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert s.sensitivity > 0

    def test_gram_matrix_present(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert s.gram_matrix is not None
        assert len(s.gram_matrix) == 25 * 25

    def test_normalized_single_participation_sensitivity_one(self):
        """Normalized + single participation → sensitivity = 1.0."""
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=1, max_participations=1)
        assert s.sensitivity == pytest.approx(1.0, abs=1e-6)

    def test_matches_old_sensitivity(self):
        new = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert new.sensitivity > 0

    def test_with_momentum(self):
        """lambda_cgd_strategy does not accept momentum (use bisr_strategy)."""
        with pytest.raises(TypeError):
            lambda_cgd_strategy(
                0.5, n_steps=100, min_sep=25, max_participations=4, momentum=0.95
            )

    def test_unnormalized(self):
        s = lambda_cgd_strategy(
            0.9, n_steps=100, min_sep=25, max_participations=4, normalized=False
        )
        assert s.sensitivity > 0

    def test_rejects_bad_lambda(self):
        with pytest.raises(ValueError):
            lambda_cgd_strategy(-0.1, n_steps=100, min_sep=25)
        with pytest.raises(ValueError):
            lambda_cgd_strategy(1.0, n_steps=100, min_sep=25)

    def test_internal_fields(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        assert s._lambda == pytest.approx(0.9)
        assert s._n_steps == 100
        assert s._normalized is True


# ── BisrStrategy ────────────────────────────────────────────────────────


class TestBisrStrategy:
    def test_returns_correct_type(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert isinstance(s, BisrStrategy)

    def test_sensitivity_positive(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert s.sensitivity > 0

    def test_gram_matrix_present(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert s.gram_matrix is not None
        assert len(s.gram_matrix) == 25 * 25

    def test_streaming_matrix_present(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert s._streaming_matrix is not None

    def test_matches_old_sensitivity(self):
        new = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        assert new.sensitivity > 0

    def test_with_momentum(self):
        s = bisr_strategy(
            bandwidth=4, n_steps=100, min_sep=25, max_participations=4, momentum=0.95
        )
        assert s.sensitivity > 0

    def test_rejects_bad_bandwidth(self):
        with pytest.raises(ValueError):
            bisr_strategy(bandwidth=1, n_steps=100, min_sep=25)


# ── PLD equivalence ────────────────────────────────────────────────────


class TestPldEquivalence:
    """New strategy + mf_gaussian produces identical PLD to old class API."""

    delta = 1e-5

    def test_band_mf_pld(self):
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        eps_new = ftrl_acc.band_mf(1.0, sensitivity=s.sensitivity).epsilon_at(
            self.delta
        )
        # Should be finite positive
        assert eps_new > 0

    def test_band_mf_cyclic_poisson_pld(self):
        """Cyclic Poisson via band_mf matches manual poisson composition."""
        s = band_mf_strategy(n_steps=100, bands=10, momentum=0.95)
        sample_rate = 0.05

        eps_new = ftrl_acc.cyclic_poisson(
            ftrl_acc.band_mf(1.0, sensitivity=s.sensitivity, num_groups=s.num_groups),
            sample_rate=sample_rate,
        ).epsilon_at(self.delta)
        eps_manual = (
            dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0 / s.sensitivity), sample_rate)
            * s.num_groups
        ).epsilon_at(self.delta)
        assert eps_new == pytest.approx(eps_manual, abs=1e-10)

    def test_blt_pld(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        eps_new = ftrl_acc.blt(1.0, sensitivity=s.sensitivity).epsilon_at(self.delta)
        assert eps_new > 0

    def test_lambda_cgd_pld(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        eps_new = ftrl_acc.lambda_cgd(1.0, sensitivity=s.sensitivity).epsilon_at(
            self.delta
        )
        assert eps_new > 0

    def test_bisr_pld(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        eps_new = ftrl_acc.bisr(1.0, sensitivity=s.sensitivity).epsilon_at(self.delta)
        assert eps_new > 0


# ── BnB equivalence ────────────────────────────────────────────────────


class TestBnbEquivalence:
    """BnB via MfGaussian(gram_matrix) matches old class-based BnB."""

    delta = 1e-5

    def test_lambda_cgd_bnb(self):
        s = lambda_cgd_strategy(0.9, n_steps=100, min_sep=25, max_participations=4)
        bnb_new = ftrl_acc.balls_in_bins(
            ftrl_acc.lambda_cgd(
                1.0, sensitivity=s.sensitivity, gram_matrix=s.gram_matrix
            ),
            num_bins=25,
            num_epochs=4,
        )
        eps = bnb_new.epsilon_at(self.delta)
        assert eps > 0

    def test_blt_bnb(self):
        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4, momentum=0.95)
        bnb_new = ftrl_acc.balls_in_bins(
            ftrl_acc.blt(1.0, sensitivity=s.sensitivity, gram_matrix=s.gram_matrix),
            num_bins=25,
            num_epochs=4,
        )
        eps = bnb_new.epsilon_at(self.delta)
        assert eps > 0

    def test_bisr_bnb(self):
        s = bisr_strategy(bandwidth=4, n_steps=100, min_sep=25, max_participations=4)
        bnb_new = ftrl_acc.balls_in_bins(
            ftrl_acc.bisr(1.0, sensitivity=s.sensitivity, gram_matrix=s.gram_matrix),
            num_bins=25,
            num_epochs=4,
        )
        eps = bnb_new.epsilon_at(self.delta)
        assert eps > 0

    def test_mf_without_gram_raises(self):
        """Blt without gram_matrix is rejected by balls_in_bins."""
        with pytest.raises(ValueError):
            ftrl_acc.balls_in_bins(
                ftrl_acc.blt(1.0, sensitivity=1.0),
                num_bins=25,
                num_epochs=4,
            ).epsilon_at(1e-5)


# ── mf_noise() tests ───────────────────────────────────────────────────


class TestMfNoise:
    """mf_noise() produces noise for all strategy types."""

    grad = {"w": torch.randn(3, 4)}

    def _clipped_grad(self):
        return clipped(self.grad, max_norm=1.0)

    def test_band_mf_noise(self):
        s = band_mf_strategy(n_steps=50, bands=10, momentum=0.95)
        nf, ns = mf_noise(self.grad, s, noise_multiplier=1.0, key=key(42))
        noised, ns2 = nf(self._clipped_grad(), ns)
        assert ns2._step_counter == 1
        assert isinstance(noised, NoisedPytree)
        assert noised.max_norm == pytest.approx(1.0)
        assert noised.noise_stddev == pytest.approx(1.0)
        assert "w" in noised.pytree
        assert noised.pytree["w"].shape == self.grad["w"].shape

    def test_blt_noise(self):
        s = blt_strategy(n_steps=50, min_sep=10, max_participations=5, momentum=0.95)
        nf, ns = mf_noise(self.grad, s, noise_multiplier=1.0, key=key(42))
        noised, ns2 = nf(self._clipped_grad(), ns)
        assert ns2._step_counter == 1
        assert isinstance(noised, NoisedPytree)

    def test_lambda_cgd_noise(self):
        s = lambda_cgd_strategy(0.9, n_steps=50, min_sep=10, max_participations=5)
        nf, ns = mf_noise(self.grad, s, noise_multiplier=1.0, key=key(42))
        noised, ns2 = nf(self._clipped_grad(), ns)
        assert ns2._step_counter == 1
        assert isinstance(noised, NoisedPytree)

    def test_bisr_noise(self):
        s = bisr_strategy(bandwidth=4, n_steps=50, min_sep=10, max_participations=5)
        nf, ns = mf_noise(self.grad, s, noise_multiplier=1.0, key=key(42))
        noised, ns2 = nf(self._clipped_grad(), ns)
        assert ns2._step_counter == 1
        assert isinstance(noised, NoisedPytree)

    def test_lambda_cgd_multi_step(self):
        """λCGD noise correctly handles step 0 (no prev) and step 1+ (with prev)."""
        s = lambda_cgd_strategy(0.9, n_steps=50, min_sep=10, max_participations=5)
        nf, ns = mf_noise(self.grad, s, noise_multiplier=1.0, key=key(42))
        _, ns = nf(self._clipped_grad(), ns)
        _, ns = nf(self._clipped_grad(), ns)
        _, ns = nf(self._clipped_grad(), ns)
        assert ns._step_counter == 3

    def test_noise_is_nonzero(self):
        """Noise adds something to the gradients."""
        s = band_mf_strategy(n_steps=50, bands=10, momentum=0.95)
        nf, ns = mf_noise(self.grad, s, noise_multiplier=1.0, key=key(42))
        noised, _ = nf(self._clipped_grad(), ns)
        diff = torch.norm(noised.pytree["w"] - self.grad["w"])
        assert diff > 0.1

    def test_raw_grads_are_rejected(self):
        s = band_mf_strategy(n_steps=50, bands=10, momentum=0.95)
        nf, ns = mf_noise(self.grad, s, noise_multiplier=1.0, key=key(42))
        with pytest.raises(TypeError, match="ClippedPytree"):
            nf(self.grad, ns)
