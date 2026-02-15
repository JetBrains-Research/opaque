"""Validation tests comparing opaque_dp_accounting against Google's dp_accounting.

This test suite ensures that the Rust-based functional API produces results
consistent with the reference Python implementation (Google's dp-accounting).

The tolerance is derived from Connect-the-Dots discretization error: O(disc^2)
per step, accumulated linearly across k composition steps. With disc=1e-4:
- k=1:   tol ~1e-7
- k=100: tol ~1e-5
- k=1000: tol ~1e-4

We use conservative tolerances to account for differences in:
- Truncation bounds (-32 in ours vs variable in Google's)
- Numerical libraries (Rust statrs vs SciPy)
- PMF representation details
"""

import math

import pytest

import opaque_dp_accounting as ours

# Google's dp_accounting library
from dp_accounting.pld import privacy_loss_distribution as pld_lib
from dp_accounting.pld import common


# =============================================================================
# Test helpers
# =============================================================================


def google_gaussian_pld(noise_multiplier, num_steps=1):
    """Create a PLD using Google's library for a Gaussian mechanism."""
    pld = pld_lib.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sensitivity=1.0,
        log_mass_truncation_bound=math.log(2) * (-32.0),
        value_discretization_interval=1e-4,
    )
    if num_steps > 1:
        pld = pld.self_compose(num_steps)
    return pld


def google_poisson_gaussian_pld(noise_multiplier, sample_rate, num_steps=1):
    """Create a PLD for Poisson-subsampled Gaussian using Google's library."""
    pld = pld_lib.from_gaussian_mechanism(
        standard_deviation=noise_multiplier,
        sensitivity=1.0,
        sampling_prob=sample_rate,
        log_mass_truncation_bound=math.log(2) * (-32.0),
        value_discretization_interval=1e-4,
        use_connect_dots=True,
    )
    if num_steps > 1:
        pld = pld.self_compose(num_steps)
    return pld


# =============================================================================
# 1. Single Gaussian mechanism
# =============================================================================


class TestGaussianMechanism:
    """Compare single Gaussian mechanism results."""

    @pytest.mark.parametrize("noise_multiplier", [0.5, 0.8, 1.0, 1.1, 1.2])
    def test_epsilon_at_delta_1e5(self, noise_multiplier):
        """epsilon_at(delta=1e-5) should match Google's PLD."""
        delta = 1e-5

        our_proc = ours.py_gaussian(noise_multiplier)
        our_eps = our_proc.epsilon_at(delta)

        google_pld = google_gaussian_pld(noise_multiplier)
        google_eps = google_pld.get_epsilon_for_delta(delta)

        # Tolerance: single step, disc=1e-4 → O(1e-8) discretization error
        # Use 1e-4 relative tolerance to be safe
        assert abs(our_eps - google_eps) < max(1e-4, 1e-3 * abs(google_eps)), (
            f"nm={noise_multiplier}: ours={our_eps:.8f}, google={google_eps:.8f}, "
            f"diff={abs(our_eps - google_eps):.2e}"
        )

    @pytest.mark.parametrize("noise_multiplier", [0.5, 0.8, 1.0, 1.2])
    def test_delta_at_epsilon_1(self, noise_multiplier):
        """delta_at(epsilon=1.0) should match Google's PLD."""
        epsilon = 1.0

        our_proc = ours.py_gaussian(noise_multiplier)
        our_delta = our_proc.delta_at(epsilon)

        google_pld = google_gaussian_pld(noise_multiplier)
        google_delta = google_pld.get_delta_for_epsilon(epsilon)

        assert abs(our_delta - google_delta) < max(1e-6, 1e-3 * abs(google_delta)), (
            f"nm={noise_multiplier}: ours={our_delta:.10f}, google={google_delta:.10f}, "
            f"diff={abs(our_delta - google_delta):.2e}"
        )


# =============================================================================
# 2. Gaussian composition (repeat k times)
# =============================================================================


class TestGaussianComposition:
    """Compare composed Gaussian mechanism results."""

    @pytest.mark.parametrize(
        "noise_multiplier,num_steps",
        [(1.0, 10), (1.0, 100), (0.8, 50), (1.1, 200)],
    )
    def test_composition_epsilon(self, noise_multiplier, num_steps):
        """Composed Gaussian epsilon should match Google's self_compose."""
        delta = 1e-5

        our_step = ours.py_gaussian(noise_multiplier)
        our_training = ours.py_repeat(our_step, num_steps)
        our_eps = our_training.epsilon_at(delta)

        google_pld = google_gaussian_pld(noise_multiplier, num_steps)
        google_eps = google_pld.get_epsilon_for_delta(delta)

        # Tolerance scales with k: O(disc^2 * k)
        tol = max(1e-3, 5e-3 * abs(google_eps))
        assert abs(our_eps - google_eps) < tol, (
            f"nm={noise_multiplier}, k={num_steps}: ours={our_eps:.6f}, "
            f"google={google_eps:.6f}, diff={abs(our_eps - google_eps):.2e}"
        )


# =============================================================================
# 3. Poisson-subsampled Gaussian
# =============================================================================


class TestPoissonGaussian:
    """Compare Poisson-subsampled Gaussian results."""

    @pytest.mark.parametrize(
        "noise_multiplier,sample_rate",
        [(1.0, 0.01), (1.0, 0.001), (0.8, 0.01), (1.1, 0.005)],
    )
    def test_single_step_epsilon(self, noise_multiplier, sample_rate):
        """Single Poisson-subsampled Gaussian step should match Google."""
        delta = 1e-5

        our_proc = ours.py_poisson(noise_multiplier, sample_rate)
        our_eps = our_proc.epsilon_at(delta)

        google_pld = google_poisson_gaussian_pld(noise_multiplier, sample_rate)
        google_eps = google_pld.get_epsilon_for_delta(delta)

        tol = max(1e-4, 5e-3 * abs(google_eps))
        assert abs(our_eps - google_eps) < tol, (
            f"nm={noise_multiplier}, q={sample_rate}: ours={our_eps:.8f}, "
            f"google={google_eps:.8f}, diff={abs(our_eps - google_eps):.2e}"
        )

    @pytest.mark.parametrize(
        "noise_multiplier,sample_rate,num_steps",
        [(1.0, 0.01, 100), (1.0, 0.01, 1000), (1.1, 0.005, 500)],
    )
    def test_composed_epsilon(self, noise_multiplier, sample_rate, num_steps):
        """Composed Poisson-subsampled Gaussian should match Google."""
        delta = 1e-5

        our_eps = ours.py_compute_epsilon(noise_multiplier, sample_rate, num_steps, delta)

        google_pld = google_poisson_gaussian_pld(noise_multiplier, sample_rate, num_steps)
        google_eps = google_pld.get_epsilon_for_delta(delta)

        # For large compositions, discretization error accumulates
        tol = max(1e-3, 0.01 * abs(google_eps))
        assert abs(our_eps - google_eps) < tol, (
            f"nm={noise_multiplier}, q={sample_rate}, k={num_steps}: "
            f"ours={our_eps:.6f}, google={google_eps:.6f}, "
            f"diff={abs(our_eps - google_eps):.2e}"
        )


# =============================================================================
# 4. Realistic DP-SGD scenarios
# =============================================================================


class TestRealisticDPSGD:
    """End-to-end tests for realistic DP-SGD training configurations."""

    def test_standard_dpsgd(self):
        """Standard DP-SGD: nm=1.1, q=0.01, 1000 steps."""
        nm, q, steps, delta = 1.1, 0.01, 1000, 1e-5

        our_eps = ours.py_compute_epsilon(nm, q, steps, delta)
        google_pld = google_poisson_gaussian_pld(nm, q, steps)
        google_eps = google_pld.get_epsilon_for_delta(delta)

        tol = max(0.01, 0.01 * abs(google_eps))
        assert abs(our_eps - google_eps) < tol, (
            f"DP-SGD: ours={our_eps:.6f}, google={google_eps:.6f}"
        )

    def test_large_noise_short_training(self):
        """Low-eps regime: nm=1.0, q=0.001, 100 steps."""
        nm, q, steps, delta = 1.0, 0.001, 100, 1e-5

        our_eps = ours.py_compute_epsilon(nm, q, steps, delta)
        google_pld = google_poisson_gaussian_pld(nm, q, steps)
        google_eps = google_pld.get_epsilon_for_delta(delta)

        tol = max(1e-3, 0.01 * abs(google_eps))
        assert abs(our_eps - google_eps) < tol, (
            f"Low-eps: ours={our_eps:.6f}, google={google_eps:.6f}"
        )


# =============================================================================
# 5. Identity and edge cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special mechanisms."""

    def test_identity_epsilon_zero(self):
        """Identity mechanism should have epsilon=0 for any delta."""
        proc = ours.py_identity()
        eps = proc.epsilon_at(1e-5)
        assert abs(eps) < 1e-10, f"Identity epsilon should be ~0, got {eps}"

    def test_identity_delta_zero(self):
        """Identity mechanism should have delta=0 for epsilon=0."""
        proc = ours.py_identity()
        delta = proc.delta_at(0.0)
        assert abs(delta) < 1e-10, f"Identity delta should be ~0, got {delta}"

    def test_eps_delta_mechanism(self):
        """EpsDelta mechanism should faithfully represent given guarantees."""
        proc = ours.py_eps_delta(1.0, 1e-5)
        eps = proc.epsilon_at(1e-5)
        # Should be close to 1.0 (the specified epsilon)
        assert abs(eps - 1.0) < 0.1, f"Expected eps~1.0, got {eps}"

    def test_compose_two_processes(self):
        """Composing two processes should increase epsilon."""
        step = ours.py_gaussian(1.0)
        single_eps = step.epsilon_at(1e-5)

        composed = ours.py_compose(step, ours.py_gaussian(1.0))
        composed_eps = composed.epsilon_at(1e-5)

        # Composed epsilon should be larger than single
        assert composed_eps > single_eps, (
            f"Composed epsilon ({composed_eps}) should exceed single ({single_eps})"
        )


# =============================================================================
# 6. AdaClip mechanism
# =============================================================================


class TestAdaClip:
    """Test adaptive clipping mechanism."""

    def test_adaclip_more_private_than_base(self):
        """AdaClip epsilon should be >= base Gaussian (extra privacy cost)."""
        nm = 1.0
        base = ours.py_gaussian(nm)
        ac = ours.py_adaclip(nm, 50.0)

        base_eps = base.epsilon_at(1e-5)
        ac_eps = ac.epsilon_at(1e-5)

        # AdaClip adds the quantile query cost, so epsilon should increase
        assert ac_eps >= base_eps - 1e-6, (
            f"AdaClip eps ({ac_eps}) should be >= base eps ({base_eps})"
        )

    def test_adaclip_large_sigma_b_matches_base(self):
        """With very large sigma_b, AdaClip should approximate the base Gaussian."""
        nm = 1.0
        base = ours.py_gaussian(nm)
        ac = ours.py_adaclip(nm, 1e10)

        base_eps = base.epsilon_at(1e-5)
        ac_eps = ac.epsilon_at(1e-5)

        assert abs(ac_eps - base_eps) < 0.01, (
            f"Large sigma_b: ac_eps={ac_eps}, base_eps={base_eps}"
        )


# =============================================================================
# 7. Calibration
# =============================================================================


class TestCalibration:
    """Test noise calibration."""

    def test_calibrate_noise_basic(self):
        """Calibrated noise should achieve approximately the target epsilon."""
        target_eps = 8.0
        delta = 1e-5
        q = 0.01
        steps = 1000

        # noise_multiplier range is [0.1, 1.2] due to numerical stability
        nm = ours.py_calibrate_noise(
            target_epsilon=target_eps,
            target_delta=delta,
            sample_rate=q,
            num_steps=steps,
            param_min=0.1,
            param_max=1.2,
        )

        # Verify the calibrated noise achieves the target
        actual_eps = ours.py_compute_epsilon(nm, q, steps, delta)
        assert abs(actual_eps - target_eps) < 0.1, (
            f"Calibrated nm={nm:.4f} gives eps={actual_eps:.6f}, target={target_eps}"
        )


# =============================================================================
# 8. Metrics consistency
# =============================================================================


class TestMetrics:
    """Test that different metrics are consistent with each other."""

    def test_delta_epsilon_roundtrip(self):
        """delta_at(epsilon_at(delta)) should be approximately delta."""
        proc = ours.py_gaussian(1.0)
        target_delta = 1e-5

        eps = proc.epsilon_at(target_delta)
        recovered_delta = proc.delta_at(eps)

        assert abs(recovered_delta - target_delta) < 1e-4, (
            f"Round-trip: target={target_delta}, recovered={recovered_delta}"
        )

    def test_advantage_is_delta_at_zero(self):
        """Advantage should equal delta_at(epsilon=0)."""
        proc = ours.py_gaussian(1.0)
        adv = proc.advantage()
        delta_0 = proc.delta_at(0.0)

        assert abs(adv - delta_0) < 1e-6, (
            f"advantage={adv}, delta_at(0)={delta_0}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
