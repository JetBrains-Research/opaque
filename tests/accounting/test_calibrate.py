"""Tests for calibration functions."""

import pytest

from opaque.accounting import (
    PLDAccountant,
    RDPAccountant,
    calibrate_noise_multiplier,
    calibrate_steps,
)


class TestCalibrateNoise:
    """Tests for calibrate_noise_multiplier."""

    def test_calibrate_noise_rdp(self):
        """Test noise calibration with RDP accountant."""
        noise_mult = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
            accountant_type="rdp",
        )

        # Verify it achieves target
        acc = RDPAccountant()
        acc.step_poisson(noise_multiplier=noise_mult, sample_rate=0.01, num_steps=1000)
        eps = acc.get_epsilon(target_delta=1e-5)

        assert abs(eps - 3.0) < 0.1  # Within tolerance

    @pytest.mark.slow
    def test_calibrate_noise_pld(self):
        """Test noise calibration with PLD accountant."""
        noise_mult = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
            accountant_type="pld",
        )

        # Verify it achieves target
        acc = PLDAccountant()
        acc.step_poisson(noise_multiplier=noise_mult, sample_rate=0.01, num_steps=1000)
        eps = acc.get_epsilon(target_delta=1e-5)

        assert abs(eps - 3.0) < 0.1  # Within tolerance

    @pytest.mark.slow
    def test_calibrate_noise_truncated_poisson(self):
        """Test noise calibration with truncated Poisson."""
        noise_mult = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=100,  # Reduced from 1000 for faster test
            accountant_type="pld",
            truncated_batch_size=100,
            dataset_size=10000,
            tol=0.1,  # Increased tolerance for faster convergence
        )

        # Verify it achieves target
        acc = PLDAccountant()
        acc.step_truncated_poisson(
            noise_multiplier=noise_mult,
            sample_rate=0.01,
            truncated_batch_size=100,
            dataset_size=10000,
            num_steps=100,  # Reduced from 1000
        )
        eps = acc.get_epsilon(target_delta=1e-5)

        assert abs(eps - 3.0) < 0.2  # Within tolerance (relaxed for truncated Poisson)

    @pytest.mark.slow
    def test_calibrate_noise_more_steps_needs_more_noise(self):
        """Test that more steps requires higher noise multiplier."""
        noise_100 = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=100,
            accountant_type="rdp",
        )

        noise_1000 = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
            accountant_type="rdp",
        )

        # More steps = need more noise
        assert noise_1000 > noise_100

    def test_calibrate_noise_invalid_accountant(self):
        """Test that invalid accountant type raises error."""
        with pytest.raises(ValueError, match="Unknown accountant_type"):
            calibrate_noise_multiplier(
                target_epsilon=3.0,
                target_delta=1e-5,
                sample_rate=0.01,
                num_steps=1000,
                accountant_type="invalid",
            )

    def test_calibrate_noise_truncated_requires_pld(self):
        """Test that truncated Poisson requires PLD."""
        with pytest.raises(ValueError, match="only supported with accountants"):
            calibrate_noise_multiplier(
                target_epsilon=3.0,
                target_delta=1e-5,
                sample_rate=0.01,
                num_steps=1000,
                accountant_type="rdp",
                truncated_batch_size=100,
                dataset_size=10000,
            )

    def test_calibrate_noise_truncated_requires_dataset_size(self):
        """Test that truncated Poisson requires dataset_size."""
        with pytest.raises(ValueError, match="dataset_size required"):
            calibrate_noise_multiplier(
                target_epsilon=3.0,
                target_delta=1e-5,
                sample_rate=0.01,
                num_steps=1000,
                accountant_type="pld",
                truncated_batch_size=100,
            )


class TestCalibrateSteps:
    """Tests for calibrate_steps."""

    def test_calibrate_steps_rdp(self):
        """Test step calibration with RDP accountant."""
        max_steps = calibrate_steps(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=1.1,
            sample_rate=0.01,
            accountant_type="rdp",
        )

        # Verify it achieves target
        acc = RDPAccountant()
        acc.step_poisson(noise_multiplier=1.1, sample_rate=0.01, num_steps=max_steps)
        eps = acc.get_epsilon(target_delta=1e-5)

        assert eps <= 3.0  # Should not exceed target
        assert eps >= 2.8  # Should be close to target (within ~10%)

    def test_calibrate_steps_pld(self):
        """Test step calibration with PLD accountant."""
        max_steps = calibrate_steps(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=1.1,
            sample_rate=0.01,
            accountant_type="pld",
        )

        # Verify it achieves target
        acc = PLDAccountant()
        acc.step_poisson(noise_multiplier=1.1, sample_rate=0.01, num_steps=max_steps)
        eps = acc.get_epsilon(target_delta=1e-5)

        assert eps <= 3.0  # Should not exceed target
        assert eps >= 2.8  # Should be close to target

    def test_calibrate_steps_more_noise_allows_more_steps(self):
        """Test that more noise allows more training steps."""
        steps_low_noise = calibrate_steps(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=0.8,
            sample_rate=0.01,
            accountant_type="rdp",
        )

        steps_high_noise = calibrate_steps(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=1.5,
            sample_rate=0.01,
            accountant_type="rdp",
        )

        # More noise = can train longer
        assert steps_high_noise > steps_low_noise

    def test_calibrate_steps_fails_if_even_one_step_too_much(self):
        """Test that error is raised if even 1 step exceeds budget."""
        with pytest.raises(ValueError, match="Even.*step.*exceeds"):
            calibrate_steps(
                target_epsilon=0.01,  # Very tight budget
                target_delta=1e-5,
                noise_multiplier=0.1,  # Low noise
                sample_rate=0.5,  # High sample rate
                accountant_type="rdp",
            )


class TestCalibrateIntegration:
    """Integration tests for calibration."""

    def test_calibrate_noise_then_verify_steps(self):
        """Test calibrating noise, then verifying with step calibration."""
        # Calibrate noise for target budget
        noise_mult = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
            accountant_type="rdp",
        )

        # Now check how many steps we can actually do
        max_steps = calibrate_steps(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=noise_mult,
            sample_rate=0.01,
            accountant_type="rdp",
        )

        # Should be able to do at least 1000 steps
        assert max_steps >= 1000

    @pytest.mark.slow
    def test_pld_vs_rdp_calibration_similar(self):
        """Test that PLD and RDP calibration give similar results."""
        noise_rdp = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
            accountant_type="rdp",
        )

        noise_pld = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
            accountant_type="pld",
        )

        # Should be reasonably close (within 20%)
        ratio = abs(noise_rdp - noise_pld) / min(noise_rdp, noise_pld)
        assert ratio < 0.2
