"""Tests for calibration using riskcal.calibration.core primitives."""

import pytest

import opaque.accounting as acc


class TestEpsilonDeltaCalibration:
    """Test epsilon/delta calibration."""

    def test_basic_calibration(self):
        """Test basic epsilon/delta calibration with Poisson sampling."""
        noise = acc.find_noise_multiplier_for_epsilon_delta(
            epsilon=3.0,
            delta=1e-5,
            num_steps=1000,
            sampling_method="poisson",
            sample_rate=0.01,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify with opaque.accounting
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=noise, sample_rate=0.01, count=1000
        )
        achieved_eps = acc.get_epsilon(state, delta=1e-5)

        # Should be within tolerance
        assert abs(achieved_eps - 3.0) < 0.01

    def test_generic_interface(self):
        """Test using generic calibrate_parameter interface with Poisson sampling."""
        # Create evaluator
        evaluator = acc.create_dpsgd_epsilon_evaluator(
            num_steps=1000,
            target_delta=1e-5,
            sampling_method="poisson",
            sample_rate=0.01,
        )

        # Define target
        target = acc.CalibrationTarget(
            kind="epsilon_delta",
            epsilon=3.0,
            delta=1e-5,
        )

        # Configure search
        config = acc.CalibrationConfig(
            param_min=0.1,
            param_max=50.0,
            target_tol=1e-3,
            increasing=False,
        )

        # Calibrate
        result = acc.calibrate_parameter(evaluator, target, config)

        assert result.converged
        assert result.achieved_epsilon <= 3.0 + 0.01
        assert abs(result.achieved_epsilon - 3.0) < 0.01

    def test_query_function(self):
        """Test get_epsilon_for_dpsgd query function with Poisson sampling."""
        eps = acc.get_epsilon_for_dpsgd(
            noise_multiplier=1.1,
            num_steps=1000,
            delta=1e-5,
            sampling_method="poisson",
            sample_rate=0.01,
        )

        assert isinstance(eps, float)
        assert eps > 0

        # Should match opaque.accounting
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.1, sample_rate=0.01, count=1000
        )
        eps_check = acc.get_epsilon(state, delta=1e-5)

        assert abs(eps - eps_check) < 0.01

    def test_fixed_batch_sampling(self):
        """Test calibration with fixed-batch sampling."""
        noise = acc.find_noise_multiplier_for_epsilon_delta(
            epsilon=3.0,
            delta=1e-5,
            num_steps=1000,
            sampling_method="fixed_batch",
            batch_size=32,
            dataset_size=10000,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify with opaque.accounting
        state = acc.create()
        state = acc.compose_sampled_gaussian(
            state,
            noise_multiplier=noise,
            batch_size=32,
            dataset_size=10000,
            count=1000,
        )
        achieved_eps = acc.get_epsilon(state, delta=1e-5)

        # Should be within tolerance
        assert abs(achieved_eps - 3.0) < 0.01

    def test_truncated_poisson_sampling(self):
        """Test calibration with truncated Poisson sampling."""
        noise = acc.find_noise_multiplier_for_epsilon_delta(
            epsilon=3.0,
            delta=1e-5,
            num_steps=1000,
            sampling_method="truncated_poisson",
            sample_rate=0.0032,
            truncated_batch_size=32,
            dataset_size=10000,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify with opaque.accounting
        state = acc.create()
        state = acc.compose_truncated_poisson_gaussian(
            state,
            noise_multiplier=noise,
            sample_rate=0.0032,
            truncated_batch_size=32,
            dataset_size=10000,
            count=1000,
        )
        achieved_eps = acc.get_epsilon(state, delta=1e-5)

        # Should be within tolerance
        assert abs(achieved_eps - 3.0) < 0.01


class TestAdvantageCalibration:
    """Test advantage calibration."""

    def test_basic_calibration(self):
        """Test basic advantage calibration with Poisson sampling."""
        noise = acc.find_noise_multiplier_for_advantage(
            advantage=0.1,
            num_steps=1000,
            sampling_method="poisson",
            sample_rate=0.01,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify
        achieved = acc.get_advantage_for_dpsgd(
            noise_multiplier=noise,
            num_steps=1000,
            sampling_method="poisson",
            sample_rate=0.01,
        )

        assert abs(achieved - 0.1) < 0.01

    def test_matches_opaque_accounting(self):
        """Test that advantage calibration matches opaque.accounting."""
        noise = 1.1
        sample_rate = 0.01
        num_steps = 500

        # Get from calibration
        adv_calib = acc.get_advantage_for_dpsgd(
            noise_multiplier=noise,
            num_steps=num_steps,
            sampling_method="poisson",
            sample_rate=sample_rate,
        )

        # Get from opaque.accounting
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=noise, sample_rate=sample_rate, count=num_steps
        )
        adv_opaque = acc.get_advantage(state)

        # Should match (within tolerance)
        assert abs(adv_calib - adv_opaque) < 0.01

    def test_fixed_batch_sampling(self):
        """Test advantage calibration with fixed-batch sampling."""
        noise = acc.find_noise_multiplier_for_advantage(
            advantage=0.1,
            num_steps=1000,
            sampling_method="fixed_batch",
            batch_size=32,
            dataset_size=10000,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify
        achieved = acc.get_advantage_for_dpsgd(
            noise_multiplier=noise,
            num_steps=1000,
            sampling_method="fixed_batch",
            batch_size=32,
            dataset_size=10000,
        )

        assert abs(achieved - 0.1) < 0.01

    def test_truncated_poisson_sampling(self):
        """Test advantage calibration with truncated Poisson sampling."""
        noise = acc.find_noise_multiplier_for_advantage(
            advantage=0.1,
            num_steps=1000,
            sampling_method="truncated_poisson",
            sample_rate=0.0032,
            truncated_batch_size=32,
            dataset_size=10000,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify
        achieved = acc.get_advantage_for_dpsgd(
            noise_multiplier=noise,
            num_steps=1000,
            sampling_method="truncated_poisson",
            sample_rate=0.0032,
            truncated_batch_size=32,
            dataset_size=10000,
        )

        assert abs(achieved - 0.1) < 0.01


@pytest.mark.slow()
@pytest.mark.skip(reason="slow test - disabled for now")
class TestErrorRatesCalibration:
    """Test error rates calibration (re-exported from riskcal)."""

    def test_basic_calibration(self):
        """Test basic error rates calibration."""
        noise = acc.find_noise_multiplier_for_err_rates(
            alpha=1e-4,
            beta=0.8,
            sample_rate=16 / 2048,
            num_steps=1000,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify
        achieved_beta = acc.get_beta_for_dpsgd(
            noise_multiplier=noise,
            sample_rate=16 / 2048,
            num_steps=1000,
            alpha=1e-4,
        )

        assert abs(achieved_beta - 0.8) < 0.01

    def test_matches_opaque_accounting(self):
        """Test that riskcal beta matches opaque.accounting."""
        noise = 1.5
        sample_rate = 0.01
        num_steps = 500
        alpha = 0.05

        # Get from riskcal
        beta_riskcal = acc.get_beta_for_dpsgd(
            noise_multiplier=noise,
            sample_rate=sample_rate,
            num_steps=num_steps,
            alpha=alpha,
        )

        # Get from opaque.accounting
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=noise, sample_rate=sample_rate, count=num_steps
        )
        beta_opaque = acc.get_beta(state, alpha=alpha)

        # Should match (within tolerance)
        assert abs(beta_riskcal - beta_opaque) < 0.01

    def test_fixed_batch_sampling(self):
        """Test error rates calibration with fixed-batch sampling."""
        noise = acc.find_noise_multiplier_for_err_rates(
            alpha=1e-4,
            beta=0.8,
            num_steps=1000,
            sampling_method="fixed_batch",
            batch_size=16,
            dataset_size=2048,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify
        achieved_beta = acc.get_beta_for_dpsgd(
            noise_multiplier=noise,
            num_steps=1000,
            alpha=1e-4,
            sampling_method="fixed_batch",
            batch_size=16,
            dataset_size=2048,
        )

        assert abs(achieved_beta - 0.8) < 0.01

    def test_truncated_poisson_sampling(self):
        """Test error rates calibration with truncated Poisson sampling."""
        noise = acc.find_noise_multiplier_for_err_rates(
            alpha=1e-4,
            beta=0.8,
            num_steps=1000,
            sampling_method="truncated_poisson",
            sample_rate=16 / 2048,
            truncated_batch_size=16,
            dataset_size=2048,
        )

        assert isinstance(noise, float)
        assert noise > 0

        # Verify
        achieved_beta = acc.get_beta_for_dpsgd(
            noise_multiplier=noise,
            num_steps=1000,
            alpha=1e-4,
            sampling_method="truncated_poisson",
            sample_rate=16 / 2048,
            truncated_batch_size=16,
            dataset_size=2048,
        )

        assert abs(achieved_beta - 0.8) < 0.01


class TestParameterValidation:
    """Test parameter validation for different sampling methods."""

    def test_poisson_missing_sample_rate(self):
        """Test that Poisson sampling requires sample_rate."""
        with pytest.raises(ValueError, match="sample_rate required"):
            acc.find_noise_multiplier_for_epsilon_delta(
                epsilon=3.0,
                delta=1e-5,
                num_steps=1000,
                sampling_method="poisson",
                # missing sample_rate
            )

    def test_fixed_batch_missing_parameters(self):
        """Test that fixed_batch requires batch_size and dataset_size."""
        with pytest.raises(ValueError, match="batch_size and dataset_size required"):
            acc.find_noise_multiplier_for_epsilon_delta(
                epsilon=3.0,
                delta=1e-5,
                num_steps=1000,
                sampling_method="fixed_batch",
                # missing batch_size and dataset_size
            )

    def test_truncated_poisson_missing_parameters(self):
        """Test that truncated_poisson requires all three parameters."""
        with pytest.raises(ValueError, match="sample_rate, dataset_size, and truncated_batch_size"):
            acc.find_noise_multiplier_for_epsilon_delta(
                epsilon=3.0,
                delta=1e-5,
                num_steps=1000,
                sampling_method="truncated_poisson",
                sample_rate=0.01,
                # missing truncated_batch_size and dataset_size
            )

    def test_advantage_poisson_missing_sample_rate(self):
        """Test advantage calibration validates Poisson parameters."""
        with pytest.raises(ValueError, match="sample_rate required"):
            acc.find_noise_multiplier_for_advantage(
                advantage=0.1,
                num_steps=1000,
                sampling_method="poisson",
                # missing sample_rate
            )

    def test_advantage_fixed_batch_missing_parameters(self):
        """Test advantage calibration validates fixed_batch parameters."""
        with pytest.raises(ValueError, match="batch_size and dataset_size required"):
            acc.find_noise_multiplier_for_advantage(
                advantage=0.1,
                num_steps=1000,
                sampling_method="fixed_batch",
                # missing parameters
            )

    def test_advantage_truncated_poisson_missing_parameters(self):
        """Test advantage calibration validates truncated_poisson parameters."""
        with pytest.raises(ValueError, match="sample_rate, dataset_size, and truncated_batch_size"):
            acc.find_noise_multiplier_for_advantage(
                advantage=0.1,
                num_steps=1000,
                sampling_method="truncated_poisson",
                sample_rate=0.01,
                # missing truncated_batch_size and dataset_size
            )

    def test_err_rates_poisson_missing_sample_rate(self):
        """Test error rates calibration validates Poisson parameters."""
        with pytest.raises(ValueError, match="sample_rate required"):
            acc.find_noise_multiplier_for_err_rates(
                alpha=0.01,
                beta=0.8,
                num_steps=1000,
                sampling_method="poisson",
                # missing sample_rate
            )

    def test_err_rates_fixed_batch_missing_parameters(self):
        """Test error rates calibration validates fixed_batch parameters."""
        with pytest.raises(ValueError, match="batch_size and dataset_size required"):
            acc.find_noise_multiplier_for_err_rates(
                alpha=0.01,
                beta=0.8,
                num_steps=1000,
                sampling_method="fixed_batch",
                # missing parameters
            )

    def test_err_rates_truncated_poisson_missing_parameters(self):
        """Test error rates calibration validates truncated_poisson parameters."""
        with pytest.raises(ValueError, match="sample_rate, dataset_size, and truncated_batch_size"):
            acc.find_noise_multiplier_for_err_rates(
                alpha=0.01,
                beta=0.8,
                num_steps=1000,
                sampling_method="truncated_poisson",
                sample_rate=0.01,
                # missing truncated_batch_size and dataset_size
            )


class TestIntegration:
    """Test that all methods integrate with opaque.accounting."""

    def test_all_metrics_consistent(self):
        """Test that different calibration methods give consistent results."""
        sample_rate = 0.01
        num_steps = 500

        # Calibrate for epsilon/delta
        noise = acc.find_noise_multiplier_for_epsilon_delta(
            epsilon=5.0,
            delta=1e-5,
            sample_rate=sample_rate,
            num_steps=num_steps,
        )

        # Query all metrics using helper functions
        eps_check = acc.get_epsilon_for_dpsgd(
            noise_multiplier=noise,
            num_steps=num_steps,
            delta=1e-5,
            sampling_method="poisson",
            sample_rate=sample_rate,
        )
        adv = acc.get_advantage_for_dpsgd(
            noise_multiplier=noise,
            num_steps=num_steps,
            sampling_method="poisson",
            sample_rate=sample_rate,
        )
        beta = acc.get_beta_for_dpsgd(
            noise_multiplier=noise,
            num_steps=num_steps,
            alpha=0.01,
            sampling_method="poisson",
            sample_rate=sample_rate,
        )

        # Also verify with opaque.accounting composition
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=noise, sample_rate=sample_rate, count=num_steps
        )

        eps_state = acc.get_epsilon(state, delta=1e-5)
        adv_state = acc.get_advantage(state)
        beta_state = acc.get_beta(state, alpha=0.01)

        # All should match
        assert abs(eps_check - 5.0) < 0.1
        assert abs(eps_check - eps_state) < 0.01
        assert abs(adv - adv_state) < 0.01
        assert abs(beta - beta_state) < 0.01

    def test_different_sampling_rates(self):
        """Test calibration works with different sampling rates."""
        # Small sample rate
        noise_small = acc.find_noise_multiplier_for_epsilon_delta(
            epsilon=3.0,
            delta=1e-5,
            sample_rate=0.001,
            num_steps=1000,
        )

        # Large sample rate
        noise_large = acc.find_noise_multiplier_for_epsilon_delta(
            epsilon=3.0,
            delta=1e-5,
            sample_rate=0.1,
            num_steps=1000,
        )

        # Higher sampling rate should need more noise
        assert noise_large > noise_small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
