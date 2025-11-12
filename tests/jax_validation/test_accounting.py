"""JAX validation tests for privacy accounting.

These tests compare Opaque's accounting implementation against JAX-Privacy
to ensure numerical equivalence.
"""

import pytest

# JAX-Privacy imports
from jax_privacy.accounting import (
    DpParams,
    DpsgdTrainingAccountant,
    PldAccountantConfig,
    RdpAccountantConfig,
    SamplingMethod,
)
from jax_privacy.accounting import (
    calibrate_batch_size as jax_calibrate_batch_size,
)
from jax_privacy.accounting import (
    calibrate_noise_multiplier as jax_calibrate_noise,
)
from jax_privacy.accounting import (
    calibrate_num_updates as jax_calibrate_steps,
)
# Opaque imports
from opaque.accounting import (
    PLDAccountant,
    RDPAccountant,
    calibrate_batch_size,
    calibrate_noise_multiplier,
    calibrate_steps,
)


@pytest.mark.jax_validation
class TestAccountantValidation:
    """Validate accountants against JAX-Privacy."""

    def test_rdp_accountant_poisson_sampling(self):
        """Test RDP accountant with Poisson sampling matches JAX-Privacy."""
        # Opaque
        opaque_acc = RDPAccountant()
        opaque_acc.step_poisson(noise_multiplier=1.1, sample_rate=0.01, num_steps=100)
        opaque_eps = opaque_acc.get_epsilon(target_delta=1e-5)

        # JAX-Privacy
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=RdpAccountantConfig())
        dp_params = DpParams(
            noise_multipliers=1.1,
            batch_size=100,  # batch_size with Poisson = sample_rate * num_samples
            num_samples=10000,
            delta=1e-5,
            sampling_method=SamplingMethod.POISSON,
        )
        jax_eps = jax_acc.compute_epsilon(num_updates=100, dp_params=dp_params)

        # Should match within reasonable tolerance
        assert abs(opaque_eps - jax_eps) < 0.01, f"Opaque: {opaque_eps}, JAX: {jax_eps}"

    def test_rdp_accountant_multiple_steps(self):
        """Test RDP accountant with multiple step calls."""
        # Opaque
        opaque_acc = RDPAccountant()
        opaque_acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=50)
        opaque_acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=50)
        opaque_eps = opaque_acc.get_epsilon(target_delta=1e-5)

        # JAX-Privacy
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=RdpAccountantConfig())
        dp_params = DpParams(
            noise_multipliers=1.0,
            batch_size=100,
            num_samples=10000,
            delta=1e-5,
            sampling_method=SamplingMethod.POISSON,
        )
        jax_eps = jax_acc.compute_epsilon(num_updates=100, dp_params=dp_params)

        assert abs(opaque_eps - jax_eps) < 0.01

    def test_pld_accountant_poisson_sampling(self):
        """Test PLD accountant with Poisson sampling matches JAX-Privacy."""
        # Opaque
        opaque_acc = PLDAccountant()
        opaque_acc.step_poisson(noise_multiplier=1.1, sample_rate=0.01, num_steps=100)
        opaque_eps = opaque_acc.get_epsilon(target_delta=1e-5)

        # JAX-Privacy
        jax_acc_config = PldAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        dp_params = DpParams(
            noise_multipliers=1.1,
            batch_size=100,
            num_samples=10000,
            delta=1e-5,
        )
        jax_eps = jax_acc.compute_epsilon(num_updates=100, dp_params=dp_params)

        # PLD should match very closely
        assert abs(opaque_eps - jax_eps) < 0.01, f"Opaque: {opaque_eps}, JAX: {jax_eps}"

    @pytest.mark.slow
    def test_pld_accountant_truncated_poisson(self):
        """Test PLD accountant with truncated Poisson sampling."""
        # Opaque
        opaque_acc = PLDAccountant()
        opaque_acc.step_truncated_poisson(
            noise_multiplier=1.1,
            sample_rate=0.01,
            truncated_batch_size=100,
            dataset_size=10000,
            num_steps=100,
        )
        opaque_eps = opaque_acc.get_epsilon(target_delta=1e-5)

        # JAX-Privacy
        jax_acc_config = PldAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        dp_params = DpParams(
            noise_multipliers=1.1,
            batch_size=100,
            num_samples=10000,
            delta=1e-5,
            truncated_batch_size=100,
        )
        jax_eps = jax_acc.compute_epsilon(num_updates=100, dp_params=dp_params)

        # Should match very closely
        assert abs(opaque_eps - jax_eps) < 0.01, f"Opaque: {opaque_eps}, JAX: {jax_eps}"

    def test_fixed_batch_sampling_equivalence(self):
        """Test that fixed batch = Poisson with half noise in both implementations."""
        # Opaque - fixed batch
        opaque_fixed = PLDAccountant()
        opaque_fixed.step_fixed_batch(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)
        opaque_fixed_eps = opaque_fixed.get_epsilon(target_delta=1e-5)

        # Opaque - Poisson with half noise
        opaque_poisson = PLDAccountant()
        opaque_poisson.step_poisson(noise_multiplier=0.5, sample_rate=0.01, num_steps=100)
        opaque_poisson_eps = opaque_poisson.get_epsilon(target_delta=1e-5)

        # Should be identical
        assert abs(opaque_fixed_eps - opaque_poisson_eps) < 1e-6

        # JAX-Privacy - Poisson with half noise (JAX doesn't have step_fixed_batch)
        jax_acc_config = PldAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        dp_params = DpParams(
            noise_multipliers=0.5,
            batch_size=100,
            num_samples=10000,
            delta=1e-5,
        )
        jax_eps = jax_acc.compute_epsilon(num_updates=100, dp_params=dp_params)

        # Opaque fixed batch should match JAX Poisson with half noise
        assert abs(opaque_fixed_eps - jax_eps) < 0.01


@pytest.mark.jax_validation
class TestCalibrationValidation:
    """Validate calibration functions against JAX-Privacy."""

    def test_calibrate_noise_rdp(self):
        """Test noise calibration with RDP matches JAX-Privacy."""
        # Opaque
        opaque_noise = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
            accountant_type="rdp",
        )

        # JAX-Privacy
        jax_acc_config = RdpAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        jax_noise = jax_calibrate_noise(
            target_epsilon=3.0,
            accountant=jax_acc,
            batch_sizes=100,
            num_samples=10000,
            target_delta=1e-5,
            num_updates=1000,
        )

        # Should be very close
        assert abs(opaque_noise - jax_noise) < 0.1, f"Opaque: {opaque_noise}, JAX: {jax_noise}"

    @pytest.mark.slow
    def test_calibrate_noise_pld(self):
        """Test noise calibration with PLD matches JAX-Privacy."""
        # Opaque
        opaque_noise = calibrate_noise_multiplier(
            target_epsilon=3.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
            accountant_type="pld",
        )

        # JAX-Privacy
        jax_acc_config = PldAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        jax_noise = jax_calibrate_noise(
            target_epsilon=3.0,
            accountant=jax_acc,
            batch_sizes=100,
            num_samples=10000,
            target_delta=1e-5,
            num_updates=1000,
        )

        # Should be very close
        assert abs(opaque_noise - jax_noise) < 0.1, f"Opaque: {opaque_noise}, JAX: {jax_noise}"

    def test_calibrate_steps_rdp(self):
        """Test step calibration with RDP matches JAX-Privacy."""
        # Opaque
        opaque_steps = calibrate_steps(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=1.1,
            sample_rate=0.01,
            accountant_type="rdp",
        )

        # JAX-Privacy
        jax_acc_config = RdpAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        jax_steps = jax_calibrate_steps(
            target_epsilon=3.0,
            accountant=jax_acc,
            noise_multipliers=1.1,
            batch_sizes=100,
            num_samples=10000,
            target_delta=1e-5,
        )

        # Should be very close (within a few steps)
        assert abs(opaque_steps - jax_steps) < 10, f"Opaque: {opaque_steps}, JAX: {jax_steps}"

    @pytest.mark.slow
    def test_calibrate_steps_pld(self):
        """Test step calibration with PLD matches JAX-Privacy."""
        # Opaque
        opaque_steps = calibrate_steps(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=1.1,
            sample_rate=0.01,
            accountant_type="pld",
        )

        # JAX-Privacy
        jax_acc_config = PldAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        jax_steps = jax_calibrate_steps(
            target_epsilon=3.0,
            accountant=jax_acc,
            noise_multipliers=1.1,
            batch_sizes=100,
            num_samples=10000,
            target_delta=1e-5,
        )

        # Should be very close (within a few steps)
        assert abs(opaque_steps - jax_steps) < 10, f"Opaque: {opaque_steps}, JAX: {jax_steps}"

    def test_calibrate_batch_size_rdp(self):
        """Test batch size calibration with RDP matches JAX-Privacy."""
        # Opaque
        opaque_batch = calibrate_batch_size(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=1.1,
            num_steps=1000,
            dataset_size=10000,
            accountant_type="rdp",
        )

        # JAX-Privacy
        jax_acc_config = RdpAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        jax_batch = jax_calibrate_batch_size(
            target_epsilon=3.0,
            accountant=jax_acc,
            noise_multipliers=1.1,
            num_samples=10000,
            target_delta=1e-5,
            num_updates=1000,
        )

        # Should be very close
        assert abs(opaque_batch - jax_batch) < 10, f"Opaque: {opaque_batch}, JAX: {jax_batch}"

    @pytest.mark.slow
    def test_calibrate_batch_size_pld(self):
        """Test batch size calibration with PLD matches JAX-Privacy."""
        # Opaque
        opaque_batch = calibrate_batch_size(
            target_epsilon=3.0,
            target_delta=1e-5,
            noise_multiplier=1.1,
            num_steps=1000,
            dataset_size=10000,
            accountant_type="pld",
        )

        # JAX-Privacy
        jax_acc_config = PldAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        jax_batch = jax_calibrate_batch_size(
            target_epsilon=3.0,
            accountant=jax_acc,
            noise_multipliers=1.1,
            num_samples=10000,
            target_delta=1e-5,
            num_updates=1000,
        )

        # Should be very close
        assert abs(opaque_batch - jax_batch) < 10, f"Opaque: {opaque_batch}, JAX: {jax_batch}"


@pytest.mark.jax_validation
class TestPrivacyProperties:
    """Test privacy properties hold in both implementations."""

    def test_more_noise_less_epsilon(self):
        """Test that more noise gives lower epsilon in both implementations."""
        # Opaque
        opaque_low = PLDAccountant()
        opaque_low.step_poisson(noise_multiplier=0.5, sample_rate=0.01, num_steps=100)

        opaque_high = PLDAccountant()
        opaque_high.step_poisson(noise_multiplier=2.0, sample_rate=0.01, num_steps=100)

        opaque_eps_low = opaque_low.get_epsilon(target_delta=1e-5)
        opaque_eps_high = opaque_high.get_epsilon(target_delta=1e-5)

        # JAX-Privacy
        jax_acc_config = PldAccountantConfig()

        jax_acc_low = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        jax_acc_high = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)

        dp_params_low = DpParams(
            noise_multipliers=0.5, batch_size=100, num_samples=10000, delta=1e-5
        )
        dp_params_high = DpParams(
            noise_multipliers=2.0, batch_size=100, num_samples=10000, delta=1e-5
        )

        jax_eps_low = jax_acc_low.compute_epsilon(num_updates=100, dp_params=dp_params_low)
        jax_eps_high = jax_acc_high.compute_epsilon(num_updates=100, dp_params=dp_params_high)

        # Both should show same property: more noise = lower epsilon
        assert opaque_eps_low > opaque_eps_high
        assert jax_eps_low > jax_eps_high

        # And they should be reasonably close to each other
        assert abs(opaque_eps_low - jax_eps_low) < 0.1
        assert abs(opaque_eps_high - jax_eps_high) < 0.1

    def test_composition_increases_epsilon(self):
        """Test that composition increases epsilon in both implementations."""
        # Opaque
        opaque_acc = PLDAccountant()
        opaque_acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=50)
        eps_50 = opaque_acc.get_epsilon(target_delta=1e-5)

        opaque_acc.step_poisson(noise_multiplier=1.0, sample_rate=0.01, num_steps=50)
        eps_100 = opaque_acc.get_epsilon(target_delta=1e-5)

        # JAX-Privacy
        jax_acc_config = PldAccountantConfig()
        jax_acc = DpsgdTrainingAccountant(dp_accountant_config=jax_acc_config)
        dp_params = DpParams(noise_multipliers=1.0, batch_size=100, num_samples=10000, delta=1e-5)

        jax_eps_50 = jax_acc.compute_epsilon(num_updates=50, dp_params=dp_params)
        jax_eps_100 = jax_acc.compute_epsilon(num_updates=100, dp_params=dp_params)

        # Both should show composition increases epsilon
        assert eps_100 > eps_50
        assert jax_eps_100 > jax_eps_50

        # And they should be reasonably close to each other
        assert abs(eps_50 - jax_eps_50) < 0.1
        assert abs(eps_100 - jax_eps_100) < 0.1
