"""JAX validation tests for privacy accounting.

These tests compare Opaque's accounting implementation against JAX-Privacy
to ensure numerical equivalence.
"""

import pytest

# Opaque imports
import opaque.accounting as acc
# JAX-Privacy imports
from jax_privacy.accounting import (
    DpParams,
    DpsgdTrainingAccountant,
    PldAccountantConfig,
)


@pytest.mark.jax_validation
class TestAccountantValidation:
    """Validate accountants against JAX-Privacy."""

    def test_pld_accountant_poisson_sampling(self):
        """Test PLD accountant with Poisson sampling matches JAX-Privacy."""
        # Opaque - using functional API
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.1, sample_rate=0.01, count=100
        )
        opaque_eps = acc.get_epsilon(state, delta=1e-5)

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
        # Opaque - using functional API
        state = acc.create()
        state = acc.compose_truncated_poisson_gaussian(
            state,
            noise_multiplier=1.1,
            sample_rate=0.01,
            truncated_batch_size=100,
            dataset_size=10000,
            count=100,
        )
        opaque_eps = acc.get_epsilon(state, delta=1e-5)

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
        # Opaque - fixed batch (sample_rate=0.01 → batch=100 for dataset=10000)
        state_fixed = acc.create()
        state_fixed = acc.compose_fixed_batch(
            state_fixed,
            noise_multiplier=1.0,
            batch_size=100,
            dataset_size=10000,
            count=100,
        )
        opaque_fixed_eps = acc.get_epsilon(state_fixed, delta=1e-5)

        # Opaque - Poisson with half noise
        state_poisson = acc.create()
        state_poisson = acc.compose_poisson_gaussian(
            state_poisson, noise_multiplier=0.5, sample_rate=0.01, count=100
        )
        opaque_poisson_eps = acc.get_epsilon(state_poisson, delta=1e-5)

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
class TestPrivacyProperties:
    """Test privacy properties hold in both implementations."""

    def test_more_noise_less_epsilon(self):
        """Test that more noise gives lower epsilon in both implementations."""
        # Opaque - low noise
        state_low = acc.create()
        state_low = acc.compose_poisson_gaussian(
            state_low, noise_multiplier=0.5, sample_rate=0.01, count=100
        )
        opaque_eps_low = acc.get_epsilon(state_low, delta=1e-5)

        # Opaque - high noise
        state_high = acc.create()
        state_high = acc.compose_poisson_gaussian(
            state_high, noise_multiplier=2.0, sample_rate=0.01, count=100
        )
        opaque_eps_high = acc.get_epsilon(state_high, delta=1e-5)

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
        # Opaque - compose 50 steps
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=50
        )
        eps_50 = acc.get_epsilon(state, delta=1e-5)

        # Compose 50 more steps (total 100)
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=50
        )
        eps_100 = acc.get_epsilon(state, delta=1e-5)

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
