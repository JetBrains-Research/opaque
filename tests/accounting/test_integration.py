"""Integration tests for functional privacy accounting.

These tests validate complete end-to-end workflows, ensuring the accounting
system works correctly for realistic DP-SGD scenarios.
"""

import pytest

import opaque.accounting as acc


class TestDPSGDWorkflows:
    """Test complete DP-SGD workflows."""

    def test_basic_dp_sgd_workflow(self):
        """Test realistic DP-SGD workflow with Poisson sampling."""
        # Parameters
        noise_multiplier = 1.1
        sample_rate = 0.01
        num_steps = 1000
        target_delta = 1e-5

        # Run accounting
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            count=num_steps,
        )

        # Query epsilon
        epsilon = acc.get_epsilon(state, delta=target_delta)

        # Sanity checks
        assert 0.5 < epsilon < 5.0  # Reasonable range for these parameters
        assert epsilon > 0.0

    def test_caching_and_incremental_composition(self):
        """Test that state can be cached and composed incrementally."""
        # Initial training: 500 steps
        state_500 = acc.create()
        state_500 = acc.compose_poisson_gaussian(
            state_500, noise_multiplier=1.0, sample_rate=0.01, count=500
        )
        epsilon_500 = acc.get_epsilon(state_500, delta=1e-5)

        # Continue training: 500 more steps
        state_1000 = acc.compose_poisson_gaussian(
            state_500, noise_multiplier=1.0, sample_rate=0.01, count=500
        )
        epsilon_1000 = acc.get_epsilon(state_1000, delta=1e-5)

        # Should match computing 1000 steps from scratch
        state_1000_direct = acc.create()
        state_1000_direct = acc.compose_poisson_gaussian(
            state_1000_direct, noise_multiplier=1.0, sample_rate=0.01, count=1000
        )
        epsilon_1000_direct = acc.get_epsilon(state_1000_direct, delta=1e-5)

        assert epsilon_1000 == pytest.approx(epsilon_1000_direct, rel=1e-6)
        assert epsilon_1000 > epsilon_500  # Privacy degrades with more steps

    def test_full_workflow_with_alpha_beta(self):
        """Test complete workflow using both epsilon/delta and alpha/beta queries."""
        # Setup
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=100
        )

        # Traditional (ε, δ) query
        epsilon = acc.get_epsilon(state, delta=1e-5)
        assert epsilon > 0.0

        # Modern alpha/beta query
        beta = acc.get_beta(state, alpha=0.01)
        assert 0.0 <= beta <= 1.0

        # Advantage query
        advantage = acc.get_advantage(state)
        assert 0.0 <= advantage <= 1.0


class TestStateImmutability:
    """Test that state behaves functionally (returns new objects)."""

    def test_composition_returns_new_state(self):
        """Test that composition returns a new state object."""
        state1 = acc.create()
        state2 = acc.compose_poisson_gaussian(
            state1, noise_multiplier=1.0, sample_rate=0.01
        )

        # Should be different objects
        assert state1 is not state2

        # Original state should be unchanged
        epsilon1 = acc.get_epsilon(state1, delta=1e-5)
        assert epsilon1 == pytest.approx(0.0, abs=1e-10)

    def test_state_can_be_reused(self):
        """Test that the same state can be used in multiple compositions."""
        base_state = acc.create()
        base_state = acc.compose_poisson_gaussian(
            base_state, noise_multiplier=1.0, sample_rate=0.01, count=10
        )

        # Branch 1: Add 10 more steps
        branch1 = acc.compose_poisson_gaussian(
            base_state, noise_multiplier=1.0, sample_rate=0.01, count=10
        )
        epsilon1 = acc.get_epsilon(branch1, delta=1e-5)

        # Branch 2: Add 20 more steps
        branch2 = acc.compose_poisson_gaussian(
            base_state, noise_multiplier=1.0, sample_rate=0.01, count=20
        )
        epsilon2 = acc.get_epsilon(branch2, delta=1e-5)

        # Branch 2 should have higher epsilon
        assert epsilon2 > epsilon1

        # Original base_state should be unchanged
        epsilon_base = acc.get_epsilon(base_state, delta=1e-5)
        assert epsilon_base < epsilon1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
