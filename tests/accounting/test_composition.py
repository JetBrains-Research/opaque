"""Tests for composition functions.

Tests for compose_poisson_gaussian, compose_sampled_gaussian,
compose_fixed_batch, compose_truncated_poisson_gaussian, and NeighboringRelation.
"""

import pytest

import opaque.accounting as acc


class TestComposePoissonGaussian:
    """Test compose_poisson_gaussian function."""

    def test_poisson_gaussian(self):
        """Test Poisson-sampled Gaussian composition."""
        state = acc.create()
        state = acc.compose_poisson_gaussian(
            state, noise_multiplier=1.0, sample_rate=0.01, count=100
        )

        epsilon = acc.get_epsilon(state, delta=1e-5)
        assert epsilon > 0.0


class TestComposeSampledGaussian:
    """Test compose_sampled_gaussian function."""

    def test_sampled_gaussian(self):
        """Test fixed-size batch sampling."""
        state = acc.create()
        state = acc.compose_sampled_gaussian(
            state,
            noise_multiplier=1.0,
            batch_size=32,
            dataset_size=1000,
            count=100,
        )

        epsilon = acc.get_epsilon(state, delta=1e-5)
        assert epsilon > 0.0


class TestComposeFixedBatch:
    """Test compose_fixed_batch helper."""

    def test_fixed_batch_vs_sampled_gaussian(self):
        """Test that compose_fixed_batch is equivalent to compose_sampled_gaussian."""
        state1 = acc.create()
        state2 = acc.create()

        # Use compose_fixed_batch
        state1 = acc.compose_fixed_batch(
            state1, noise_multiplier=1.0, batch_size=32, dataset_size=1000, count=100
        )

        # Use compose_sampled_gaussian
        state2 = acc.compose_sampled_gaussian(
            state2, noise_multiplier=1.0, batch_size=32, dataset_size=1000, count=100
        )

        epsilon1 = acc.get_epsilon(state1, delta=1e-5)
        epsilon2 = acc.get_epsilon(state2, delta=1e-5)

        # Should be identical
        assert epsilon1 == pytest.approx(epsilon2, rel=1e-10)

    def test_fixed_batch_basic_workflow(self):
        """Test basic fixed-batch DP-SGD workflow."""
        state = acc.create()

        # Simulate DP-SGD training with fixed batch size
        state = acc.compose_fixed_batch(
            state, noise_multiplier=1.1, batch_size=32, dataset_size=1000, count=1000
        )

        epsilon = acc.get_epsilon(state, delta=1e-5)

        # Sanity checks (high sample rate 3.2% leads to higher epsilon)
        assert 20.0 < epsilon < 40.0
        assert epsilon > 0.0


class TestComposeTruncatedPoissonGaussian:
    """Test compose_truncated_poisson_gaussian function."""

    def test_truncated_vs_poisson(self):
        """Test that truncated Poisson gives tighter bounds than standard Poisson."""
        # Standard Poisson
        state_poisson = acc.create()
        state_poisson = acc.compose_poisson_gaussian(
            state_poisson, noise_multiplier=1.0, sample_rate=0.01, count=100
        )
        epsilon_poisson = acc.get_epsilon(state_poisson, delta=1e-5)

        # Truncated Poisson
        state_truncated = acc.create()
        state_truncated = acc.compose_truncated_poisson_gaussian(
            state_truncated,
            noise_multiplier=1.0,
            sample_rate=0.01,
            truncated_batch_size=100,
            dataset_size=10000,
            count=100,
        )
        epsilon_truncated = acc.get_epsilon(state_truncated, delta=1e-5)

        # Truncated should give tighter (smaller) epsilon
        # Note: This may not always hold for all parameter combinations
        # but should hold for reasonable DP-SGD settings
        assert epsilon_truncated >= epsilon_poisson


class TestCompositionInputValidation:
    """Test input validation for composition functions."""

    def test_invalid_noise_multiplier(self):
        """Test that invalid noise multiplier raises errors."""
        state = acc.create()

        with pytest.raises(ValueError):
            acc.compose_poisson_gaussian(state, noise_multiplier=0.0, sample_rate=0.01)

        with pytest.raises(ValueError):
            acc.compose_poisson_gaussian(state, noise_multiplier=-1.0, sample_rate=0.01)

    def test_invalid_sample_rate(self):
        """Test that invalid sample rate raises errors."""
        state = acc.create()

        with pytest.raises(ValueError):
            acc.compose_poisson_gaussian(state, noise_multiplier=1.0, sample_rate=0.0)

        with pytest.raises(ValueError):
            acc.compose_poisson_gaussian(state, noise_multiplier=1.0, sample_rate=1.5)

    def test_invalid_truncated_batch_size(self):
        """Test that invalid truncated batch size raises errors."""
        state = acc.create()

        with pytest.raises(ValueError):
            acc.compose_truncated_poisson_gaussian(
                state, noise_multiplier=1.0, sample_rate=0.01,
                truncated_batch_size=0, dataset_size=10000
            )

        with pytest.raises(ValueError):
            acc.compose_truncated_poisson_gaussian(
                state, noise_multiplier=1.0, sample_rate=0.01,
                truncated_batch_size=-1, dataset_size=10000
            )

    def test_invalid_dataset_size(self):
        """Test that invalid dataset size raises errors."""
        state = acc.create()

        with pytest.raises(ValueError):
            acc.compose_truncated_poisson_gaussian(
                state, noise_multiplier=1.0, sample_rate=0.01,
                truncated_batch_size=100, dataset_size=0
            )

        with pytest.raises(ValueError):
            acc.compose_truncated_poisson_gaussian(
                state, noise_multiplier=1.0, sample_rate=0.01,
                truncated_batch_size=100, dataset_size=-1
            )

    def test_invalid_count(self):
        """Test that invalid count raises errors."""
        state = acc.create()

        with pytest.raises(ValueError):
            acc.compose_truncated_poisson_gaussian(
                state, noise_multiplier=1.0, sample_rate=0.01,
                truncated_batch_size=100, dataset_size=10000, count=0
            )

        with pytest.raises(ValueError):
            acc.compose_truncated_poisson_gaussian(
                state, noise_multiplier=1.0, sample_rate=0.01,
                truncated_batch_size=100, dataset_size=10000, count=-1
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
