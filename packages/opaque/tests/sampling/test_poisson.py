"""Tests for Poisson samplers."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from opaque.random import key
from opaque.sampling import PoissonSampler, TruncatedPoissonSampler


class TestPoissonSampler:
    """Tests for PoissonSampler."""

    def test_init_basic(self):
        """Test basic initialization."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, num_iterations=5, key=key(0))

        assert sampler.sample_rate == 0.1
        assert sampler.num_iterations == 5
        assert len(sampler) == 5

    def test_init_invalid_sample_rate(self):
        """Test that invalid sample_rate raises error."""
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="sample_rate must be in"):
            PoissonSampler(dataset, sample_rate=0.0, key=key(0))

        with pytest.raises(ValueError, match="sample_rate must be in"):
            PoissonSampler(dataset, sample_rate=1.5, key=key(0))

    def test_init_invalid_num_iterations(self):
        """Test that invalid num_iterations raises error."""
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="num_iterations must be"):
            PoissonSampler(dataset, sample_rate=0.1, num_iterations=0, key=key(0))

    def test_iteration_produces_variable_batches(self):
        """Test that iteration produces variable-sized batches."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, num_iterations=10, key=key(42))

        batch_sizes = [len(batch) for batch in sampler]

        # Should have variable sizes
        assert len(set(batch_sizes)) > 1, "Batch sizes should vary"

    def test_statistical_properties_mean(self):
        """Test that mean batch size matches expectation."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.1,
            num_iterations=100,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]

        # Mean should be approximately n * p
        expected_mean = 1000 * 0.1
        actual_mean = np.mean(batch_sizes)

        assert abs(actual_mean - expected_mean) < 10, (
            f"Mean {actual_mean} not close to {expected_mean}"
        )

    def test_statistical_properties_variance(self):
        """Test that variance matches Poisson distribution."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.1,
            num_iterations=500,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]

        # Variance should be approximately n * p * (1 - p)
        expected_var = 1000 * 0.1 * (1 - 0.1)  # = 90
        actual_var = np.var(batch_sizes)

        # Allow 20% tolerance for variance estimate
        assert abs(actual_var - expected_var) < 20, (
            f"Variance {actual_var} not close to {expected_var}"
        )

    def test_no_duplicate_indices_within_batch(self):
        """Test that each batch has unique indices."""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.5, num_iterations=20, key=key(42))

        for batch_indices in sampler:
            # No duplicates within batch
            assert len(batch_indices) == len(set(batch_indices))

            # All indices valid
            assert all(0 <= idx < 100 for idx in batch_indices)

    def test_indices_in_valid_range(self):
        """Test that indices are within valid range."""
        dataset = TensorDataset(torch.randn(500, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, num_iterations=10, key=key(42))

        for batch_indices in sampler:
            assert all(0 <= idx < 500 for idx in batch_indices)

    def test_expected_batch_size_property(self):
        """Test expected_batch_size property."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, key=key(0))

        assert sampler.expected_batch_size == 100.0

    def test_batch_size_variance_property(self):
        """Test batch_size_variance property."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, key=key(0))

        expected_var = 1000 * 0.1 * 0.9
        assert sampler.batch_size_variance == expected_var

    def test_reproducibility_with_generator(self):
        """Test that results are reproducible with same generator seed."""
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = PoissonSampler(dataset, sample_rate=0.1, num_iterations=5, key=key(42))
        batches1 = list(sampler1)

        sampler2 = PoissonSampler(dataset, sample_rate=0.1, num_iterations=5, key=key(42))
        batches2 = list(sampler2)

        # Same seed should produce same batches
        for b1, b2 in zip(batches1, batches2, strict=True):
            assert b1 == b2

    def test_integration_with_dataloader(self):
        """Test integration with PyTorch DataLoader."""
        dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 5))
        sampler = PoissonSampler(dataset, sample_rate=0.1, num_iterations=5, key=key(42))

        # Use batch_sampler parameter (not sampler)
        loader = DataLoader(dataset, batch_sampler=sampler)

        # Should iterate successfully
        batch_sizes = []
        for X, y in loader:
            assert X.shape[1] == 10
            assert y.shape[1] == 5
            batch_sizes.append(X.shape[0])

        # Should have 5 batches (num_iterations)
        assert len(batch_sizes) == 5

        # Batch sizes should be variable
        assert len(set(batch_sizes)) > 1


class TestTruncatedPoissonSampler:
    """Tests for TruncatedPoissonSampler."""

    def test_init_basic(self):
        """Test basic initialization."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = TruncatedPoissonSampler(
            dataset, sample_rate=0.1, max_batch_size=50, num_iterations=5, key=key(0)
        )

        assert sampler.sample_rate == 0.1
        assert sampler.max_batch_size == 50
        assert sampler.num_iterations == 5

    def test_init_invalid_max_batch_size(self):
        """Test that invalid max_batch_size raises error."""
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="max_batch_size must be"):
            TruncatedPoissonSampler(
                dataset, sample_rate=0.1, max_batch_size=0, key=key(0)
            )

    def test_truncation_enforced(self):
        """Test that batch size never exceeds max_batch_size."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.5,  # High rate to ensure truncation
            max_batch_size=100,
            num_iterations=50,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]

        # No batch should exceed max
        assert max(batch_sizes) <= 100

        # At least some should be capped (with high sample rate)
        assert max(batch_sizes) >= 90  # Should hit cap often

    def test_inherits_from_poisson_sampler(self):
        """Test that TruncatedPoissonSampler inherits from PoissonSampler."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = TruncatedPoissonSampler(
            dataset, sample_rate=0.1, max_batch_size=50, key=key(0)
        )

        assert isinstance(sampler, PoissonSampler)

    def test_same_as_poisson_when_no_truncation(self):
        """Test behavior matches PoissonSampler when max_batch_size is large."""
        dataset = TensorDataset(torch.randn(1000, 10))

        # Very large max_batch_size (won't truncate)
        sampler_truncated = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.05,
            max_batch_size=10000,
            num_iterations=20,
            key=key(42),
        )

        sampler_regular = PoissonSampler(
            dataset,
            sample_rate=0.05,
            num_iterations=20,
            key=key(42),
        )

        batches_truncated = [len(b) for b in sampler_truncated]
        batches_regular = [len(b) for b in sampler_regular]

        # Should be identical when no truncation occurs
        assert batches_truncated == batches_regular

    def test_no_duplicate_indices_after_truncation(self):
        """Test that truncated batches still have unique indices."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.5,
            max_batch_size=100,
            num_iterations=20,
            key=key(42),
        )

        for batch_indices in sampler:
            # No duplicates
            assert len(batch_indices) == len(set(batch_indices))

            # All valid
            assert all(0 <= idx < 1000 for idx in batch_indices)

    def test_truncation_is_uniform_subsample(self):
        """Test that truncation randomly subsamples (uniform)."""
        dataset = TensorDataset(torch.randn(200, 10))
        sampler = TruncatedPoissonSampler(
            dataset,
            sample_rate=1.0,  # Include all examples
            max_batch_size=50,  # Force truncation
            num_iterations=100,
            key=key(42),
        )

        # Count how often each index appears
        index_counts = np.zeros(200)
        for batch_indices in sampler:
            for idx in batch_indices:
                index_counts[idx] += 1

        # All indices should appear roughly equally often
        # (uniform subsampling from full set)
        mean_count = index_counts.mean()
        std_count = index_counts.std()

        # All counts should be within 3 std of mean (rough check)
        assert (np.abs(index_counts - mean_count) < 3 * std_count).all(), (
            "Truncation should uniformly subsample"
        )

    def test_integration_with_dataloader(self):
        """Test integration with PyTorch DataLoader."""
        dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 5))
        sampler = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.5,
            max_batch_size=100,
            num_iterations=5,
            key=key(42),
        )

        loader = DataLoader(dataset, batch_sampler=sampler)

        # Should iterate successfully
        batch_sizes = []
        for X, y in loader:
            batch_sizes.append(X.shape[0])

            # Never exceed max
            assert X.shape[0] <= 100
            assert y.shape[0] <= 100

        # Should have 5 batches
        assert len(batch_sizes) == 5


class TestEdgeCases:
    """Test edge cases for both samplers."""

    def test_sample_rate_one(self):
        """Test with sample_rate=1.0 (include all examples)."""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(dataset, sample_rate=1.0, num_iterations=5, key=key(42))

        batch_sizes = [len(batch) for batch in sampler]

        # All batches should be close to 100
        assert all(90 <= size <= 100 for size in batch_sizes)

    def test_very_small_sample_rate(self):
        """Test with very small sample_rate."""
        dataset = TensorDataset(torch.randn(10000, 10))
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.001,
            num_iterations=100,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]

        # Expected: ~10 examples per batch
        mean_size = np.mean(batch_sizes)
        assert 5 <= mean_size <= 15

    def test_truncated_with_max_batch_size_one(self):
        """Test truncated sampler with max_batch_size=1."""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.5,
            max_batch_size=1,
            num_iterations=20,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]

        # All batches should have 0 or 1 example
        assert all(size <= 1 for size in batch_sizes)
