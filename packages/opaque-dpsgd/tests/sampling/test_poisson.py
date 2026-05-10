"""Tests for the unified PoissonSubsampler (plain + truncated)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from opaque.random import key
from opaque.dpsgd.sampling import PoissonSubsampler


class TestPoissonSubsampler:
    """Tests for plain PoissonSubsampler."""

    def test_init_basic(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(dataset, sample_rate=0.1, n_steps=5, key=key(0))

        assert sampler.sample_rate == 0.1
        assert sampler.n_steps == 5
        assert sampler.truncated_batch_size is None
        assert len(sampler) == 5

    def test_init_invalid_sample_rate(self):
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="sample_rate must be in"):
            PoissonSubsampler(dataset, sample_rate=0.0, key=key(0))

        with pytest.raises(ValueError, match="sample_rate must be in"):
            PoissonSubsampler(dataset, sample_rate=1.5, key=key(0))

    def test_init_invalid_n_steps(self):
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="n_steps must be"):
            PoissonSubsampler(dataset, sample_rate=0.1, n_steps=0, key=key(0))

    def test_iteration_produces_variable_batches(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(dataset, sample_rate=0.1, n_steps=10, key=key(42))

        batch_sizes = [len(batch) for batch in sampler]

        assert len(set(batch_sizes)) > 1, "Batch sizes should vary"

    def test_statistical_properties_mean(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(
            dataset,
            sample_rate=0.1,
            n_steps=100,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]
        expected_mean = 1000 * 0.1
        actual_mean = np.mean(batch_sizes)

        assert abs(actual_mean - expected_mean) < 10

    def test_statistical_properties_variance(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(
            dataset,
            sample_rate=0.1,
            n_steps=500,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]
        expected_var = 1000 * 0.1 * (1 - 0.1)
        actual_var = np.var(batch_sizes)

        assert abs(actual_var - expected_var) < 20

    def test_no_duplicate_indices_within_batch(self):
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSubsampler(dataset, sample_rate=0.5, n_steps=20, key=key(42))

        for batch_indices in sampler:
            assert len(batch_indices) == len(set(batch_indices))
            assert all(0 <= idx < 100 for idx in batch_indices)

    def test_indices_in_valid_range(self):
        dataset = TensorDataset(torch.randn(500, 10))
        sampler = PoissonSubsampler(dataset, sample_rate=0.1, n_steps=10, key=key(42))

        for batch_indices in sampler:
            assert all(0 <= idx < 500 for idx in batch_indices)

    def test_expected_batch_size_property(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(dataset, sample_rate=0.1, key=key(0))

        assert sampler.expected_batch_size == 100.0

    def test_batch_size_variance_property(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(dataset, sample_rate=0.1, key=key(0))

        expected_var = 1000 * 0.1 * 0.9
        assert sampler.batch_size_variance == expected_var

    def test_reproducibility_with_generator(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = PoissonSubsampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))
        batches1 = list(sampler1)

        sampler2 = PoissonSubsampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))
        batches2 = list(sampler2)

        for b1, b2 in zip(batches1, batches2, strict=True):
            assert b1 == b2

    def test_integration_with_dataloader(self):
        dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 5))
        sampler = PoissonSubsampler(dataset, sample_rate=0.1, n_steps=5, key=key(42))

        loader = DataLoader(dataset, batch_sampler=sampler)

        batch_sizes = []
        for X, y in loader:
            assert X.shape[1] == 10
            assert y.shape[1] == 5
            batch_sizes.append(X.shape[0])

        assert len(batch_sizes) == 5
        assert len(set(batch_sizes)) > 1


class TestPoissonSubsamplerTruncated:
    """Tests for PoissonSubsampler with truncated_batch_size set."""

    def test_init_basic(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(
            dataset,
            sample_rate=0.1,
            n_steps=5,
            truncated_batch_size=50,
            key=key(0),
        )

        assert sampler.sample_rate == 0.1
        assert sampler.truncated_batch_size == 50
        assert sampler.n_steps == 5

    def test_init_invalid_truncated_batch_size(self):
        dataset = TensorDataset(torch.randn(100, 10))

        with pytest.raises(ValueError, match="truncated_batch_size must be"):
            PoissonSubsampler(dataset, sample_rate=0.1, truncated_batch_size=0, key=key(0))

    def test_truncation_enforced(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=100,
            n_steps=50,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]
        assert max(batch_sizes) <= 100
        assert max(batch_sizes) >= 90

    def test_same_as_plain_when_no_truncation(self):
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler_truncated = PoissonSubsampler(
            dataset,
            sample_rate=0.05,
            truncated_batch_size=10000,
            n_steps=20,
            key=key(42),
        )

        sampler_regular = PoissonSubsampler(
            dataset,
            sample_rate=0.05,
            n_steps=20,
            key=key(42),
        )

        batches_truncated = [len(b) for b in sampler_truncated]
        batches_regular = [len(b) for b in sampler_regular]

        assert batches_truncated == batches_regular

    def test_no_duplicate_indices_after_truncation(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        sampler = PoissonSubsampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=100,
            n_steps=20,
            key=key(42),
        )

        for batch_indices in sampler:
            assert len(batch_indices) == len(set(batch_indices))
            assert all(0 <= idx < 1000 for idx in batch_indices)

    def test_integration_with_dataloader(self):
        dataset = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 5))
        sampler = PoissonSubsampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=100,
            n_steps=5,
            key=key(42),
        )

        loader = DataLoader(dataset, batch_sampler=sampler)

        batch_sizes = []
        for X, y in loader:
            batch_sizes.append(X.shape[0])
            assert X.shape[0] <= 100
            assert y.shape[0] <= 100

        assert len(batch_sizes) == 5


class TestEdgeCases:
    """Test edge cases."""

    def test_sample_rate_one(self):
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSubsampler(dataset, sample_rate=1.0, n_steps=5, key=key(42))

        batch_sizes = [len(batch) for batch in sampler]
        assert all(90 <= size <= 100 for size in batch_sizes)

    def test_very_small_sample_rate(self):
        dataset = TensorDataset(torch.randn(10000, 10))
        sampler = PoissonSubsampler(
            dataset,
            sample_rate=0.001,
            n_steps=100,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]
        mean_size = np.mean(batch_sizes)
        assert 5 <= mean_size <= 15

    def test_truncated_with_max_batch_size_one(self):
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSubsampler(
            dataset,
            sample_rate=0.5,
            truncated_batch_size=1,
            n_steps=20,
            key=key(42),
        )

        batch_sizes = [len(batch) for batch in sampler]
        assert all(size <= 1 for size in batch_sizes)
