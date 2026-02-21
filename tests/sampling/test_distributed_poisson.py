"""Tests for distributed Poisson sampling with external sharding (inner composition).

Samplers no longer accept rank/world_size. Instead, the dataset is sharded
externally using local_shard_bounds() + Subset, and per-rank keys are
derived via fold_in(key, rank).
"""

import numpy as np
import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from opaque.random import fold_in, key
from opaque.sampling import PoissonSampler, TruncatedPoissonSampler
from opaque.sampling.distributed import local_shard_bounds


def _make_shard(dataset, rank, world_size):
    """Create a Subset shard for the given rank."""
    start, end = local_shard_bounds(len(dataset), rank=rank, world_size=world_size)
    return Subset(dataset, range(start, end))


class TestLocalShardBounds:
    """Tests for the local_shard_bounds utility."""

    def test_single_device(self):
        """world_size=1 returns full range."""
        assert local_shard_bounds(100) == (0, 100)

    def test_even_split(self):
        """Even dataset divides equally."""
        assert local_shard_bounds(100, rank=0, world_size=4) == (0, 25)
        assert local_shard_bounds(100, rank=1, world_size=4) == (25, 50)
        assert local_shard_bounds(100, rank=2, world_size=4) == (50, 75)
        assert local_shard_bounds(100, rank=3, world_size=4) == (75, 100)

    def test_uneven_split_last_rank_gets_remainder(self):
        """Last rank receives remainder examples."""
        assert local_shard_bounds(103, rank=0, world_size=4) == (0, 25)
        assert local_shard_bounds(103, rank=1, world_size=4) == (25, 50)
        assert local_shard_bounds(103, rank=2, world_size=4) == (50, 75)
        assert local_shard_bounds(103, rank=3, world_size=4) == (75, 103)

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="rank must be in"):
            local_shard_bounds(100, rank=4, world_size=4)

    def test_invalid_world_size_raises(self):
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            local_shard_bounds(100, rank=0, world_size=0)

    def test_negative_dataset_size_raises(self):
        with pytest.raises(ValueError, match="dataset_size must be >= 0"):
            local_shard_bounds(-1, rank=0, world_size=1)


class TestShardedSampling:
    """Tests for sharded sampling via external Subset + fold_in."""

    def test_shards_correct_sizes(self):
        """Each shard has the expected number of examples."""
        dataset = TensorDataset(torch.randn(1000, 10))
        world_size = 4

        for rank in range(world_size):
            shard = _make_shard(dataset, rank, world_size)
            assert len(shard) == 250

    def test_uneven_split_last_rank_larger(self):
        """Last rank gets remainder."""
        dataset = TensorDataset(torch.randn(103, 10))
        world_size = 4

        shard_last = _make_shard(dataset, 3, world_size)
        expected = 103 - 3 * (103 // 4)
        assert len(shard_last) == expected

    def test_shards_produce_valid_local_indices(self):
        """Each shard samples indices in [0, shard_size)."""
        dataset = TensorDataset(torch.randn(1000, 10))
        world_size = 4

        for rank in range(world_size):
            shard = _make_shard(dataset, rank, world_size)
            sampler = PoissonSampler(
                shard, sample_rate=0.5, num_epochs=1, key=fold_in(key(42), rank)
            )
            batch = list(sampler)[0]
            shard_size = len(shard)
            assert all(0 <= idx < shard_size for idx in batch)

    def test_statistical_properties_per_shard(self):
        """Each shard maintains Poisson properties."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sample_rate = 0.1
        world_size = 4

        for rank in range(world_size):
            shard = _make_shard(dataset, rank, world_size)
            sampler = PoissonSampler(
                shard,
                sample_rate=sample_rate,
                num_epochs=100,
                key=fold_in(key(100), rank),
            )
            batch_sizes = [len(b) for b in sampler]
            expected_mean = len(shard) * sample_rate
            actual_mean = np.mean(batch_sizes)
            assert abs(actual_mean - expected_mean) < expected_mean * 0.5 + 3

    def test_different_ranks_independent(self):
        """Different ranks produce independent samples."""
        dataset = TensorDataset(torch.randn(1000, 10))
        world_size = 4

        shard0 = _make_shard(dataset, 0, world_size)
        shard1 = _make_shard(dataset, 1, world_size)

        sizes0 = [
            len(b)
            for b in PoissonSampler(
                shard0, sample_rate=0.5, num_epochs=20, key=fold_in(key(42), 0)
            )
        ]
        sizes1 = [
            len(b)
            for b in PoissonSampler(
                shard1, sample_rate=0.5, num_epochs=20, key=fold_in(key(42), 1)
            )
        ]
        assert sizes0 != sizes1, "Different ranks should sample independently"


class TestSingleDeviceMode:
    """Tests for single-device (default) sampling — no sharding needed."""

    def test_single_device_full_dataset(self):
        """Default: sampler operates on full dataset."""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1, key=key(0))
        batches = list(sampler)
        assert len(batches) == 1

    def test_different_keys_produce_different_batches(self):
        dataset = TensorDataset(torch.randn(1000, 10))
        s0 = PoissonSampler(dataset, sample_rate=0.1, num_epochs=10, key=key(42))
        s1 = PoissonSampler(dataset, sample_rate=0.1, num_epochs=10, key=key(43))
        assert list(s0) != list(s1)

    def test_same_key_reproduces_batches(self):
        dataset = TensorDataset(torch.randn(500, 10))
        s1 = PoissonSampler(dataset, sample_rate=0.2, num_epochs=5, key=key(12345))
        s2 = PoissonSampler(dataset, sample_rate=0.2, num_epochs=5, key=key(12345))
        assert list(s1) == list(s2)


class TestTruncatedPoissonDistributed:
    """Tests for TruncatedPoissonSampler with external sharding."""

    def test_truncated_sharded_respects_max_batch_size(self):
        """TruncatedPoissonSampler respects max_batch_size on each shard."""
        dataset = TensorDataset(torch.randn(1000, 10))
        max_batch_size = 50
        world_size = 4

        for rank in range(world_size):
            shard = _make_shard(dataset, rank, world_size)
            sampler = TruncatedPoissonSampler(
                shard,
                sample_rate=0.5,
                max_batch_size=max_batch_size,
                num_epochs=1,
                key=fold_in(key(100), rank),
            )
            batch = list(sampler)[0]
            assert len(batch) <= max_batch_size

    def test_truncated_single_device(self):
        """Default (no sharding) uses full dataset."""
        dataset = TensorDataset(torch.randn(500, 10))
        sampler = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.2,
            max_batch_size=60,
            num_epochs=5,
            key=key(999),
        )
        batches = list(sampler)
        assert len(batches) == 5
        for batch in batches:
            assert len(batch) <= 60


class TestEdgeCases:
    """Edge cases for distributed sampling."""

    def test_world_size_one_same_as_no_sharding(self):
        """Explicit world_size=1 produces same shard as full dataset."""
        dataset = TensorDataset(torch.randn(100, 10))
        shard = _make_shard(dataset, 0, 1)
        assert len(shard) == len(dataset)

    def test_very_small_shards(self):
        """Works with more workers than typical shard size."""
        dataset = TensorDataset(torch.randn(50, 10))
        world_size = 10

        for rank in range(world_size):
            shard = _make_shard(dataset, rank, world_size)
            sampler = PoissonSampler(
                shard,
                sample_rate=0.5,
                num_epochs=10,
                key=fold_in(key(0), rank),
            )
            batches = list(sampler)
            assert len(batches) == 10
