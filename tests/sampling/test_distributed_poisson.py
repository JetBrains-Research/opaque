"""Tests for distributed Poisson sampling modes."""

import os

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from opaque.sampling import PoissonSampler, SamplingMode, TruncatedPoissonSampler


class TestSamplingModeValidation:
    """Tests for SamplingMode parameter validation."""

    def test_independent_mode_no_rank_or_world_size(self):
        """Test INDEPENDENT mode works without rank/world_size."""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(
            dataset, sample_rate=0.1, mode=SamplingMode.INDEPENDENT
        )
        assert sampler.mode == SamplingMode.INDEPENDENT

    def test_sharded_mode_requires_rank_and_world_size(self, monkeypatch):
        """Test SHARDED mode requires rank and world_size from environment."""
        dataset = TensorDataset(torch.randn(100, 10))

        # Ensure env vars are not set
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        with pytest.raises(
            ValueError, match="RANK and WORLD_SIZE environment variables"
        ):
            PoissonSampler(dataset, sample_rate=0.1, mode=SamplingMode.SHARDED)

    def test_independent_mode_with_distributed_params_warns(self, monkeypatch):
        """Test INDEPENDENT mode with world_size > 1 emits warning."""
        dataset = TensorDataset(torch.randn(100, 10))

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")

        with pytest.warns(UserWarning, match="mixture Gaussian accounting"):
            PoissonSampler(
                dataset,
                sample_rate=0.1,
                mode=SamplingMode.INDEPENDENT,
            )


class TestShardedMode:
    """Tests for SHARDED sampling mode (partition-aware)."""

    def test_sharded_mode_disjoint_indices(self, monkeypatch):
        """Test that workers sample from disjoint shards."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sample_rate = 0.5  # High rate to ensure overlap would be detected
        num_epochs = 1

        # Collect indices from each worker
        all_worker_indices = []
        for rank in range(4):
            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("WORLD_SIZE", "4")
            sampler = PoissonSampler(
                dataset,
                sample_rate=sample_rate,
                num_epochs=num_epochs,
                mode=SamplingMode.SHARDED,
                generator=np.random.default_rng(42 + rank),  # Different seeds
            )
            batches = list(sampler)
            all_worker_indices.append(batches[0])

        # Check that indices from different workers are disjoint
        for i in range(4):
            for j in range(i + 1, 4):
                indices_i = set(all_worker_indices[i])
                indices_j = set(all_worker_indices[j])
                overlap = indices_i & indices_j
                assert len(overlap) == 0, (
                    f"Workers {i} and {j} have overlapping indices: {overlap}"
                )

    def test_sharded_mode_correct_shard_boundaries(self, monkeypatch):
        """Test that each worker samples from correct shard."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sample_rate = 1.0  # Include all to verify boundaries
        world_size = 4

        for rank in range(world_size):
            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("WORLD_SIZE", str(world_size))
            sampler = PoissonSampler(
                dataset,
                sample_rate=sample_rate,
                num_epochs=1,
                mode=SamplingMode.SHARDED,
            )
            batch = list(sampler)[0]

            # Compute expected shard boundaries
            shard_size = 1000 // world_size
            start_idx = rank * shard_size
            end_idx = start_idx + shard_size if rank < world_size - 1 else 1000

            # All indices should be in this worker's shard
            assert all(start_idx <= idx < end_idx for idx in batch), (
                f"Worker {rank} has indices outside its shard [{start_idx}, {end_idx})"
            )

    def test_sharded_mode_handles_uneven_splits(self, monkeypatch):
        """Test that last worker gets remainder in uneven splits."""
        dataset = TensorDataset(torch.randn(103, 10))  # Not divisible by 4
        sample_rate = 1.0
        world_size = 4

        # Last worker should get largest shard
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        sampler_last = PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            num_epochs=1,
            mode=SamplingMode.SHARDED,
        )
        batch_last = list(sampler_last)[0]

        # Worker 3 gets indices [75, 103) = 28 examples
        # Other workers get 25 examples each
        expected_size_last = 103 - 3 * (103 // 4)
        assert len(batch_last) == expected_size_last

    def test_sharded_mode_statistical_properties_per_shard(self, monkeypatch):
        """Test that each shard maintains Poisson properties."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sample_rate = 0.1
        world_size = 4

        for rank in range(world_size):
            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("WORLD_SIZE", str(world_size))
            sampler = PoissonSampler(
                dataset,
                sample_rate=sample_rate,
                num_epochs=100,
                mode=SamplingMode.SHARDED,
                generator=np.random.default_rng(100 + rank),
            )

            batch_sizes = [len(batch) for batch in sampler]

            # Expected mean for this shard
            shard_size = 1000 // world_size
            if rank == world_size - 1:
                shard_size = 1000 - rank * shard_size
            expected_mean = shard_size * sample_rate

            actual_mean = np.mean(batch_sizes)
            # Tolerance: 50% of expected mean + 3 for small shards
            # Small shards have high variance, so we need loose tolerance
            tolerance = expected_mean * 0.5 + 3
            assert abs(actual_mean - expected_mean) < tolerance

    def test_sharded_mode_different_workers_independent(self, monkeypatch):
        """Test that different workers sample independently."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sample_rate = 0.5

        # Get batch sizes from two workers
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        sampler0 = PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            num_epochs=20,
            mode=SamplingMode.SHARDED,
            generator=np.random.default_rng(42),
        )

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "4")
        sampler1 = PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            num_epochs=20,
            mode=SamplingMode.SHARDED,
            generator=np.random.default_rng(43),
        )

        sizes0 = [len(batch) for batch in sampler0]
        sizes1 = [len(batch) for batch in sampler1]

        # Batch sizes should be different (independent sampling)
        assert sizes0 != sizes1, "Workers should sample independently"


class TestIndependentMode:
    """Tests for INDEPENDENT sampling mode (backward compatibility)."""

    def test_independent_mode_is_default(self):
        """Test that INDEPENDENT is the default mode."""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = PoissonSampler(dataset, sample_rate=0.1)
        assert sampler.mode == SamplingMode.INDEPENDENT

    def test_independent_mode_different_batches_across_workers(self):
        """Test that workers in INDEPENDENT mode get different batches."""
        dataset = TensorDataset(torch.randn(1000, 10))
        sample_rate = 0.1

        # Create samplers for two workers with different RNG seeds
        sampler0 = PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            num_epochs=10,
            mode=SamplingMode.INDEPENDENT,
            generator=np.random.default_rng(42),
        )
        sampler1 = PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            num_epochs=10,
            mode=SamplingMode.INDEPENDENT,
            generator=np.random.default_rng(43),
        )

        batches0 = list(sampler0)
        batches1 = list(sampler1)

        # Batches should be different
        assert batches0 != batches1, "Independent workers should differ"

    def test_independent_mode_original_behavior(self):
        """Test that INDEPENDENT mode matches original PoissonSampler behavior."""
        dataset = TensorDataset(torch.randn(500, 10))
        sample_rate = 0.2
        seed = 12345

        # Old-style sampler (implicit INDEPENDENT)
        sampler_old = PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            num_epochs=5,
            generator=np.random.default_rng(seed),
        )

        # New-style explicit INDEPENDENT
        sampler_new = PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            num_epochs=5,
            mode=SamplingMode.INDEPENDENT,
            generator=np.random.default_rng(seed),
        )

        batches_old = list(sampler_old)
        batches_new = list(sampler_new)

        # Should produce identical results
        assert batches_old == batches_new


class TestTruncatedPoissonDistributed:
    """Tests for TruncatedPoissonSampler with distributed modes."""

    def test_truncated_sharded_mode(self, monkeypatch):
        """Test TruncatedPoissonSampler in SHARDED mode."""
        dataset = TensorDataset(torch.randn(1000, 10))
        max_batch_size = 50
        world_size = 4

        all_indices = []
        for rank in range(world_size):
            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("WORLD_SIZE", str(world_size))
            sampler = TruncatedPoissonSampler(
                dataset,
                sample_rate=0.5,
                max_batch_size=max_batch_size,
                num_epochs=1,
                mode=SamplingMode.SHARDED,
                generator=np.random.default_rng(rank + 100),
            )
            batch = list(sampler)[0]
            all_indices.append(batch)

            # Respect max_batch_size
            assert len(batch) <= max_batch_size

        # Indices should be disjoint
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert not (set(all_indices[i]) & set(all_indices[j]))

    def test_truncated_independent_mode_backward_compat(self):
        """Test TruncatedPoissonSampler backward compatibility."""
        dataset = TensorDataset(torch.randn(500, 10))
        sample_rate = 0.2
        max_batch_size = 60
        seed = 999

        # Without mode parameter (should default to INDEPENDENT)
        sampler = TruncatedPoissonSampler(
            dataset,
            sample_rate=sample_rate,
            max_batch_size=max_batch_size,
            num_epochs=5,
            generator=np.random.default_rng(seed),
        )

        assert sampler.mode == SamplingMode.INDEPENDENT

        # Should work as before
        batches = list(sampler)
        assert len(batches) == 5
        for batch in batches:
            assert len(batch) <= max_batch_size


class TestEdgeCases:
    """Edge cases for distributed sampling modes."""

    def test_single_worker_sharded_mode(self, monkeypatch):
        """Test SHARDED mode with world_size=1."""
        dataset = TensorDataset(torch.randn(100, 10))
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        sampler = PoissonSampler(
            dataset,
            sample_rate=0.5,
            num_epochs=5,
            mode=SamplingMode.SHARDED,
            generator=np.random.default_rng(42),
        )

        batches = list(sampler)
        assert len(batches) == 5

        # Should sample from full dataset
        all_indices = set()
        for batch in batches:
            all_indices.update(batch)
        # With p=0.5 and multiple epochs, should see many indices
        assert len(all_indices) > 20

    def test_very_small_shards(self, monkeypatch):
        """Test SHARDED mode with more workers than optimal."""
        dataset = TensorDataset(torch.randn(50, 10))
        world_size = 10  # Each shard has only 5 examples

        for rank in range(world_size):
            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("WORLD_SIZE", str(world_size))
            sampler = PoissonSampler(
                dataset,
                sample_rate=0.5,
                num_epochs=10,
                mode=SamplingMode.SHARDED,
                generator=np.random.default_rng(rank),
            )
            # Should not crash
            batches = list(sampler)
            assert len(batches) == 10
