"""Tests for CyclicPoissonSampler with DDP support."""

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from opaque.random import key
from opaque.sampling import CyclicPoissonSampler, PartitionType


class TestCyclicPoissonSamplerBasic:
    """Basic functionality tests (single device)."""

    def test_standard_poisson(self):
        """cycle_length=1 behaves like standard Poisson sampling."""
        dataset_size = 100
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.5,
            cycle_length=1,
            iterations=10,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 10

        # All indices should be valid
        for batch in batches:
            assert all(0 <= idx < dataset_size for idx in batch)

        # In standard Poisson, should get some variation in batch sizes
        batch_sizes = [len(b) for b in batches]
        assert len(set(batch_sizes)) > 1  # Not all same size

    def test_fixed_order(self):
        """sampling_prob=1.0 gives deterministic fixed order."""
        dataset_size = 20
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=1.0,
            cycle_length=4,
            iterations=8,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 8

        # With prob=1, each batch should have all its group's elements
        group_size = dataset_size / 4
        for batch in batches:
            assert len(batch) == int(group_size)

    def test_cyclic_structure(self):
        """Verify cyclic group structure."""
        dataset_size = 15
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=1.0,
            cycle_length=3,
            iterations=6,
            key=key(42),
        )

        batches = list(sampler)

        # Batches 0 and 3 should have same elements (same group)
        assert set(batches[0]) == set(batches[3])
        # Batches 1 and 4 should have same elements
        assert set(batches[1]) == set(batches[4])
        # Batches 2 and 5 should have same elements
        assert set(batches[2]) == set(batches[5])

    def test_truncated_batch_size(self):
        """truncated_batch_size caps batch sizes."""
        dataset_size = 200
        max_batch_size = 10
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=1.0,  # High prob to get large natural batches
            cycle_length=1,
            iterations=10,
            truncated_batch_size=max_batch_size,
            key=key(42),
        )

        batches = list(sampler)
        for batch in batches:
            assert len(batch) <= max_batch_size

    def test_equal_split_partition(self):
        """EQUAL_SPLIT partitions evenly."""
        dataset_size = 100
        cycle_length = 5
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=1.0,
            cycle_length=cycle_length,
            iterations=cycle_length,
            partition_type=PartitionType.EQUAL_SPLIT,
            key=key(42),
        )

        batches = list(sampler)
        # Each batch should have size dataset_size / cycle_length
        expected_size = dataset_size // cycle_length
        for batch in batches:
            assert len(batch) == expected_size

    def test_independent_partition(self):
        """INDEPENDENT partition works."""
        dataset_size = 100
        cycle_length = 4
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.5,
            cycle_length=cycle_length,
            iterations=8,
            partition_type=PartitionType.INDEPENDENT,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 8

        # Groups may have different sizes with INDEPENDENT
        # Just verify all indices are valid
        for batch in batches:
            assert all(0 <= idx < dataset_size for idx in batch)

    def test_reproducibility(self):
        """Same seed produces same batches."""
        dataset_size = 100

        def create_sampler():
            return CyclicPoissonSampler(
                range(dataset_size),
                sampling_prob=0.5,
                cycle_length=3,
                iterations=10,
                key=key(42),
            )

        batches1 = [b.copy() for b in create_sampler()]
        batches2 = [b.copy() for b in create_sampler()]

        assert batches1 == batches2

    def test_different_seeds_different_batches(self):
        """Different seeds produce different batches."""
        dataset_size = 100

        sampler1 = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.5,
            cycle_length=3,
            iterations=10,
            key=key(42),
        )

        sampler2 = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.5,
            cycle_length=3,
            iterations=10,
            key=key(123),
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # With high probability, should get different batches
        assert batches1 != batches2

    def test_key_object(self):
        """Can pass RngKey directly."""
        dataset_size = 100
        k = key(42)

        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.5,
            cycle_length=2,
            iterations=5,
            key=k,
        )

        batches = list(sampler)
        assert len(batches) == 5


class TestCyclicPoissonSamplerProperties:
    """Test statistical properties."""

    def test_expected_batch_size(self):
        """expected_batch_size property is reasonable."""
        dataset_size = 1000
        sampling_prob = 0.1
        cycle_length = 5

        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=sampling_prob,
            cycle_length=cycle_length,
            iterations=1,
            key=key(0),
        )

        # Expected: avg_group_size * sampling_prob
        group_size = dataset_size / cycle_length
        expected = group_size * sampling_prob

        assert abs(sampler.expected_batch_size - expected) < 1e-6

    def test_batch_size_variance(self):
        """batch_size_variance property is reasonable."""
        dataset_size = 1000
        sampling_prob = 0.1
        cycle_length = 5

        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=sampling_prob,
            cycle_length=cycle_length,
            iterations=1,
            key=key(0),
        )

        # Variance: avg_group_size * p * (1 - p)
        group_size = dataset_size / cycle_length
        expected_var = group_size * sampling_prob * (1 - sampling_prob)

        assert abs(sampler.batch_size_variance - expected_var) < 1e-6

    def test_len(self):
        """__len__ returns iterations."""
        dataset_size = 100
        iterations = 25

        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.5,
            cycle_length=2,
            iterations=iterations,
            key=key(0),
        )

        assert len(sampler) == iterations


class TestCyclicPoissonSamplerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset_raises(self):
        """Empty dataset raises error."""
        with pytest.raises(ValueError):
            CyclicPoissonSampler([], sampling_prob=0.5, key=key(0))

    def test_single_example(self):
        """Single example dataset works."""
        sampler = CyclicPoissonSampler(
            [0],
            sampling_prob=1.0,
            cycle_length=1,
            iterations=3,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 3
        # With prob=1.0, should always get [0]
        assert all(batch == [0] for batch in batches)

    def test_cycle_length_larger_than_dataset(self):
        """cycle_length > dataset_size works but gives small groups."""
        dataset_size = 10
        sampler = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.5,
            cycle_length=20,
            iterations=20,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 20
        # All indices valid
        for batch in batches:
            assert all(0 <= idx < dataset_size for idx in batch)

    def test_sampling_prob_edge_values(self):
        """Test sampling_prob near 0 and 1."""
        dataset_size = 100

        # Near 1
        sampler_high = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.99,
            cycle_length=1,
            iterations=5,
            key=key(42),
        )
        batches_high = list(sampler_high)
        assert all(len(b) > 0.9 * dataset_size for b in batches_high)

        # Near 0
        sampler_low = CyclicPoissonSampler(
            range(dataset_size),
            sampling_prob=0.01,
            cycle_length=1,
            iterations=20,
            key=key(42),
        )
        batches_low = list(sampler_low)
        assert all(len(b) < 0.1 * dataset_size for b in batches_low)

    def test_invalid_sampling_prob_raises(self):
        """Invalid sampling_prob raises ValueError."""
        with pytest.raises(ValueError):
            CyclicPoissonSampler(range(10), sampling_prob=0.0, key=key(0))

        with pytest.raises(ValueError):
            CyclicPoissonSampler(range(10), sampling_prob=1.5, key=key(0))

    def test_invalid_cycle_length_raises(self):
        """Invalid cycle_length raises ValueError."""
        with pytest.raises(ValueError):
            CyclicPoissonSampler(range(10), sampling_prob=0.5, cycle_length=0, key=key(0))

    def test_invalid_truncated_batch_size_raises(self):
        """Invalid truncated_batch_size raises ValueError."""
        with pytest.raises(ValueError):
            CyclicPoissonSampler(
                range(10),
                sampling_prob=0.5,
                truncated_batch_size=0,
                key=key(0),
            )


class TestCyclicPoissonSamplerDistributedSimulation:
    """Test DDP support via mocking."""

    def test_ddp_auto_detection(self):
        """Auto-detects distributed environment."""
        # Mock distributed functions
        with patch("opaque.sampling.cyclic_poisson.is_distributed", return_value=True):
            with patch("opaque.sampling.cyclic_poisson.get_rank", return_value=0):
                with patch(
                    "opaque.sampling.cyclic_poisson.get_world_size", return_value=4
                ):
                    sampler = CyclicPoissonSampler(
                        range(100),
                        sampling_prob=0.5,
                        cycle_length=2,
                        iterations=5,
                        key=key(0),
                    )

                    assert sampler.is_distributed
                    assert sampler.world_size == 4
                    assert sampler.rank == 0
                    assert sampler.mode == "SHARDED"

    def test_ddp_sharding_rank_0(self):
        """Rank 0 gets correct shard."""
        with patch("opaque.sampling.cyclic_poisson.is_distributed", return_value=True):
            with patch("opaque.sampling.cyclic_poisson.get_rank", return_value=0):
                with patch(
                    "opaque.sampling.cyclic_poisson.get_world_size", return_value=4
                ):
                    dataset_size = 100
                    sampler = CyclicPoissonSampler(
                        range(dataset_size),
                        sampling_prob=1.0,
                        cycle_length=1,
                        iterations=1,
                        key=key(42),
                    )

                    # Rank 0: [0, 25)
                    assert sampler.start_idx == 0
                    assert sampler.end_idx == 25

                    batches = list(sampler)
                    assert all(0 <= idx < 25 for batch in batches for idx in batch)

    def test_ddp_sharding_rank_last(self):
        """Last rank gets remainder."""
        with patch("opaque.sampling.cyclic_poisson.is_distributed", return_value=True):
            with patch("opaque.sampling.cyclic_poisson.get_rank", return_value=3):
                with patch(
                    "opaque.sampling.cyclic_poisson.get_world_size", return_value=4
                ):
                    dataset_size = 100
                    sampler = CyclicPoissonSampler(
                        range(dataset_size),
                        sampling_prob=1.0,
                        cycle_length=1,
                        iterations=1,
                        key=key(42),
                    )

                    # Rank 3: [75, 100)
                    assert sampler.start_idx == 75
                    assert sampler.end_idx == 100

                    batches = list(sampler)
                    assert all(75 <= idx < 100 for batch in batches for idx in batch)

    def test_ddp_seed_shifting(self):
        """Seed auto-shifts by rank."""
        # Rank 0: seed=42
        with patch("opaque.sampling.cyclic_poisson.is_distributed", return_value=True):
            with patch("opaque.sampling.cyclic_poisson.get_rank", return_value=0):
                with patch(
                    "opaque.sampling.cyclic_poisson.get_world_size", return_value=4
                ):
                    sampler0 = CyclicPoissonSampler(
                        range(100),
                        sampling_prob=0.5,
                        cycle_length=1,
                        iterations=5,
                        key=key(42),
                    )

                    batches0 = list(sampler0)

        # Rank 1: seed should be different via fold_in
        with patch("opaque.sampling.cyclic_poisson.is_distributed", return_value=True):
            with patch("opaque.sampling.cyclic_poisson.get_rank", return_value=1):
                with patch(
                    "opaque.sampling.cyclic_poisson.get_world_size", return_value=4
                ):
                    sampler1 = CyclicPoissonSampler(
                        range(100),
                        sampling_prob=0.5,
                        cycle_length=1,
                        iterations=5,
                        key=key(42),
                    )

                    batches1 = list(sampler1)

        # Different ranks should get different batches (with high probability)
        assert batches0 != batches1

    def test_ddp_cyclic_independence(self):
        """Each rank cycles through its shard independently."""
        # Rank 0
        with patch("opaque.sampling.cyclic_poisson.is_distributed", return_value=True):
            with patch("opaque.sampling.cyclic_poisson.get_rank", return_value=0):
                with patch(
                    "opaque.sampling.cyclic_poisson.get_world_size", return_value=2
                ):
                    dataset_size = 100
                    sampler0 = CyclicPoissonSampler(
                        range(dataset_size),
                        sampling_prob=1.0,
                        cycle_length=2,
                        iterations=4,
                        key=key(42),
                    )

                    batches0 = list(sampler0)
                    # Rank 0 cycles: [0-50) → groups 0,1,0,1

                    # Check all indices are in shard 0: [0, 50)
                    for batch in batches0:
                        assert all(0 <= idx < 50 for idx in batch)  # Rank 0's shard


class TestCyclicPoissonSamplerDataLoader:
    """Test integration with PyTorch DataLoader."""

    def test_dataloader_integration(self):
        """Works with PyTorch DataLoader."""
        dataset = list(range(100))
        sampler = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.5,
            cycle_length=2,
            iterations=10,
            key=key(42),
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
        )

        batches = list(loader)
        assert len(batches) == 10

        # Each batch should be a tensor of valid indices
        for batch in batches:
            assert isinstance(batch, torch.Tensor)
            assert all(0 <= x < 100 for x in batch)

    def test_dataloader_variable_batch_sizes(self):
        """DataLoader handles variable batch sizes."""
        dataset = list(range(200))
        sampler = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.3,
            cycle_length=1,
            iterations=20,
            key=key(42),
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
        )

        batch_sizes = [len(batch) for batch in loader]
        # Poisson property: sizes should vary
        assert len(set(batch_sizes)) > 1


class TestCyclicPoissonSamplerRangeDataset:
    """Test with actual Dataset objects."""

    def test_with_torch_dataset(self):
        """Works with torch.utils.data.Dataset."""

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return idx

        dataset = SimpleDataset(100)
        sampler = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.5,
            cycle_length=3,
            iterations=9,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) == 9

        # All indices valid
        for batch in batches:
            assert all(0 <= idx < 100 for idx in batch)
