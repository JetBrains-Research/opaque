"""Tests for CyclicPoissonSampling and batch splitting utilities."""

import numpy as np
import pytest

from opaque.sampling.cyclic_poisson import (
    CyclicPoissonSampling,
    PartitionType,
    pad_to_multiple_of,
    split_and_pad_global_batch,
)


class TestCyclicPoissonSampling:
    def test_standard_poisson(self):
        """cycle_length=1 is standard Poisson sampling."""
        rng = np.random.default_rng(42)
        sampler = CyclicPoissonSampling(
            sampling_prob=0.5, iterations=10, cycle_length=1
        )
        batches = list(sampler.batch_iterator(20, rng=rng))
        assert len(batches) == 10
        for batch in batches:
            assert all(0 <= idx < 20 for idx in batch)

    def test_fixed_order(self):
        """sampling_prob=1, cycle_length=n//b is fixed-order multi-epoch."""
        rng = np.random.default_rng(42)
        sampler = CyclicPoissonSampling(sampling_prob=1.0, iterations=8, cycle_length=4)
        batches = list(sampler.batch_iterator(12, rng=rng))
        assert len(batches) == 8
        # With prob=1, each batch contains all examples in its group
        for batch in batches:
            assert len(batch) == 3  # 12/4 = 3 per group

    def test_cyclic_structure(self):
        """Same group yields same set of eligible examples across cycles."""
        rng = np.random.default_rng(42)
        sampler = CyclicPoissonSampling(sampling_prob=1.0, iterations=6, cycle_length=3)
        batches = list(sampler.batch_iterator(9, rng=rng))
        # Batches 0 and 3 come from same group
        assert set(batches[0]) == set(batches[3])
        # Batches 1 and 4 come from same group
        assert set(batches[1]) == set(batches[4])

    def test_truncated_batch_size(self):
        rng = np.random.default_rng(42)
        sampler = CyclicPoissonSampling(
            sampling_prob=1.0,
            iterations=5,
            truncated_batch_size=2,
            cycle_length=1,
        )
        batches = list(sampler.batch_iterator(100, rng=rng))
        for batch in batches:
            assert len(batch) <= 2

    def test_independent_partition(self):
        rng = np.random.default_rng(42)
        sampler = CyclicPoissonSampling(
            sampling_prob=0.5,
            iterations=5,
            cycle_length=2,
            partition_type=PartitionType.INDEPENDENT,
        )
        batches = list(sampler.batch_iterator(20, rng=rng))
        assert len(batches) == 5

    def test_all_indices_valid(self):
        rng = np.random.default_rng(42)
        sampler = CyclicPoissonSampling(
            sampling_prob=0.5, iterations=20, cycle_length=3
        )
        all_indices = set()
        for batch in sampler.batch_iterator(30, rng=rng):
            for idx in batch:
                assert 0 <= idx < 30
                all_indices.add(int(idx))

    def test_reproducibility(self):
        """Same seed produces same batches."""
        sampler = CyclicPoissonSampling(sampling_prob=0.5, iterations=5)
        batches1 = [b.tolist() for b in sampler.batch_iterator(20, rng=42)]
        batches2 = [b.tolist() for b in sampler.batch_iterator(20, rng=42)]
        assert batches1 == batches2


class TestSplitAndPadGlobalBatch:
    def test_exact_split(self):
        indices = np.arange(8)
        result = split_and_pad_global_batch(indices, minibatch_size=4)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(result[1], [4, 5, 6, 7])

    def test_padding(self):
        indices = np.arange(10)
        result = split_and_pad_global_batch(indices, minibatch_size=4)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(result[1], [4, 5, 6, 7])
        # Last batch has padding
        last = result[2]
        assert len(last) == 4
        assert 8 in last
        assert 9 in last
        assert -1 in last

    def test_single_element(self):
        indices = np.array([42])
        result = split_and_pad_global_batch(indices, minibatch_size=4)
        assert len(result) == 1
        assert 42 in result[0]
        assert np.sum(result[0] == -1) == 3


class TestPadToMultipleOf:
    def test_no_padding_needed(self):
        indices = np.arange(8)
        result = pad_to_multiple_of(indices, 4)
        np.testing.assert_array_equal(result, np.arange(8))

    def test_padding(self):
        indices = np.arange(10)
        result = pad_to_multiple_of(indices, 4)
        assert len(result) == 12
        np.testing.assert_array_equal(result[:10], np.arange(10))
        np.testing.assert_array_equal(result[10:], [-1, -1])

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            pad_to_multiple_of(np.ones((3, 3)), 4)
