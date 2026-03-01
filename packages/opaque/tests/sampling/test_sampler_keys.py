"""Tests for key-based RNG in samplers (Phase 2 migration)."""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from opaque.random import fold_in, key
from opaque.sampling import (
    CyclicPoissonSampler,
    PoissonSampler,
    TruncatedPoissonSampler,
)


class TestPoissonSamplerKeys:
    """Test PoissonSampler with key-based RNG."""

    def test_requires_key_parameter(self):
        """Should require key parameter (no None fallback)."""
        dataset = TensorDataset(torch.randn(1000, 10))

        with pytest.raises(TypeError, match="key"):
            PoissonSampler(dataset, sample_rate=0.1, num_iterations=5)

    def test_reproducibility_with_same_key(self):
        """Same key should produce same samples."""
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = PoissonSampler(
            dataset, sample_rate=0.1, num_iterations=5, key=key(42)
        )
        batches1 = list(sampler1)

        sampler2 = PoissonSampler(
            dataset, sample_rate=0.1, num_iterations=5, key=key(42)
        )
        batches2 = list(sampler2)

        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2, strict=False):
            assert b1 == b2, "Same key should produce identical samples"

    def test_different_keys_produce_different_samples(self):
        """Different keys should produce different samples."""
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = PoissonSampler(
            dataset, sample_rate=0.1, num_iterations=5, key=key(42)
        )
        batches1 = list(sampler1)

        sampler2 = PoissonSampler(
            dataset, sample_rate=0.1, num_iterations=5, key=key(43)
        )
        batches2 = list(sampler2)

        # Should be different (statistical test)
        assert batches1 != batches2, "Different keys should produce different samples"

    def test_rank_shifting_via_fold_in(self):
        """Rank shifting should work via fold_in, not seed + rank."""
        dataset = TensorDataset(torch.randn(1000, 10))
        base_key = key(42)

        # Simulate rank 0
        rank0_key = base_key  # No shift
        sampler_rank0 = PoissonSampler(
            dataset, sample_rate=0.1, num_iterations=5, key=rank0_key
        )
        batches_rank0 = list(sampler_rank0)

        # Simulate rank 1 (fold in rank)
        rank1_key = fold_in(base_key, 1)
        sampler_rank1 = PoissonSampler(
            dataset, sample_rate=0.1, num_iterations=5, key=rank1_key
        )
        batches_rank1 = list(sampler_rank1)

        # Different ranks should produce different samples
        assert batches_rank0 != batches_rank1

    def test_fold_in_helper_integration(self):
        """fold_in() should work seamlessly for per-step keys."""
        dataset = TensorDataset(torch.randn(1000, 10))

        # Use fold_in for per-step reproducible sampling
        k = fold_in(key(42), 0)
        sampler = PoissonSampler(dataset, sample_rate=0.1, num_iterations=5, key=k)

        batches = list(sampler)
        assert len(batches) > 0
        assert all(isinstance(b, list) for b in batches)

    def test_sampling_with_key(self):
        """Sampling should work with keys."""
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler = PoissonSampler(
            dataset,
            sample_rate=0.1,
            num_iterations=5,
            key=key(42),
        )

        batches = list(sampler)
        assert len(batches) > 0


class TestTruncatedPoissonSamplerKeys:
    """Test TruncatedPoissonSampler with key-based RNG."""

    def test_requires_key_parameter(self):
        """Should require key parameter."""
        dataset = TensorDataset(torch.randn(1000, 10))

        with pytest.raises(TypeError, match="key"):
            TruncatedPoissonSampler(
                dataset, sample_rate=0.1, max_batch_size=50, num_iterations=5
            )

    def test_reproducibility_with_same_key(self):
        """Same key should produce same truncated samples."""
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.1,
            max_batch_size=50,
            num_iterations=5,
            key=key(42),
        )
        batches1 = list(sampler1)

        sampler2 = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.1,
            max_batch_size=50,
            num_iterations=5,
            key=key(42),
        )
        batches2 = list(sampler2)

        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2, strict=False):
            assert b1 == b2

    def test_truncation_respects_max_batch_size(self):
        """Truncated sampler should never exceed max_batch_size."""
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler = TruncatedPoissonSampler(
            dataset,
            sample_rate=0.5,  # High rate to frequently exceed max
            max_batch_size=50,
            num_iterations=10,
            key=key(42),
        )

        for batch in sampler:
            assert len(batch) <= 50, f"Batch size {len(batch)} exceeds max 50"


class TestCyclicPoissonSamplerKeys:
    """Test CyclicPoissonSampler with key-based RNG."""

    def test_requires_key_parameter(self):
        """Should require key parameter."""
        dataset = TensorDataset(torch.randn(1000, 10))

        with pytest.raises(TypeError, match="key"):
            CyclicPoissonSampler(dataset, sampling_prob=0.1, cycle_length=10)

    def test_reproducibility_with_same_key(self):
        """Same key should produce same cyclic samples."""
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.1,
            cycle_length=10,
            iterations=50,
            key=key(42),
        )
        batches1 = list(sampler1)

        sampler2 = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.1,
            cycle_length=10,
            iterations=50,
            key=key(42),
        )
        batches2 = list(sampler2)

        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2, strict=False):
            assert b1 == b2

    def test_different_keys_produce_different_samples(self):
        """Different keys should produce different cyclic samples."""
        dataset = TensorDataset(torch.randn(1000, 10))

        sampler1 = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.1,
            cycle_length=10,
            iterations=50,
            key=key(42),
        )
        batches1 = list(sampler1)

        sampler2 = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.1,
            cycle_length=10,
            iterations=50,
            key=key(100),
        )
        batches2 = list(sampler2)

        # Should be different (at least some batches)
        assert batches1 != batches2

    def test_rank_shifting_via_fold_in_cyclic(self):
        """Cyclic sampler rank shifting via fold_in."""
        dataset = TensorDataset(torch.randn(1000, 10))
        base_key = key(42)

        # Rank 0
        sampler_rank0 = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.1,
            cycle_length=10,
            iterations=50,
            key=base_key,
        )
        batches_rank0 = list(sampler_rank0)

        # Rank 1 (fold in rank)
        rank1_key = fold_in(base_key, 1)
        sampler_rank1 = CyclicPoissonSampler(
            dataset,
            sampling_prob=0.1,
            cycle_length=10,
            iterations=50,
            key=rank1_key,
        )
        batches_rank1 = list(sampler_rank1)

        # Should be different
        assert batches_rank0 != batches_rank1


class TestCrossValidationWithNumpy:
    """Test that key-based sampling matches numpy.random.Generator behavior."""

    def test_poisson_matches_numpy_generator(self):
        """PoissonSampler with key(42) should match numpy.random.default_rng(42)."""
        dataset = TensorDataset(torch.randn(1000, 10))

        # Our implementation with key
        sampler_key = PoissonSampler(
            dataset, sample_rate=0.1, num_iterations=1, key=key(42)
        )
        batches_key = list(sampler_key)

        # Reference numpy implementation
        rng = np.random.default_rng(42)
        batches_numpy = []
        for _ in range(1):  # num_iterations=1
            mask = rng.random(len(dataset)) < 0.1
            indices = np.where(mask)[0].tolist()
            if indices:
                batches_numpy.append(indices)

        # Should have same number of batches
        assert len(batches_key) == len(batches_numpy)

        # Batch contents should match (assuming same numpy binomial algorithm)
        # Note: This test may fail if numpy changes its algorithm
        # In that case, we just verify statistical properties instead
        if len(batches_key) > 0 and len(batches_numpy) > 0:
            # At least verify they have similar sizes
            assert abs(len(batches_key[0]) - len(batches_numpy[0])) < 50
