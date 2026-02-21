"""Tests for RNG convenience helpers."""

import pytest
import torch

from opaque.random import RngKey, key, random_key, training_key


class TestRandomKey:
    """Test non-deterministic key generation for prototyping."""

    def test_returns_rng_key(self):
        """Should return an RngKey instance."""
        k = random_key()
        assert isinstance(k, RngKey)

    def test_different_on_each_call(self):
        """Should generate different keys on consecutive calls."""
        k1 = random_key()
        k2 = random_key()
        assert k1.seed != k2.seed

    def test_produces_working_generators(self):
        """Generated keys should work with generator_from_key."""
        from opaque.random import generator_from_key

        k = random_key()
        gen = generator_from_key(k)
        assert isinstance(gen, torch.Generator)

        # Should produce different random values
        t1 = torch.randn(10, generator=gen)
        k2 = random_key()
        gen2 = generator_from_key(k2)
        t2 = torch.randn(10, generator=gen2)
        assert not torch.allclose(t1, t2)


class TestTrainingKey:
    """Test deterministic key derivation for training loops."""

    def test_basic_usage(self):
        """Should fold in step by default."""
        k = training_key(base_seed=42, step=10)
        assert isinstance(k, RngKey)

        # Same base_seed + step should give identical key
        k2 = training_key(base_seed=42, step=10)
        assert k.seed == k2.seed

    def test_different_steps_different_keys(self):
        """Different steps should produce different keys."""
        k1 = training_key(base_seed=42, step=0)
        k2 = training_key(base_seed=42, step=1)
        assert k1.seed != k2.seed

    def test_synchronized_mode_ignores_rank(self):
        """synchronized=True should produce same key across ranks."""
        k_rank0 = training_key(base_seed=42, step=10, rank=0, synchronized=True)
        k_rank1 = training_key(base_seed=42, step=10, rank=1, synchronized=True)
        assert k_rank0.seed == k_rank1.seed

    def test_unsynchronized_mode_folds_rank(self):
        """synchronized=False should fold in rank."""
        k_rank0 = training_key(base_seed=42, step=10, rank=0, synchronized=False)
        k_rank1 = training_key(base_seed=42, step=10, rank=1, synchronized=False)
        assert k_rank0.seed != k_rank1.seed

    def test_auto_mode_no_rank_synchronized(self):
        """synchronized='auto' with no rank should be synchronized."""
        k1 = training_key(base_seed=42, step=10, synchronized="auto")
        k2 = training_key(base_seed=42, step=10, rank=None, synchronized="auto")
        assert k1.seed == k2.seed

    def test_auto_mode_with_rank_unsynchronized(self):
        """synchronized='auto' with rank should fold in rank."""
        k_rank0 = training_key(base_seed=42, step=10, rank=0, synchronized="auto")
        k_rank1 = training_key(base_seed=42, step=10, rank=1, synchronized="auto")
        assert k_rank0.seed != k_rank1.seed

    def test_derivation_order_step_then_rank(self):
        """Should fold step first, then rank (if unsynchronized)."""
        # Manual derivation: key(42) -> fold_in(step=10) -> fold_in(rank=1)
        from opaque.random import fold_in

        manual = fold_in(fold_in(key(42), 10), 1)
        auto = training_key(base_seed=42, step=10, rank=1, synchronized=False)
        assert manual.seed == auto.seed

    def test_invalid_synchronized_value(self):
        """Should reject invalid synchronized values."""
        with pytest.raises((ValueError, TypeError)):
            training_key(base_seed=42, step=0, synchronized="invalid")

    def test_rank_without_synchronized_param(self):
        """Passing rank without synchronized should raise helpful error."""
        with pytest.raises(ValueError, match="synchronized.*rank"):
            training_key(base_seed=42, step=0, rank=1)

    def test_negative_step(self):
        """Should handle negative steps (for validation/pre-training)."""
        k = training_key(base_seed=42, step=-1)
        assert isinstance(k, RngKey)

    def test_worker_id_folding(self):
        """Should support optional worker_id for dataloader workers."""
        k1 = training_key(base_seed=42, step=10, worker_id=0)
        k2 = training_key(base_seed=42, step=10, worker_id=1)
        assert k1.seed != k2.seed

    def test_full_derivation_chain(self):
        """Should support step -> rank -> worker chain."""
        from opaque.random import fold_in

        # Manual: key(42) -> fold(step=5) -> fold(rank=2) -> fold(worker=3)
        manual = fold_in(fold_in(fold_in(key(42), 5), 2), 3)
        auto = training_key(
            base_seed=42,
            step=5,
            rank=2,
            worker_id=3,
            synchronized=False,
        )
        assert manual.seed == auto.seed


class TestHelperIntegration:
    """Test helpers work with actual noise functions."""

    def test_random_key_with_gaussian_noise(self):
        """random_key() should work with gaussian_noise()."""
        from opaque.noise import gaussian_noise

        k = random_key()
        noise_fn, state = gaussian_noise(
            stddev=1.0,
            key=k,
        )
        assert callable(noise_fn)

    def test_training_key_with_gaussian_noise(self):
        """training_key() should work with gaussian_noise()."""
        from opaque.noise import gaussian_noise

        k = training_key(base_seed=42, step=10)
        noise_fn, state = gaussian_noise(
            stddev=1.0,
            key=k,
        )
        assert callable(noise_fn)

    def test_training_loop_pattern(self):
        """Should demonstrate typical training loop usage."""
        from opaque.noise import gaussian_noise

        losses = []
        for step in range(3):
            k = training_key(base_seed=42, step=step)
            noise_fn, state = gaussian_noise(
                stddev=1.0,
                key=k,
            )
            # Simulate loss
            loss = step * 0.1
            losses.append(loss)

        assert len(losses) == 3
