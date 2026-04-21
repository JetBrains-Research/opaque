"""Tests for RNG convenience helpers."""

import torch

from opaque.core.random import RngKey, key, random_key


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
        from opaque.core.random import generator_from_key

        k = random_key()
        gen = generator_from_key(k)
        assert isinstance(gen, torch.Generator)

        # Should produce different random values
        t1 = torch.randn(10, generator=gen)
        k2 = random_key()
        gen2 = generator_from_key(k2)
        t2 = torch.randn(10, generator=gen2)
        assert not torch.allclose(t1, t2)


class TestHelperIntegration:
    """Test helpers work with actual noise functions."""

    def test_random_key_with_gaussian_noise(self):
        """random_key() should work with gaussian_noise()."""
        from opaque.dpsgd.noise.gaussian import gaussian_noise

        k = random_key()
        noise_fn, state = gaussian_noise(
            stddev=1.0,
            key=k,
        )
        assert callable(noise_fn)

    def test_fold_in_with_gaussian_noise(self):
        """fold_in() derived key should work with gaussian_noise()."""
        from opaque.dpsgd.noise.gaussian import gaussian_noise
        from opaque.core.random import fold_in

        k = fold_in(key(42), 10)
        noise_fn, state = gaussian_noise(
            stddev=1.0,
            key=k,
        )
        assert callable(noise_fn)

    def test_training_loop_pattern(self):
        """Should demonstrate typical training loop usage with fold_in."""
        from opaque.dpsgd.noise.gaussian import gaussian_noise
        from opaque.core.random import fold_in

        base = key(42)
        losses = []
        for step in range(3):
            k = fold_in(base, step)
            noise_fn, state = gaussian_noise(
                stddev=1.0,
                key=k,
            )
            # Simulate loss
            loss = step * 0.1
            losses.append(loss)

        assert len(losses) == 3
