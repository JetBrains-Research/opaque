"""DP-SGD integration tests for ``opaque.random`` helpers.

The engine-only RNG helper tests live in
``packages/opaque-engine/tests/rng/test_rng_helpers.py``. The tests in
this file exercise the helpers alongside ``opaque.dpsgd.noise`` to
validate the full step-key derivation pattern; they live in
opaque-dpsgd because dpsgd depends on engine, not the other way around.
"""

from opaque.random import fold_in, key, random_key


class TestHelperIntegration:
    """Test helpers work with actual noise functions."""

    def test_random_key_with_gaussian_noise(self):
        """random_key() should work with gaussian_noise()."""
        from opaque.dpsgd.noise import gaussian_noise

        k = random_key()
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            key=k,
        )
        assert callable(noise_fn)

    def test_fold_in_with_gaussian_noise(self):
        """fold_in() derived key should work with gaussian_noise()."""
        from opaque.dpsgd.noise import gaussian_noise

        k = fold_in(key(42), 10)
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            key=k,
        )
        assert callable(noise_fn)

    def test_training_loop_pattern(self):
        """Should demonstrate typical training loop usage with fold_in."""
        from opaque.dpsgd.noise import gaussian_noise

        base = key(42)
        losses = []
        for step in range(3):
            k = fold_in(base, step)
            noise_fn, state = gaussian_noise(
                noise_multiplier=1.0,
                key=k,
            )
            # Simulate loss
            loss = step * 0.1
            losses.append(loss)

        assert len(losses) == 3
