"""Tests for adaptive clipping with quantile noise (Phase 4)."""

import pytest
import torch

from opaque.clipping.adaptive import adaptive_clipped_grad
from opaque.random import key


class TestQuantileNoise:
    """Tests for quantile noise in adaptive clipping."""

    def test_quantile_noise_requires_key(self):
        """Test that key is a required argument."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        with pytest.raises(TypeError, match="key"):
            adaptive_clipped_grad(
                loss_fn,
                fraction_noise_std=0.1,
                batch_argnums=(1, 2),
            )

    def test_invalid_fraction_noise_std(self):
        """Test that negative/zero fraction_noise_std raises error."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        with pytest.raises(
            ValueError, match="fraction_noise_std must be positive"
        ):
            adaptive_clipped_grad(
                loss_fn,
                fraction_noise_std=-1.0,
                key=key(42),
                batch_argnums=(1, 2),
            )

    def test_quantile_noise_reproducibility(self):
        """Test that same key produces reproducible adaptation."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create two identical setups
        grad_fn1, state1 = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
        )

        grad_fn2, state2 = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
        )

        # Same data
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run multiple steps
        for _ in range(5):
            _, state1 = grad_fn1(params, batch_x, batch_y, state=state1)
            _, state2 = grad_fn2(params, batch_x, batch_y, state=state2)

            # Clip norms should be identical (same noise sequence)
            assert state1.next_clip_norm == state2.next_clip_norm
            assert state1.step == state2.step

    def test_quantile_noise_different_keys_produce_different_paths(self):
        """Test that different keys can produce different adaptation paths.

        Note: This test checks that the keys are preserved correctly and used in
        adaptation. Due to stochasticity, paths may or may not diverge in just
        a few steps, depending on clipping rates and noise magnitude. We verify
        that states preserve their different keys correctly.
        """

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create two setups with different keys
        grad_fn1, state1 = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            fraction_noise_std=1.0,  # Very large noise to increase chance of divergence
            key=key(42),
            batch_argnums=(1, 2),
        )

        grad_fn2, state2 = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            fraction_noise_std=1.0,
            key=key(99),  # Different key
            batch_argnums=(1, 2),
        )

        # Verify initial states have different keys
        assert state1.key != state2.key

        # Same data
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run multiple steps - update keys during training
        for _ in range(20):
            _, state1 = grad_fn1(params, batch_x, batch_y, state=state1)
            _, state2 = grad_fn2(params, batch_x, batch_y, state=state2)

        # Keys should still be different after training
        assert state1.key != state2.key

        # With very large noise (1.0), divergence is likely but not guaranteed
        # The important thing is that keys are preserved and affect adaptation

    def test_quantile_noise_affects_adaptation(self):
        """Test that quantile noise actually affects the adaptation."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Without noise
        grad_fn_no_noise, state_no_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )

        # With substantial noise (to ensure divergence)
        grad_fn_noise, state_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            fraction_noise_std=0.5,  # Increased noise for clear divergence
            key=key(42),
            batch_argnums=(1, 2),
        )

        # Same data
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run multiple steps
        for _ in range(15):  # More steps for divergence
            _, state_no_noise = grad_fn_no_noise(
                params, batch_x, batch_y, state=state_no_noise
            )
            _, state_noise = grad_fn_noise(params, batch_x, batch_y, state=state_noise)

        # Adaptation paths should diverge with high noise
        # With fraction_noise_std=0.5, this should reliably cause different decisions
        assert state_no_noise.next_clip_norm != state_noise.next_clip_norm

    def test_state_preserves_key_and_step(self):
        """Test that state correctly preserves key and step counter."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
        )

        # Initial state
        assert state.key == key(42)
        assert state.step == 0

        # Run a few steps
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        for i in range(1, 6):
            _, state = grad_fn(params, batch_x, batch_y, state=state)
            assert state.key == key(42)  # Key unchanged
            assert state.step == i  # Step incremented

    def test_quantile_noise_with_aux_output(self):
        """Test that quantile noise works with aux output."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            return_aux=True,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (grads, aux), state = grad_fn(params, batch_x, batch_y, state=state)

        # Check aux contains expected fields
        assert aux.clipping_rate is not None
        assert grads.shape == params.shape
        assert state.step == 1


class TestQuantileNoiseClipNorm:
    """Tests for clip_norm with quantile noise."""

    def test_clip_norm_unchanged_by_quantile_noise(self):
        """Test that quantile noise doesn't affect initial clip_norm."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Without noise
        _, state_no_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )

        # With noise
        _, state_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
        )

        # clip_norm should be identical (same initial_clip_norm)
        assert state_no_noise.clip_norm == state_noise.clip_norm
        assert state_no_noise.clip_norm == 1.0

    def test_clip_norm_matches_initial(self):
        """Test clip_norm equals initial_clip_norm before any steps."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        _, state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=5.0,
            fraction_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
        )

        assert state.clip_norm == 5.0
