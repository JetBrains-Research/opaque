"""Tests for adaptive clipping with quantile noise (Phase 4)."""

import pytest
import torch

from opaque.clipping.adaptive import adaptive_clipped_grad
from opaque.random import key


class TestQuantileNoise:
    """Tests for quantile noise in adaptive clipping."""

    def test_quantile_noise_requires_key(self):
        """Test that quantile_noise_std requires a key."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        with pytest.raises(ValueError, match="key must be provided"):
            adaptive_clipped_grad(
                loss_fn,
                quantile_noise_std=1.0,
                key=None,  # Missing key
                batch_argnums=(1, 2),
            )

    def test_key_without_noise_raises(self):
        """Test that providing key without quantile_noise_std raises error."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        with pytest.raises(ValueError, match="key provided but quantile_noise_std is None"):
            adaptive_clipped_grad(
                loss_fn,
                key=key(42),  # Key provided
                quantile_noise_std=None,  # But no noise
                batch_argnums=(1, 2),
            )

    def test_invalid_quantile_noise_std(self):
        """Test that negative/zero quantile_noise_std raises error."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        with pytest.raises(ValueError, match="quantile_noise_std must be positive"):
            adaptive_clipped_grad(
                loss_fn,
                quantile_noise_std=-1.0,
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
            quantile_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
        )

        grad_fn2, state2 = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            quantile_noise_std=0.1,
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
            assert state1.clip_norm == state2.clip_norm
            assert state1.step == state2.step

    def test_quantile_noise_different_keys_produce_different_paths(self):
        """Test that different keys produce different adaptation paths."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create two setups with different keys
        grad_fn1, state1 = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            quantile_noise_std=0.5,  # Larger noise
            key=key(42),
            batch_argnums=(1, 2),
        )

        grad_fn2, state2 = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            quantile_noise_std=0.5,
            key=key(99),  # Different key
            batch_argnums=(1, 2),
        )

        # Same data
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run multiple steps
        for _ in range(10):
            _, state1 = grad_fn1(params, batch_x, batch_y, state=state1)
            _, state2 = grad_fn2(params, batch_x, batch_y, state=state2)

        # Clip norms should diverge due to different noise
        assert state1.clip_norm != state2.clip_norm

    def test_quantile_noise_affects_adaptation(self):
        """Test that quantile noise actually affects the adaptation."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Without noise
        grad_fn_no_noise, state_no_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
        )

        # With substantial noise (to ensure divergence)
        grad_fn_noise, state_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            quantile_noise_std=0.5,  # Increased noise for clear divergence
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
        # With quantile_noise_std=0.5, this should reliably cause different decisions
        assert state_no_noise.clip_norm != state_noise.clip_norm

    def test_state_preserves_key_and_step(self):
        """Test that state correctly preserves key and step counter."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            quantile_noise_std=0.1,
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
            quantile_noise_std=0.1,
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


class TestQuantileNoiseSensitivity:
    """Tests for sensitivity computation with quantile noise."""

    def test_sensitivity_unchanged_by_quantile_noise(self):
        """Test that quantile noise doesn't affect sensitivity calculation."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Without noise
        _, state_no_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
        )

        # With noise
        _, state_noise = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            quantile_noise_std=0.1,
            key=key(42),
            batch_argnums=(1, 2),
        )

        # Sensitivity should be identical (same clip_norm)
        assert state_no_noise.sensitivity() == state_noise.sensitivity()
        assert state_no_noise.sensitivity() == 1.0  # clip_norm=1.0

    def test_sensitivity_with_rescale(self):
        """Test sensitivity with rescale_to_unit_norm and quantile noise."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        _, state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=5.0,
            quantile_noise_std=0.1,
            key=key(42),
            rescale_to_unit_norm=True,
            batch_argnums=(1, 2),
        )

        # With rescale_to_unit_norm, sensitivity is always 1.0
        assert state.sensitivity() == 1.0
        assert state.rescale_to_unit_norm is True
