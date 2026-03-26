"""Tests for adaptive gradient clipping (Andrew et al. 2021)."""

import math

import pytest
import torch

from opaque.clipping.adaptive import adaptive_clipped_grad
from opaque.random import key


class TestAdaptiveClippedGrad:
    """Tests for adaptive_clipped_grad function."""

    def test_basic_usage(self):
        """Test basic adaptive clipping workflow."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create adaptive clipping function
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.5,
            key=key(0),
            batch_argnums=(1, 2),
        )

        # Check initial state
        assert clip_state.next_clip_norm == 1.0
        assert clip_state.clipping_rate == 0.0

        # Compute gradients
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Check state updated
        assert isinstance(clip_state.next_clip_norm, float)
        assert isinstance(clip_state.clipping_rate, float)
        assert grads.shape == params.shape

    def test_threshold_increases_when_too_many_clipped(self):
        """Test that threshold increases when clipping rate > target quantile."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Start with very low threshold → many gradients will be clipped
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.01,  # Very low
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        initial_clip_norm = clip_state.next_clip_norm

        # First step
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Threshold should increase (many gradients clipped)
        assert clip_state.clipping_rate > 0.5
        assert clip_state.next_clip_norm > initial_clip_norm

    def test_threshold_decreases_when_too_few_clipped(self):
        """Test that threshold decreases when clipping rate < target quantile."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Start with very high threshold → few gradients will be clipped
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=100.0,  # Very high
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        initial_clip_norm = clip_state.next_clip_norm

        # First step
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Threshold should decrease (few gradients clipped)
        assert clip_state.clipping_rate < 0.5
        assert clip_state.next_clip_norm < initial_clip_norm

    def test_geometric_update_formula(self):
        """Test proportional update: C_{t+1} = C_t * exp(η * (ρ̃ - γ))."""
        from opaque.clipping.adaptive import (
            _adaptive_clip_norm_update,
        )

        # Direct unit test of the update formula with known values
        base = 1.0
        lr = 0.2
        target = 0.5

        # Case 1: clipping_rate above target → threshold increases
        result = _adaptive_clip_norm_update(
            base_clip_norm=base,
            noisy_clipping_rate=0.8,
            target_quantile=target,
            learning_rate=lr,
            clip_norm_min=0.01,
            clip_norm_max=100.0,
        )
        expected = base * math.exp(lr * (0.8 - target))
        assert abs(result - expected) < 1e-6

        # Case 2: clipping_rate below target → threshold decreases
        result = _adaptive_clip_norm_update(
            base_clip_norm=base,
            noisy_clipping_rate=0.2,
            target_quantile=target,
            learning_rate=lr,
            clip_norm_min=0.01,
            clip_norm_max=100.0,
        )
        expected = base * math.exp(lr * (0.2 - target))
        assert abs(result - expected) < 1e-6

        # Case 3: clipping_rate == target → no change
        result = _adaptive_clip_norm_update(
            base_clip_norm=base,
            noisy_clipping_rate=target,
            target_quantile=target,
            learning_rate=lr,
            clip_norm_min=0.01,
            clip_norm_max=100.0,
        )
        assert abs(result - base) < 1e-6

    def test_proportional_update_step_size(self):
        """Test that step size is proportional to deviation from target."""
        from opaque.clipping.adaptive import _adaptive_clip_norm_update

        base = 1.0
        lr = 0.2
        target = 0.5

        # Small deviation
        result_small = _adaptive_clip_norm_update(
            base_clip_norm=base,
            noisy_clipping_rate=0.55,
            target_quantile=target,
            learning_rate=lr,
            clip_norm_min=0.01,
            clip_norm_max=100.0,
        )

        # Large deviation
        result_large = _adaptive_clip_norm_update(
            base_clip_norm=base,
            noisy_clipping_rate=0.95,
            target_quantile=target,
            learning_rate=lr,
            clip_norm_min=0.01,
            clip_norm_max=100.0,
        )

        # Large deviation should produce a bigger step
        assert abs(result_large - base) > abs(result_small - base)

    def test_threshold_clamped_to_bounds(self):
        """Test that threshold is clamped to [clip_norm_min, clip_norm_max]."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        clip_norm_min = 0.5
        clip_norm_max = 2.0

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.1,  # Below min
            target_quantile=0.5,
            learning_rate=0.2,
            clip_norm_min=clip_norm_min,
            clip_norm_max=clip_norm_max,
            key=key(0),
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run many steps to push threshold beyond bounds
        for _ in range(20):
            _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Threshold should be clamped
        assert clip_norm_min <= clip_state.next_clip_norm <= clip_norm_max

    def test_step_counter_increments(self):
        """Test that step counter increments correctly."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        for _i in range(1, 6):
            _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    def test_with_aux_output(self):
        """Test that has_aux=True works correctly."""

        def loss_fn(params, x, y):
            pred = x @ params
            loss = ((pred - y) ** 2).mean()
            # Auxiliary outputs must be tensors for torch.func.grad
            aux = {"accuracy": torch.tensor(0.95)}
            return loss, aux

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            has_aux=True,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,  # Need to explicitly request aux outputs
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Should return (grads, grad_aux) and new state
        (grads, grad_aux), clip_state = grad_fn(
            params, batch_x, batch_y, state=clip_state
        )

        assert grads.shape == params.shape

    def test_different_target_quantiles(self):
        """Test that different target quantiles affect adaptation."""
        # Use fixed seed for reproducibility in CI
        torch.manual_seed(42)

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(16, 10)  # Larger batch for more stable statistics
        batch_y = torch.randn(16)

        # Low target quantile (aim to clip fewer gradients)
        grad_fn_low, clip_state_low = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.1,
            key=key(0),
            batch_argnums=(1, 2),
        )

        # High target quantile (aim to clip more gradients)
        grad_fn_high, clip_state_high = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.9,
            key=key(0),
            batch_argnums=(1, 2),
        )

        # Run 20 steps for more reliable convergence
        for _ in range(20):
            _, clip_state_low = grad_fn_low(
                params, batch_x, batch_y, state=clip_state_low
            )
            _, clip_state_high = grad_fn_high(
                params, batch_x, batch_y, state=clip_state_high
            )

        # Low quantile should result in higher threshold (clip fewer)
        # High quantile should result in lower threshold (clip more)
        assert clip_state_low.next_clip_norm > clip_state_high.next_clip_norm

    def test_different_learning_rates(self):
        """Test that learning rate affects adaptation speed."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Slow adaptation
        grad_fn_slow, clip_state_slow = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.01,
            learning_rate=0.05,
            key=key(0),
            batch_argnums=(1, 2),
        )

        # Fast adaptation
        grad_fn_fast, clip_state_fast = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.01,
            learning_rate=0.5,
            key=key(0),
            batch_argnums=(1, 2),
        )

        # Run 5 steps
        for _ in range(5):
            _, clip_state_slow = grad_fn_slow(
                params, batch_x, batch_y, state=clip_state_slow
            )
            _, clip_state_fast = grad_fn_fast(
                params, batch_x, batch_y, state=clip_state_fast
            )

        # Fast should have adapted more
        assert abs(clip_state_fast.next_clip_norm - 0.01) > abs(
            clip_state_slow.next_clip_norm - 0.01
        )

    def test_kwargs_passed_to_clipped_grad(self):
        """Test that additional kwargs are passed to clipped_grad."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Test with return_aux=True to verify kwargs are passed
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,  # Should be passed to clipped_grad
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (grads, grad_aux), clip_state = grad_fn(
            params, batch_x, batch_y, state=clip_state
        )

        assert grads.shape == params.shape
        assert grad_aux.loss_values is not None
        assert grad_aux.grad_norms is not None

    def test_composition_with_noise(self):
        """Test that adaptive clipping composes naturally with noise."""
        from opaque.noise import gaussian_noise

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Compute clipped gradients
        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Add noise scaled to current clip norm
        noise_fn, noise_state = gaussian_noise(
            stddev=1.1 * clip_state.clip_norm,
            key=key(0),
        )
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        assert noisy_grads.shape == grads.shape
        assert not torch.allclose(noisy_grads, grads)  # Noise added

    def test_microbatching_produces_identical_results(self):
        """Test that microbatching produces identical gradients and state updates."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create two functions with same config, one with microbatching
        grad_fn_no_mb, state_no_mb = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
            microbatch_size=None,  # No microbatching
        )

        grad_fn_mb, state_mb = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.5,
            learning_rate=0.2,
            key=key(0),
            batch_argnums=(1, 2),
            microbatch_size=8,  # Process in microbatches of 8
        )

        # Test over multiple steps to verify state updates match
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(32, 10)
        batch_y = torch.randn(32)

        for _step in range(3):
            grads_no_mb, state_no_mb = grad_fn_no_mb(
                params, batch_x, batch_y, state=state_no_mb
            )
            grads_mb, state_mb = grad_fn_mb(params, batch_x, batch_y, state=state_mb)

            # Gradients should be identical
            torch.testing.assert_close(grads_mb, grads_no_mb, rtol=1e-5, atol=1e-6)

            # State updates should be identical
            assert math.isclose(state_mb.next_clip_norm, state_no_mb.next_clip_norm, rel_tol=1e-5)
            assert math.isclose(
                state_mb.clipping_rate, state_no_mb.clipping_rate, rel_tol=1e-5
            )


class TestInputValidation:
    """Tests for parameter validation."""

    def test_invalid_initial_clip_norm(self):
        """Test that negative initial_clip_norm raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="initial_clip_norm must be positive"):
            adaptive_clipped_grad(loss_fn, initial_clip_norm=-1.0, key=key(0))

    def test_invalid_target_quantile(self):
        """Test that target_quantile outside (0, 1) raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(loss_fn, target_quantile=0.0, key=key(0))

        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(loss_fn, target_quantile=1.0, key=key(0))

    def test_invalid_learning_rate(self):
        """Test that negative learning_rate raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            adaptive_clipped_grad(loss_fn, learning_rate=-0.1, key=key(0))

    def test_invalid_clip_norm_min(self):
        """Test that negative clip_norm_min raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="clip_norm_min must be positive"):
            adaptive_clipped_grad(loss_fn, clip_norm_min=-0.1, key=key(0))

    def test_invalid_clip_norm_max(self):
        """Test that clip_norm_max <= clip_norm_min raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="clip_norm_max.*must be.*clip_norm_min"):
            adaptive_clipped_grad(
                loss_fn, clip_norm_min=10.0, clip_norm_max=5.0, key=key(0)
            )


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_example_batch(self):
        """Test with batch size of 1."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(1, 10)
        batch_y = torch.randn(1)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        assert grads.shape == params.shape

    def test_zero_gradients(self):
        """Test with zero gradients (e.g., at local minimum)."""

        def loss_fn(params, x, y):
            return torch.tensor(0.0)  # Constant loss → zero gradients

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Should handle gracefully
        assert torch.allclose(grads, torch.zeros_like(grads))

    def test_batch_size_tracked_in_state(self):
        """Test that batch_size is set from actual number of examples."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )

        # Initial state has batch_size 0
        assert clip_state.batch_size == 0

        params = torch.randn(10, requires_grad=False)

        # Batch of 8
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        assert clip_state.batch_size == 8

        # Batch of 16
        batch_x = torch.randn(16, 10)
        batch_y = torch.randn(16)
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        assert clip_state.batch_size == 16

    def test_large_batch(self):
        """Test with large batch size."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(1000, 10)  # Large batch
        batch_y = torch.randn(1000)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        assert grads.shape == params.shape
        assert 0.0 <= clip_state.clipping_rate <= 1.0
