"""Tests for adaptive gradient clipping (Andrew et al. 2021)."""

import math

import pytest
import torch

from opaque.clipping.adaptive import adaptive_clipped_grad


class TestAdaptiveClippedGrad:
    """Tests for adaptive_clipped_grad function."""

    def test_basic_usage(self):
        """Test basic adaptive clipping workflow."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create adaptive clipping function
        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.5,
            batch_argnums=(1, 2),
        )

        # Check initial state
        assert grad_fn.clip_norm == 1.0
        assert grad_fn.step == 0
        assert grad_fn.clipping_rate == 0.0

        # Compute gradients
        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads = grad_fn(params, batch_x, batch_y)

        # Check state updated
        assert grad_fn.step == 1
        assert isinstance(grad_fn.clip_norm, float)
        assert isinstance(grad_fn.clipping_rate, float)
        assert grads.shape == params.shape

    def test_threshold_increases_when_too_many_clipped(self):
        """Test that threshold increases when clipping rate > target quantile."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Start with very low threshold → many gradients will be clipped
        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.01,  # Very low
            target_quantile=0.5,
            learning_rate=0.2,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        initial_clip_norm = grad_fn.clip_norm

        # First step
        grad_fn(params, batch_x, batch_y)

        # Threshold should increase (many gradients clipped)
        assert grad_fn.clipping_rate > 0.5
        assert grad_fn.clip_norm > initial_clip_norm

    def test_threshold_decreases_when_too_few_clipped(self):
        """Test that threshold decreases when clipping rate < target quantile."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Start with very high threshold → few gradients will be clipped
        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=100.0,  # Very high
            target_quantile=0.5,
            learning_rate=0.2,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        initial_clip_norm = grad_fn.clip_norm

        # First step
        grad_fn(params, batch_x, batch_y)

        # Threshold should decrease (few gradients clipped)
        assert grad_fn.clipping_rate < 0.5
        assert grad_fn.clip_norm < initial_clip_norm

    def test_geometric_update_formula(self):
        """Test that updates follow geometric formula: C_{t+1} = C_t * exp(η * direction)."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        learning_rate = 0.2
        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.5,
            learning_rate=learning_rate,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Step 1
        initial_clip_norm = grad_fn.clip_norm
        grad_fn(params, batch_x, batch_y)
        clipping_rate = grad_fn.clipping_rate

        # Verify geometric update
        if clipping_rate > 0.5:
            expected_clip_norm = initial_clip_norm * math.exp(learning_rate)
        else:
            expected_clip_norm = initial_clip_norm * math.exp(-learning_rate)

        assert abs(grad_fn.clip_norm - expected_clip_norm) < 1e-6

    def test_threshold_clamped_to_bounds(self):
        """Test that threshold is clamped to [clip_norm_min, clip_norm_max]."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        clip_norm_min = 0.5
        clip_norm_max = 2.0

        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.1,  # Below min
            target_quantile=0.5,
            learning_rate=0.2,
            clip_norm_min=clip_norm_min,
            clip_norm_max=clip_norm_max,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Run many steps to push threshold beyond bounds
        for _ in range(20):
            grad_fn(params, batch_x, batch_y)

        # Threshold should be clamped
        assert clip_norm_min <= grad_fn.clip_norm <= clip_norm_max

    def test_step_counter_increments(self):
        """Test that step counter increments correctly."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        assert grad_fn.step == 0

        for i in range(1, 6):
            grad_fn(params, batch_x, batch_y)
            assert grad_fn.step == i

    def test_with_aux_output(self):
        """Test that has_aux=True works correctly."""

        def loss_fn(params, x, y):
            pred = x @ params
            loss = ((pred - y) ** 2).mean()
            # Auxiliary outputs must be tensors for torch.func.grad
            aux = {"accuracy": torch.tensor(0.95)}
            return loss, aux

        grad_fn = adaptive_clipped_grad(
            loss_fn,
            has_aux=True,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Should return only gradients (aux is internal)
        grads = grad_fn(params, batch_x, batch_y)

        assert grads.shape == params.shape
        assert grad_fn.step == 1

    def test_different_target_quantiles(self):
        """Test that different target quantiles affect adaptation."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Low target quantile (aim to clip fewer gradients)
        grad_fn_low = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.1,
            batch_argnums=(1, 2),
        )

        # High target quantile (aim to clip more gradients)
        grad_fn_high = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.9,
            batch_argnums=(1, 2),
        )

        # Run 10 steps
        for _ in range(10):
            grad_fn_low(params, batch_x, batch_y)
            grad_fn_high(params, batch_x, batch_y)

        # Low quantile should result in higher threshold (clip fewer)
        # High quantile should result in lower threshold (clip more)
        assert grad_fn_low.clip_norm > grad_fn_high.clip_norm

    def test_different_learning_rates(self):
        """Test that learning rate affects adaptation speed."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Slow adaptation
        grad_fn_slow = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.01,
            learning_rate=0.05,
            batch_argnums=(1, 2),
        )

        # Fast adaptation
        grad_fn_fast = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.01,
            learning_rate=0.5,
            batch_argnums=(1, 2),
        )

        # Run 5 steps
        for _ in range(5):
            grad_fn_slow(params, batch_x, batch_y)
            grad_fn_fast(params, batch_x, batch_y)

        # Fast should have adapted more
        assert abs(grad_fn_fast.clip_norm - 0.01) > abs(grad_fn_slow.clip_norm - 0.01)

    def test_kwargs_passed_to_clipped_grad(self):
        """Test that additional kwargs are passed to clipped_grad."""
        from opaque.clipping import AuxiliaryOutput

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Test with return_values=True to verify kwargs are passed
        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
            return_values=True,  # Should be passed to clipped_grad
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        result = grad_fn(params, batch_x, batch_y)

        # With return_values=True, should return tuple (grads, AuxiliaryOutput)
        assert isinstance(result, tuple)
        assert len(result) == 2
        grads, aux = result
        assert grads.shape == params.shape
        assert isinstance(aux, AuxiliaryOutput)
        assert aux.values is not None
        assert aux.grad_norms is not None  # We force return_grad_norms=True internally

    def test_composition_with_noise(self):
        """Test that adaptive clipping composes naturally with noise."""
        from opaque.noise import gaussian

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        # Compute clipped gradients
        grads = grad_fn(params, batch_x, batch_y)

        # Add noise scaled to current clip norm
        noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)
        noisy_grads = noise_fn(grads)

        assert noisy_grads.shape == grads.shape
        assert not torch.allclose(noisy_grads, grads)  # Noise added


class TestInputValidation:
    """Tests for parameter validation."""

    def test_invalid_initial_clip_norm(self):
        """Test that negative initial_clip_norm raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="initial_clip_norm must be positive"):
            adaptive_clipped_grad(loss_fn, initial_clip_norm=-1.0)

    def test_invalid_target_quantile(self):
        """Test that target_quantile outside (0, 1) raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(loss_fn, target_quantile=0.0)

        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(loss_fn, target_quantile=1.0)

    def test_invalid_learning_rate(self):
        """Test that negative learning_rate raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            adaptive_clipped_grad(loss_fn, learning_rate=-0.1)

    def test_invalid_clip_norm_min(self):
        """Test that negative clip_norm_min raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="clip_norm_min must be positive"):
            adaptive_clipped_grad(loss_fn, clip_norm_min=-0.1)

    def test_invalid_clip_norm_max(self):
        """Test that clip_norm_max <= clip_norm_min raises error."""

        def loss_fn(params):
            return params.sum()

        with pytest.raises(ValueError, match="clip_norm_max.*must be.*clip_norm_min"):
            adaptive_clipped_grad(loss_fn, clip_norm_min=10.0, clip_norm_max=5.0)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_example_batch(self):
        """Test with batch size of 1."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(1, 10)
        batch_y = torch.randn(1)

        grads = grad_fn(params, batch_x, batch_y)

        assert grads.shape == params.shape
        assert grad_fn.step == 1

    def test_zero_gradients(self):
        """Test with zero gradients (e.g., at local minimum)."""

        def loss_fn(params, x, y):
            return torch.tensor(0.0)  # Constant loss → zero gradients

        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads = grad_fn(params, batch_x, batch_y)

        # Should handle gracefully
        assert torch.allclose(grads, torch.zeros_like(grads))
        assert grad_fn.step == 1

    def test_large_batch(self):
        """Test with large batch size."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        grad_fn = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            batch_argnums=(1, 2),
        )

        params = torch.randn(10, requires_grad=False)
        batch_x = torch.randn(1000, 10)  # Large batch
        batch_y = torch.randn(1000)

        grads = grad_fn(params, batch_x, batch_y)

        assert grads.shape == params.shape
        assert grad_fn.step == 1
        assert 0.0 <= grad_fn.clipping_rate <= 1.0
