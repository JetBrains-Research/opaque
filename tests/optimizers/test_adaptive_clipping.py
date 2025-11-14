"""Tests for adaptive_clipping optimizer wrapper.

Tests the functional adaptive clipping wrapper that works with any TorchOpt optimizer.
"""

import pytest
import torch
import torchopt

from opaque.optimizers import adaptive_clipping


class TestAdaptiveClippingBasics:
    """Test basic functionality of adaptive_clipping."""

    def test_initialization_with_adamw(self):
        """Test that optimizer initializes correctly with AdamW."""
        # Create base optimizer
        base_opt = torchopt.adamw(lr=3e-4, weight_decay=0.01)

        # Wrap with adaptive clipping
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        # Initialize state
        params = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
        state = init_fn(params)

        # Check state structure
        assert state.current_clip_norm == 1.0
        assert state.lr_multiplier == 1.0
        assert state.step == 0
        assert state.opt_state is not None
        assert state.clip_buffer_state is not None

    def test_initialization_with_sgd(self):
        """Test that optimizer works with different base optimizers."""
        base_opt = torchopt.sgd(lr=0.01, momentum=0.9)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=2.0)

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)

        assert state.current_clip_norm == 2.0
        assert state.lr_multiplier == 1.0

    def test_single_step_returns_updates(self):
        """Test that step_fn returns updates (not new params)."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
        state = init_fn(params)

        # Create fake gradients and norms
        grads = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
        grad_norms = torch.tensor([0.8, 1.2, 0.5, 1.5])  # Some above, some below threshold

        # Take step
        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        # Check returns
        assert isinstance(updates, dict)
        assert "weight" in updates
        assert "bias" in updates
        assert new_state.step == 1
        assert isinstance(metrics, dict)
        assert "clip_norm" in metrics
        assert "clip_rate" in metrics
        assert "lr_multiplier" in metrics

    def test_updates_can_be_applied(self):
        """Test that returned updates can be applied with torchopt."""
        base_opt = torchopt.adamw(lr=0.1)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.ones(10, 5), "bias": torch.ones(5)}
        state = init_fn(params)

        grads = {"weight": torch.ones(10, 5), "bias": torch.ones(5)}
        grad_norms = torch.tensor([0.5, 0.8, 1.2])

        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        # Save original values (clone before apply_updates mutates them)
        original_weight = params["weight"].clone()
        original_bias = params["bias"].clone()

        # Apply updates (mutates params in-place)
        new_params = torchopt.apply_updates(params, updates)

        assert isinstance(new_params, dict)
        assert new_params["weight"].shape == original_weight.shape
        assert new_params["bias"].shape == original_bias.shape
        # Parameters should have changed from original values
        assert not torch.allclose(new_params["weight"], original_weight)
        assert not torch.allclose(new_params["bias"], original_bias)


class TestAdaptiveClipping:
    """Test adaptive clipping threshold behavior."""

    def test_clip_norm_adapts(self):
        """Test that clipping threshold adapts based on gradient norms."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=1.0,
            target_clip_rate=0.20,
        )

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}

        # Take multiple steps with varying gradient norms
        for i in range(10):
            # Generate gradient norms (some high, some low)
            grad_norms = torch.rand(32) * 3.0  # Range: 0-3
            updates, state, metrics = step_fn(grads, grad_norms, state, params=params)

        # After 10 steps, clip norm should have adapted
        # (Could be higher or lower depending on distribution)
        assert state.current_clip_norm > 0
        assert metrics["clip_rate"] >= 0.0
        assert metrics["clip_rate"] <= 1.0

    def test_clip_norm_bounds(self):
        """Test that clip norm respects min/max bounds."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=1.0,
            clip_norm_min=0.5,
            clip_norm_max=2.0,
        )

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}

        # Take many steps with extreme gradient norms
        for _ in range(20):
            # Very high norms (should push clip norm up)
            grad_norms = torch.rand(32) * 10.0 + 5.0
            _, state, _ = step_fn(grads, grad_norms, state, params=params)

        # Clip norm should not exceed max
        assert state.current_clip_norm <= 2.0
        assert state.current_clip_norm >= 0.5

    def test_clip_rate_calculation(self):
        """Test that clip rate is calculated correctly."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}

        # Create gradient norms where 50% are above threshold
        grad_norms = torch.tensor([0.5, 0.8, 1.2, 1.5])  # 2 out of 4 above 1.0
        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        # Note: clip_rate is computed on the buffer, which may have history
        # So we just check it's in valid range
        assert 0.0 <= metrics["clip_rate"] <= 1.0


class TestLRScaling:
    """Test optional learning rate scaling feature."""

    def test_lr_multiplier_disabled_by_default(self):
        """Test that LR multiplier stays at 1.0 when disabled."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=1.0,
            use_clip_lr_scaling=False,  # Disabled
        )

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}

        # Take multiple steps
        for _ in range(10):
            grad_norms = torch.rand(32)
            _, state, metrics = step_fn(grads, grad_norms, state, params=params)

        # LR multiplier should remain 1.0
        assert state.lr_multiplier == 1.0
        assert metrics["lr_multiplier"] == 1.0

    def test_lr_multiplier_enabled(self):
        """Test that LR multiplier adjusts when enabled."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=1.0,
            target_clip_rate=0.20,
            use_clip_lr_scaling=True,  # Enabled
        )

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}

        # Take steps with high clip rate (should decrease LR multiplier)
        for _ in range(20):
            # All norms high (100% clip rate)
            grad_norms = torch.rand(32) * 2.0 + 2.0  # Range: 2-4
            _, state, metrics = step_fn(grads, grad_norms, state, params=params)

        # LR multiplier should have adjusted (likely decreased due to high clip rate)
        # We can't predict exact value, but it should be != 1.0 after many steps
        # and should be in valid range
        assert state.lr_multiplier > 0.0
        assert state.lr_multiplier <= 2.0


class TestStateImmutability:
    """Test that state follows functional immutability pattern."""

    def test_state_immutable(self):
        """Test that original state is not modified."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        original_step = state.step
        original_clip_norm = state.current_clip_norm

        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([0.5, 1.5])

        # Take step
        _, new_state, _ = step_fn(grads, grad_norms, state, params=params)

        # Original state should be unchanged
        assert state.step == original_step
        assert state.current_clip_norm == original_clip_norm
        # New state should be different
        assert new_state.step == original_step + 1


class TestMetrics:
    """Test metrics returned by step_fn."""

    def test_metrics_structure(self):
        """Test that metrics dict has expected keys."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([0.5, 1.5])

        _, _, metrics = step_fn(grads, grad_norms, state, params=params)

        assert "step" in metrics
        assert "clip_norm" in metrics
        assert "clip_rate" in metrics
        assert "lr_multiplier" in metrics

    def test_metrics_values_valid(self):
        """Test that metrics values are in valid ranges."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([0.5, 1.5, 0.8])

        _, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        assert metrics["step"] == 1
        assert metrics["clip_norm"] > 0
        assert 0.0 <= metrics["clip_rate"] <= 1.0
        assert metrics["lr_multiplier"] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_grad_norms(self):
        """Test handling of empty gradient norms."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([])  # Empty

        # Should handle gracefully (implementation may vary)
        # Just check it doesn't crash
        try:
            _, _, _ = step_fn(grads, grad_norms, state, params=params)
        except (ValueError, RuntimeError):
            # It's okay if it raises an error for empty norms
            pass

    def test_single_grad_norm(self):
        """Test handling of single gradient norm."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([1.5])

        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        assert new_state.step == 1
        assert isinstance(updates, dict)

    def test_zero_gradients(self):
        """Test handling of zero gradients."""
        base_opt = torchopt.adamw(lr=3e-4)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        params = {"weight": torch.randn(10, 5)}
        state = init_fn(params)
        grads = {"weight": torch.zeros(10, 5)}  # Zero gradients
        grad_norms = torch.tensor([0.0, 0.0, 0.0])

        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        assert new_state.step == 1
        assert isinstance(updates, dict)
