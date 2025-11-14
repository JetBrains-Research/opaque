"""Tests for adaptive_clipping optimizer wrapper.

Tests verify that adaptive_clipping:
1. Wraps any TorchOpt base optimizer correctly
2. Adapts clipping threshold based on gradient distribution
3. Optionally scales learning rate based on clip rate
4. Maintains functional purity (immutable state)
5. Returns correct metrics for monitoring
"""

import pytest
import torch
import torchopt

from opaque import clipped_grad, add_gaussian_noise
from opaque.optimizers import DPAdaptiveClipState, adaptive_clipping


class TestInitialization:
    """Test optimizer initialization."""

    def test_init_with_adamw(self):
        """Test initialization with AdamW base optimizer."""
        params = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}

        base_opt = torchopt.adamw(lr=0.01, weight_decay=0.01)
        init_fn, _ = adaptive_clipping(base_opt, initial_clip_norm=1.0)

        state = init_fn(params)

        assert isinstance(state, DPAdaptiveClipState)
        assert state.step == 0
        assert state.current_clip_norm == 1.0
        assert state.lr_multiplier == 1.0
        assert isinstance(state.clip_buffer_state, tuple)
        assert len(state.clip_buffer_state) == 2  # (norms_tensor, size)

    def test_init_with_sgd(self):
        """Test initialization with SGD base optimizer."""
        params = {"weight": torch.randn(10, 5)}

        base_opt = torchopt.sgd(lr=0.1, momentum=0.9)
        init_fn, _ = adaptive_clipping(base_opt, initial_clip_norm=2.0)

        state = init_fn(params)
        assert isinstance(state, DPAdaptiveClipState)
        assert state.current_clip_norm == 2.0

    def test_init_with_adam(self):
        """Test initialization with Adam base optimizer."""
        params = {"weight": torch.randn(10, 5)}

        base_opt = torchopt.adam(lr=0.001)
        init_fn, _ = adaptive_clipping(base_opt)

        state = init_fn(params)
        assert isinstance(state, DPAdaptiveClipState)


class TestStepFunction:
    """Test optimizer step function."""

    def test_step_basic(self):
        """Test basic optimizer step."""
        params = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
        grads = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
        grad_norms = torch.tensor([1.0, 1.5, 0.8, 2.0])

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)
        state = init_fn(params)

        # Step returns updates, not new params
        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        # Check state updates
        assert new_state.step == 1
        assert new_state.current_clip_norm > 0
        assert isinstance(updates, dict)
        assert "weight" in updates
        assert "bias" in updates

        # Check metrics
        assert "clip_norm" in metrics
        assert "clip_rate" in metrics
        assert "lr_multiplier" in metrics

    def test_step_immutability(self):
        """Test that step doesn't modify original state."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([1.0, 1.5])

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(base_opt)
        state = init_fn(params)

        original_step = state.step
        original_clip_norm = state.current_clip_norm

        # Call step
        _, new_state, _ = step_fn(grads, grad_norms, state, params=params)

        # Original state unchanged
        assert state.step == original_step
        assert state.current_clip_norm == original_clip_norm
        # New state updated
        assert new_state.step == original_step + 1

    def test_step_updates_clip_norm(self):
        """Test that clip norm adapts based on gradient norms."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}

        # Simulate gradients with norms much larger than initial clip norm
        grad_norms = torch.tensor([5.0, 6.0, 7.0, 8.0])

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=1.0,
            target_clip_rate=0.20,
        )
        state = init_fn(params)

        # Take multiple steps to fill buffer
        for _ in range(10):
            _, state, _ = step_fn(grads, grad_norms, state, params=params)

        # Clip norm should have adapted upward
        assert state.current_clip_norm > 1.0


class TestAdaptiveClipping:
    """Test adaptive clipping behavior."""

    def test_clip_rate_converges_to_target(self):
        """Test that clip rate converges toward target."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}

        target_clip_rate = 0.20

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=1.0,
            target_clip_rate=target_clip_rate,
        )
        state = init_fn(params)

        # Take many steps with varying gradient norms
        clip_rates = []
        for i in range(50):
            # Vary norms to simulate real training
            grad_norms = torch.randn(4).abs() * (2.0 + 0.5 * torch.sin(torch.tensor(i * 0.1)))
            _, state, metrics = step_fn(grads, grad_norms, state, params=params)
            clip_rates.append(metrics["clip_rate"])

        # After enough steps, clip rate should be near target
        recent_avg_clip_rate = sum(clip_rates[-10:]) / 10
        assert abs(recent_avg_clip_rate - target_clip_rate) < 0.15

    def test_clip_norm_respects_bounds(self):
        """Test that clip norm stays within specified bounds."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}

        clip_norm_min = 0.5
        clip_norm_max = 3.0

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            initial_clip_norm=1.0,
            clip_norm_min=clip_norm_min,
            clip_norm_max=clip_norm_max,
        )
        state = init_fn(params)

        # Take steps with extreme gradient norms
        for _ in range(20):
            # Very large norms
            grad_norms = torch.tensor([10.0, 15.0, 20.0])
            _, state, _ = step_fn(grads, grad_norms, state, params=params)

        assert clip_norm_min <= state.current_clip_norm <= clip_norm_max


class TestLRScaling:
    """Test learning rate scaling based on clip rate."""

    def test_lr_multiplier_disabled_by_default(self):
        """Test that LR multiplier stays at 1.0 when scaling disabled."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([1.0, 2.0, 3.0])

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            use_clip_lr_scaling=False,  # Disabled
        )
        state = init_fn(params)

        # Take several steps
        for _ in range(10):
            _, state, metrics = step_fn(grads, grad_norms, state, params=params)
            # LR multiplier should always be 1.0
            assert metrics["lr_multiplier"] == 1.0

    def test_lr_multiplier_enabled(self):
        """Test that LR multiplier changes when scaling enabled."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(
            base_opt,
            target_clip_rate=0.20,
            use_clip_lr_scaling=True,  # Enabled
        )
        state = init_fn(params)

        # Take steps with very high clip rate (should decrease LR)
        for _ in range(20):
            # Norms well below initial clip norm → low clip rate → increase LR
            grad_norms = torch.tensor([0.1, 0.2, 0.15, 0.1])
            _, state, _ = step_fn(grads, grad_norms, state, params=params)

        # LR multiplier should have changed from initial value
        assert state.lr_multiplier != 1.0


class TestDifferentOptimizers:
    """Test that adaptive_clipping works with different base optimizers."""

    def test_works_with_sgd(self):
        """Test that it works with SGD."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([1.0, 1.5])

        base_opt = torchopt.sgd(lr=0.1, momentum=0.9)
        init_fn, step_fn = adaptive_clipping(base_opt)
        state = init_fn(params)

        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        assert isinstance(updates, dict)
        assert new_state.step == 1

    def test_works_with_adam(self):
        """Test that it works with Adam."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([1.0, 1.5])

        base_opt = torchopt.adam(lr=0.001)
        init_fn, step_fn = adaptive_clipping(base_opt)
        state = init_fn(params)

        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        assert isinstance(updates, dict)
        assert new_state.step == 1

    def test_works_with_adamw(self):
        """Test that it works with AdamW."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([1.0, 1.5])

        base_opt = torchopt.adamw(lr=0.001, weight_decay=0.01)
        init_fn, step_fn = adaptive_clipping(base_opt)
        state = init_fn(params)

        updates, new_state, metrics = step_fn(grads, grad_norms, state, params=params)

        assert isinstance(updates, dict)
        assert new_state.step == 1


class TestMetrics:
    """Test metrics returned by optimizer."""

    def test_metrics_structure(self):
        """Test that all expected metrics are present."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([1.0, 1.5, 0.8])

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(base_opt)
        state = init_fn(params)

        _, _, metrics = step_fn(grads, grad_norms, state, params=params)

        required_keys = ["clip_norm", "clip_rate", "lr_multiplier"]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_metrics_values_valid(self):
        """Test that metric values are valid."""
        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5)}
        grad_norms = torch.tensor([1.0, 1.5, 0.8])

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(base_opt)
        state = init_fn(params)

        _, _, metrics = step_fn(grads, grad_norms, state, params=params)

        assert metrics["clip_norm"] > 0
        assert 0 <= metrics["clip_rate"] <= 1
        assert metrics["lr_multiplier"] > 0


class TestIntegrationWithClippedGrad:
    """Test integration with clipped_grad function."""

    def test_complete_dp_training_step(self):
        """Test complete DP training step with external noise."""
        # Simple linear model
        params = {"weight": torch.randn(5, 3), "bias": torch.randn(3)}

        def loss_fn(params, x, y):
            logits = x @ params["weight"] + params["bias"]
            return ((logits - y) ** 2).mean()

        # Create optimizer
        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)
        state = init_fn(params)

        # Create data
        x = torch.randn(4, 5)  # batch of 4
        y = torch.randn(4, 3)

        # 1. Compute clipped gradients with norms
        clipped_grad_fn = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            l2_clip_norm=state.current_clip_norm,
            return_grad_norms=True,
        )
        grads, aux = clipped_grad_fn(params, x, y)
        grad_norms = aux.grad_norms

        # 2. Add DP noise (EXTERNAL)
        noise_multiplier = 1.1
        stddev = noise_multiplier * state.current_clip_norm
        rng = torch.Generator().manual_seed(42)
        noisy_grads = add_gaussian_noise(grads, stddev=stddev, generator=rng)

        # 3. Optimizer step
        updates, new_state, metrics = step_fn(noisy_grads, grad_norms, state, params=params)

        # 4. Apply updates
        new_params = torchopt.apply_updates(params, updates)

        # Verify
        assert isinstance(new_params, dict)
        assert new_state.step == 1
        assert "clip_norm" in metrics

    def test_multi_step_training(self):
        """Test multiple training steps."""
        params = {"weight": torch.randn(5, 3)}

        def loss_fn(params, x, y):
            return ((x @ params["weight"] - y) ** 2).mean()

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, step_fn = adaptive_clipping(base_opt)
        state = init_fn(params)

        rng = torch.Generator().manual_seed(42)

        for step in range(10):
            x = torch.randn(4, 5)
            y = torch.randn(4, 3)

            # Compute clipped gradients
            clipped_grad_fn = clipped_grad(
                loss_fn,
                argnums=0,
                batch_argnums=(1, 2),
                l2_clip_norm=state.current_clip_norm,
                return_grad_norms=True,
            )
            grads, aux = clipped_grad_fn(params, x, y)

            # Add noise
            noisy_grads = add_gaussian_noise(
                grads,
                stddev=1.1 * state.current_clip_norm,
                generator=rng,
            )

            # Step
            updates, state, metrics = step_fn(noisy_grads, aux.grad_norms, state, params=params)
            params = torchopt.apply_updates(params, updates)

            assert state.step == step + 1


class TestStateImmutability:
    """Test that state is truly immutable."""

    def test_state_is_namedtuple(self):
        """Test that state is a NamedTuple (immutable)."""
        params = {"weight": torch.randn(10, 5)}

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, _ = adaptive_clipping(base_opt)
        state = init_fn(params)

        # Should be a NamedTuple
        assert isinstance(state, DPAdaptiveClipState)
        # NamedTuples have _fields
        assert hasattr(state, "_fields")

    def test_state_cannot_be_modified(self):
        """Test that attempting to modify state fails."""
        params = {"weight": torch.randn(10, 5)}

        base_opt = torchopt.adamw(lr=0.01)
        init_fn, _ = adaptive_clipping(base_opt)
        state = init_fn(params)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            state.step = 999
