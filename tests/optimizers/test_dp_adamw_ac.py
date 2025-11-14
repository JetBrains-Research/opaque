"""Tests for DP-AdamW-AC (Adaptive Clipping) optimizer."""

import pytest
import torch

from opaque.adaptive import ClipNormBuffer
from opaque.optimizers import AdaptiveClipState, dp_adamw_ac


class TestDPAdamWACCreation:
    """Tests for DP-AdamW-AC optimizer creation."""

    def test_creates_init_and_step_functions(self):
        """Test that dp_adamw_ac returns init_fn and step_fn."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        assert callable(init_fn)
        assert callable(step_fn)

    def test_default_parameters(self):
        """Test DP-AdamW-AC with default parameters."""
        init_fn, _ = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, AdaptiveClipState)
        assert state.step == 0
        assert state.current_clip_norm == 3.0
        assert state.lr_multiplier == 1.0
        assert isinstance(state.clip_buffer, ClipNormBuffer)

    def test_custom_parameters(self):
        """Test DP-AdamW-AC with custom parameters."""
        init_fn, _ = dp_adamw_ac(
            learning_rate=1e-4,
            initial_clip_norm=5.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            target_clip_rate=0.30,
            history_size=500,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, AdaptiveClipState)
        assert state.current_clip_norm == 5.0
        assert state.clip_buffer.target_clip_rate == 0.30
        assert state.clip_buffer.capacity == 500


class TestDPAdamWACState:
    """Tests for AdaptiveClipState."""

    def test_state_structure(self):
        """Test that state has all required fields."""
        init_fn, _ = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        # Check all fields exist
        assert hasattr(state, "opt_state")
        assert hasattr(state, "accountant")
        assert hasattr(state, "noise_gen")
        assert hasattr(state, "clip_buffer")
        assert hasattr(state, "current_clip_norm")
        assert hasattr(state, "lr_multiplier")
        assert hasattr(state, "ema_params")
        assert hasattr(state, "step")

    def test_ema_params_initialized(self):
        """Test that EMA parameters are initialized correctly."""
        init_fn, _ = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.ones(5, 3)}
        state = init_fn(params)

        # EMA params should be a copy of initial params
        assert torch.allclose(state.ema_params["weight"], params["weight"])

    def test_clip_buffer_initialized(self):
        """Test that clip buffer is properly initialized."""
        init_fn, _ = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            target_clip_rate=0.25,
            history_size=800,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert len(state.clip_buffer) == 0  # Empty initially
        assert state.clip_buffer.capacity == 800
        assert state.clip_buffer.target_clip_rate == 0.25


class TestDPAdamWACOptimization:
    """Tests for DP-AdamW-AC optimization behavior."""

    def test_single_step(self):
        """Test a single DP-AdamW-AC optimization step."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.ones(5, 3)}
        original_weight = params["weight"].clone()
        grads = {"weight": torch.ones(5, 3) * 0.5}
        grad_norms = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])

        state = init_fn(params)
        # batch_sizes is optional - defaults to ones
        new_params, new_state, metrics = step_fn(params, grads, grad_norms, state)

        # Parameters should change
        assert not torch.allclose(new_params["weight"], original_weight)

        # State should update
        assert new_state.step == 1
        assert metrics["step"] == 1
        assert "epsilon" in metrics
        assert "clip_norm" in metrics
        assert "clip_rate" in metrics
        assert "lr_multiplier" in metrics

    def test_multiple_steps(self):
        """Test multiple DP-AdamW-AC steps."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.ones(5, 3)}
        grads = {"weight": torch.ones(5, 3) * 0.5}
        grad_norms = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
        batch_sizes = torch.ones(5)

        state = init_fn(params)

        # Run 5 steps
        for i in range(5):
            params, state, metrics = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)
            assert state.step == i + 1
            assert metrics["epsilon"] > 0


class TestDPAdamWACAdaptiveClipping:
    """Tests for adaptive clipping functionality."""

    def test_clip_norm_adapts_to_gradient_distribution(self):
        """Test that clip norm adapts based on gradient norms."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            target_clip_rate=0.20,
            history_size=100,
        )

        params = {"weight": torch.ones(10)}
        grads = {"weight": torch.ones(10) * 0.1}

        # Small gradient norms (most around 1.0)
        grad_norms = torch.ones(5) * 1.0

        state = init_fn(params)
        initial_clip_norm = state.current_clip_norm

        # Run many steps to build up history (batch_sizes defaults to ones)
        for _ in range(30):
            params, state, metrics = step_fn(params, grads, grad_norms, state)

        # Clip norm should adapt down toward 1.0 (80th percentile of 1.0)
        assert state.current_clip_norm < initial_clip_norm

    def test_clip_norm_increases_for_large_gradients(self):
        """Test that clip norm increases when gradients are large."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=1.0,  # Start low
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            target_clip_rate=0.20,
            history_size=100,
        )

        params = {"weight": torch.ones(10)}
        grads = {"weight": torch.ones(10) * 0.5}

        # Large gradient norms (most around 5.0)
        grad_norms = torch.ones(5) * 5.0
        state = init_fn(params)
        initial_clip_norm = state.current_clip_norm

        # Run many steps (batch_sizes defaults to ones)
        for _ in range(30):
            params, state, metrics = step_fn(params, grads, grad_norms, state)

        # Clip norm should adapt up toward 5.0
        assert state.current_clip_norm > initial_clip_norm

    def test_clip_norm_respects_bounds(self):
        """Test that clip norm stays within min/max bounds."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            clip_norm_min=1.0,
            clip_norm_max=5.0,
        )

        params = {"weight": torch.ones(10)}
        grads = {"weight": torch.ones(10) * 0.1}

        # Very small norms
        small_norms = torch.ones(5) * 0.1
        # Very large norms
        large_norms = torch.ones(5) * 10.0
        batch_sizes = torch.ones(5)

        state = init_fn(params)

        # Run with small norms
        for _ in range(20):
            params, state, _ = step_fn(params, grads, small_norms, state)

        # Should be clamped to min
        assert state.current_clip_norm >= 1.0

        # Run with large norms
        for _ in range(20):
            params, state, _ = step_fn(params, grads, large_norms, state)

        # Should be clamped to max
        assert state.current_clip_norm <= 5.0


class TestDPAdamWACLearningRateAdjustment:
    """Tests for learning rate adjustment based on clip rate."""

    def test_lr_increases_when_clip_rate_low(self):
        """Test that LR multiplier increases when clip rate is low."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=10.0,  # High C so nothing gets clipped
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            target_clip_rate=0.20,
            history_size=50,
        )

        params = {"weight": torch.ones(10)}
        grads = {"weight": torch.ones(10) * 0.1}

        # Small gradient norms (won't be clipped by high C)
        grad_norms = torch.ones(5) * 1.0
        batch_sizes = torch.ones(5)

        state = init_fn(params)
        initial_lr_mult = state.lr_multiplier

        # Run steps (clip rate will be very low)
        for _ in range(20):
            params, state, _ = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # LR multiplier should increase
        assert state.lr_multiplier > initial_lr_mult

    def test_lr_decreases_when_clip_rate_high(self):
        """Test that LR multiplier decreases when clip rate is high."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,  # Fixed clip norm
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            target_clip_rate=0.20,
            history_size=50,
            clip_norm_min=3.0,  # Prevent adaptation
            clip_norm_max=3.0,  # Keep C fixed at 3.0
        )

        params = {"weight": torch.ones(10)}
        grads = {"weight": torch.ones(10) * 0.5}

        # Large gradient norms (will be clipped by C=3.0, so clip rate > 0.20)
        grad_norms = torch.ones(5) * 5.0
        batch_sizes = torch.ones(5)

        state = init_fn(params)
        initial_lr_mult = state.lr_multiplier

        # Run steps (clip rate will be consistently high since norms > C)
        for _ in range(20):
            params, state, _ = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # LR multiplier should decrease (high clip rate → decrease LR)
        assert state.lr_multiplier < initial_lr_mult

    def test_lr_respects_bounds(self):
        """Test that LR multiplier stays within bounds."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            lr_multiplier_min=0.5,
            lr_multiplier_max=1.5,
        )

        params = {"weight": torch.ones(10)}
        grads = {"weight": torch.ones(10) * 0.1}
        grad_norms = torch.ones(5) * 1.0
        batch_sizes = torch.ones(5)

        state = init_fn(params)

        # Run many steps
        for _ in range(50):
            params, state, _ = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # Should stay within bounds
        assert 0.5 <= state.lr_multiplier <= 1.5


class TestDPAdamWACEMA:
    """Tests for EMA parameter updates."""

    def test_ema_params_update(self):
        """Test that EMA parameters are updated."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            ema_decay=0.9,
        )

        params = {"weight": torch.ones(5)}
        original_ema = params["weight"].clone()
        grads = {"weight": torch.ones(5) * 0.5}
        grad_norms = torch.ones(3) * 1.0
        batch_sizes = torch.ones(3)

        state = init_fn(params)
        params, state, _ = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # EMA params should be different from original
        assert not torch.allclose(state.ema_params["weight"], original_ema)

    def test_ema_smooths_parameters(self):
        """Test that EMA parameters are smoother than current params."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            ema_decay=0.99,  # High decay for strong smoothing
        )

        params = {"weight": torch.zeros(10)}
        grads = {"weight": torch.ones(10) * 0.5}
        grad_norms = torch.ones(3) * 1.0
        batch_sizes = torch.ones(3)

        state = init_fn(params)

        # Run several steps
        for _ in range(10):
            params, state, _ = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # Current params should move more than EMA
        ema_change = torch.abs(state.ema_params["weight"]).mean()
        param_change = torch.abs(params["weight"]).mean()

        # EMA should lag behind current params
        assert ema_change < param_change


class TestDPAdamWACMetrics:
    """Tests for metrics returned by DP-AdamW-AC."""

    def test_metrics_structure(self):
        """Test that metrics have all required fields."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}
        grad_norms = torch.ones(5)
        batch_sizes = torch.ones(5)

        state = init_fn(params)
        _, _, metrics = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # Check all fields
        assert "epsilon" in metrics
        assert "delta" in metrics
        assert "step" in metrics
        assert "clip_norm" in metrics
        assert "clip_rate" in metrics
        assert "lr_multiplier" in metrics
        assert "effective_lr" in metrics

    def test_effective_lr_computation(self):
        """Test that effective_lr = base_lr * lr_multiplier."""
        base_lr = 3e-4

        init_fn, step_fn = dp_adamw_ac(
            learning_rate=base_lr,
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}
        grad_norms = torch.ones(5)
        batch_sizes = torch.ones(5)

        state = init_fn(params)
        _, state, metrics = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # Check effective LR
        expected_effective_lr = base_lr * metrics["lr_multiplier"]
        assert metrics["effective_lr"] == pytest.approx(expected_effective_lr)


class TestDPAdamWACReproducibility:
    """Tests for reproducibility."""

    def test_seed_reproducibility(self):
        """Test that same seed gives reproducible results."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}
        grad_norms = torch.ones(5)
        batch_sizes = torch.ones(5)

        # First run
        state1 = init_fn(params)
        new_params1, _, _ = step_fn(params, grads, grad_norms, state1)

        # Second run
        state2 = init_fn(params)
        new_params2, _, _ = step_fn(params, grads, grad_norms, state2)

        # Should be identical
        assert torch.allclose(new_params1["weight"], new_params2["weight"])


class TestDPAdamWACPrivacy:
    """Tests for privacy accounting."""

    def test_privacy_cost_increases(self):
        """Test that epsilon increases with steps."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}
        grad_norms = torch.ones(5)
        batch_sizes = torch.ones(5)

        state = init_fn(params)
        epsilons = []

        for _ in range(10):
            params, state, metrics = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)
            epsilons.append(metrics["epsilon"])

        # Should be monotonically increasing
        for i in range(len(epsilons) - 1):
            assert epsilons[i + 1] > epsilons[i]


class TestDPAdamWACEdgeCases:
    """Tests for edge cases."""

    def test_empty_gradient_norms(self):
        """Test behavior when given zero gradients."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.zeros(10)}
        original_weight = params["weight"].clone()
        grads = {"weight": torch.zeros(10)}
        grad_norms = torch.zeros(5)  # Zero norms
        batch_sizes = torch.ones(5)

        state = init_fn(params)
        new_params, _, metrics = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # Should still work (noise will cause changes)
        assert not torch.allclose(new_params["weight"], original_weight, atol=1e-6)
        assert metrics["step"] == 1

    def test_varying_batch_sizes(self):
        """Test with varying batch sizes (microbatching)."""
        init_fn, step_fn = dp_adamw_ac(
            initial_clip_norm=3.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.ones(10)}
        grads = {"weight": torch.ones(10) * 0.1}
        grad_norms = torch.tensor([2.0, 4.0, 3.0])
        batch_sizes = torch.tensor([2.0, 4.0, 3.0])  # Different sizes

        state = init_fn(params)
        new_params, new_state, metrics = step_fn(params, grads, grad_norms, state, batch_sizes=batch_sizes)

        # Should work correctly (unit normalization handles varying sizes)
        assert new_state.step == 1
        assert len(new_state.clip_buffer) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
