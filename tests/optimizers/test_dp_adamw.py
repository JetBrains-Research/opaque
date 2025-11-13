"""Tests for DP-AdamW optimizer."""

import pytest
import torch

from opaque.optimizers import DPOptimizerState, dp_adamw


class TestDPAdamWCreation:
    """Tests for DP-AdamW optimizer creation."""

    def test_creates_init_and_step_functions(self):
        """Test that dp_adamw returns init_fn and step_fn."""
        init_fn, step_fn = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        assert callable(init_fn)
        assert callable(step_fn)

    def test_default_parameters(self):
        """Test DP-AdamW with default parameters."""
        init_fn, step_fn = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)
        assert state.step == 0

    def test_custom_learning_rate(self):
        """Test DP-AdamW with custom learning rate."""
        init_fn, step_fn = dp_adamw(
            learning_rate=1e-4,  # Custom LR
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)

    def test_custom_weight_decay(self):
        """Test DP-AdamW with custom weight decay."""
        init_fn, step_fn = dp_adamw(
            weight_decay=0.1,  # Custom weight decay
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)

    def test_custom_betas(self):
        """Test DP-AdamW with custom beta parameters."""
        init_fn, step_fn = dp_adamw(
            betas=(0.95, 0.995),  # Custom betas
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)

    def test_custom_eps(self):
        """Test DP-AdamW with custom eps parameter."""
        init_fn, step_fn = dp_adamw(
            eps=1e-10,  # Custom eps
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)


class TestDPAdamWOptimization:
    """Tests for DP-AdamW optimization behavior."""

    def test_single_step(self):
        """Test a single DP-AdamW optimization step."""
        init_fn, step_fn = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.ones(5, 3)}
        original_weight = params["weight"].clone()
        grads = {"weight": torch.ones(5, 3) * 0.5}

        state = init_fn(params)
        new_params, new_state, metrics = step_fn(params, grads, state)

        # Parameters should change
        assert not torch.allclose(new_params["weight"], original_weight)

        # State should update
        assert new_state.step == 1
        assert metrics["step"] == 1
        assert metrics["epsilon"] > 0

    def test_multiple_steps(self):
        """Test multiple DP-AdamW steps."""
        init_fn, step_fn = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.ones(5, 3)}
        grads = {"weight": torch.ones(5, 3) * 0.5}

        state = init_fn(params)

        # Run 5 steps
        for i in range(5):
            params, state, metrics = step_fn(params, grads, state)
            assert state.step == i + 1
            assert metrics["epsilon"] > 0

    def test_convergence_on_simple_problem(self):
        """Test that DP-AdamW can minimize a simple quadratic loss."""
        init_fn, step_fn = dp_adamw(
            learning_rate=0.1,
            weight_decay=0.0,  # Disable weight decay for convergence test
            l2_clip_norm=1.0,
            noise_multiplier=0.1,  # Low noise for convergence test
            sample_rate=0.01,
            target_delta=1e-5,
        )

        # Minimize ||w - target||^2
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"w": torch.zeros(3)}

        state = init_fn(params)

        # Run optimization
        for _ in range(20):
            # Gradient of ||w - target||^2 = 2(w - target)
            grad = {"w": 2 * (params["w"] - target)}
            params, state, _ = step_fn(params, grad, state)

        # Should move toward target (won't reach exactly due to noise)
        assert torch.norm(params["w"] - target) < 2.0  # Started at distance 3.74

    def test_weight_decay_regularization(self):
        """Test that weight decay pulls parameters toward zero."""
        init_fn, step_fn = dp_adamw(
            learning_rate=0.01,
            weight_decay=0.1,  # Strong weight decay
            l2_clip_norm=1.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        # Start with non-zero weights
        params = {"w": torch.ones(10) * 2.0}
        state = init_fn(params)

        # Apply zero gradients - only weight decay should affect parameters
        for _ in range(20):
            grad = {"w": torch.zeros(10)}
            params, state, _ = step_fn(params, grad, state)

        # Weight decay should pull parameters toward zero
        # (not exactly zero due to noise and momentum)
        assert torch.abs(params["w"]).mean() < 2.0  # Started at 2.0


class TestDPAdamWPrivacy:
    """Tests for privacy accounting in DP-AdamW."""

    def test_privacy_cost_increases(self):
        """Test that privacy cost increases with steps."""
        init_fn, step_fn = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}

        state = init_fn(params)
        epsilons = []

        # Run 10 steps
        for _ in range(10):
            params, state, metrics = step_fn(params, grads, state)
            epsilons.append(metrics["epsilon"])

        # Should be monotonically increasing
        for i in range(len(epsilons) - 1):
            assert epsilons[i + 1] > epsilons[i]

    def test_higher_noise_gives_better_privacy(self):
        """Test that higher noise multiplier gives better (lower) epsilon."""
        # Low noise
        init_fn1, step_fn1 = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=0.5,  # Low noise
            sample_rate=0.01,
            target_delta=1e-5,
        )

        # High noise
        init_fn2, step_fn2 = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=2.0,  # High noise
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}

        # Run 10 steps
        state1 = init_fn1(params)
        state2 = init_fn2(params)

        for _ in range(10):
            params_copy = {"weight": params["weight"].clone()}
            grads_copy = {"weight": grads["weight"].clone()}

            _, state1, metrics1 = step_fn1(params_copy, grads_copy, state1)
            _, state2, metrics2 = step_fn2(params, grads, state2)

        # Higher noise → better privacy (lower epsilon)
        assert metrics2["epsilon"] < metrics1["epsilon"]

    def test_rdp_accountant(self):
        """Test using RDP accountant."""
        init_fn, _ = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            accountant_type="rdp",
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        from opaque.accounting import RDPAccountant

        assert isinstance(state.accountant, RDPAccountant)

    def test_pld_accountant(self):
        """Test using PLD accountant."""
        init_fn, _ = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            accountant_type="pld",
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        from opaque.accounting import PLDAccountant

        assert isinstance(state.accountant, PLDAccountant)


class TestDPAdamWReproducibility:
    """Tests for reproducibility in DP-AdamW."""

    def test_seed_reproducibility(self):
        """Test that same seed gives reproducible results."""
        init_fn, step_fn = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}

        # First run
        state1 = init_fn(params)
        new_params1, _, _ = step_fn(params, grads, state1)

        # Second run with same seed
        state2 = init_fn(params)
        new_params2, _, _ = step_fn(params, grads, state2)

        # Should be identical
        assert torch.allclose(new_params1["weight"], new_params2["weight"])

    def test_different_seeds_differ(self):
        """Test that different seeds give different results."""
        # Seed 42
        init_fn1, step_fn1 = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        # Seed 43
        init_fn2, step_fn2 = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=43,
        )

        params1 = {"weight": torch.randn(5, 3)}
        params2 = {"weight": params1["weight"].clone()}
        grads1 = {"weight": torch.randn(5, 3)}
        grads2 = {"weight": grads1["weight"].clone()}

        state1 = init_fn1(params1)
        new_params1, _, _ = step_fn1(params1, grads1, state1)

        state2 = init_fn2(params2)
        new_params2, _, _ = step_fn2(params2, grads2, state2)

        # Should be different due to different noise
        assert not torch.allclose(new_params1["weight"], new_params2["weight"])


class TestDPAdamWPyTree:
    """Tests for PyTree parameter support."""

    def test_nested_pytree(self):
        """Test with nested PyTree parameters."""
        init_fn, step_fn = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {
            "layer1": {"weight": torch.randn(5, 3), "bias": torch.randn(3)},
            "layer2": {"weight": torch.randn(3, 2)},
        }
        original_weight = params["layer1"]["weight"].clone()

        grads = {
            "layer1": {"weight": torch.randn(5, 3), "bias": torch.randn(3)},
            "layer2": {"weight": torch.randn(3, 2)},
        }

        state = init_fn(params)
        new_params, _, _ = step_fn(params, grads, state)

        # Structure preserved
        assert "layer1" in new_params
        assert "layer2" in new_params
        assert "weight" in new_params["layer1"]

        # Updates applied
        assert not torch.allclose(new_params["layer1"]["weight"], original_weight)


class TestDPAdamWComparison:
    """Tests comparing DP-AdamW to DP-Adam behavior."""

    def test_adamw_vs_adam_weight_decay(self):
        """Test that AdamW applies weight decay correctly."""
        # DP-AdamW with weight decay
        init_adamw, step_adamw = dp_adamw(
            learning_rate=0.01,
            weight_decay=0.1,
            l2_clip_norm=1.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        # DP-Adam (no weight decay)
        from opaque.optimizers import dp_adam

        init_adam, step_adam = dp_adam(
            learning_rate=0.01,
            l2_clip_norm=1.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        # Start with non-zero weights
        params_adamw = {"w": torch.ones(10) * 2.0}
        params_adam = {"w": torch.ones(10) * 2.0}

        state_adamw = init_adamw(params_adamw)
        state_adam = init_adam(params_adam)

        # Apply zero gradients
        for _ in range(20):
            grad = {"w": torch.zeros(10)}
            params_adamw, state_adamw, _ = step_adamw(params_adamw, grad, state_adamw)
            params_adam, state_adam, _ = step_adam(params_adam, grad, state_adam)

        # AdamW should have smaller magnitude (weight decay pulls toward zero)
        # Adam should stay close to original (only noise affects it)
        assert torch.abs(params_adamw["w"]).mean() < torch.abs(params_adam["w"]).mean()


class TestDPAdamWEdgeCases:
    """Tests for edge cases."""

    def test_zero_gradient(self):
        """Test with zero gradients (only noise and weight decay affect parameters)."""
        init_fn, step_fn = dp_adamw(
            weight_decay=0.01,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.ones(10)}
        original_weight = params["weight"].clone()
        grads = {"weight": torch.zeros(10)}  # Zero gradient

        state = init_fn(params)
        new_params, _, _ = step_fn(params, grads, state)

        # With zero gradient, noise + weight decay should cause changes
        assert not torch.allclose(new_params["weight"], original_weight, atol=1e-6)

    def test_large_gradients(self):
        """Test with large gradients (should be clipped)."""
        init_fn, step_fn = dp_adamw(
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.zeros(10)}
        grads = {"weight": torch.ones(10) * 10.0}  # Large gradient

        state = init_fn(params)
        new_params, _, metrics = step_fn(params, grads, state)

        # Should still work (clipping handles large gradients)
        assert metrics["step"] == 1
        assert metrics["epsilon"] > 0

    def test_zero_weight_decay(self):
        """Test that weight_decay=0 behaves like Adam."""
        init_fn, step_fn = dp_adamw(
            learning_rate=0.01,
            weight_decay=0.0,  # No weight decay
            l2_clip_norm=1.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        params = {"weight": torch.ones(10) * 2.0}
        state = init_fn(params)

        # Apply zero gradients
        for _ in range(10):
            grad = {"weight": torch.zeros(10)}
            params, state, _ = step_fn(params, grad, state)

        # Without weight decay and zero gradients, only noise affects parameters
        # Magnitude should stay roughly the same
        assert torch.abs(params["weight"]).mean() > 1.0  # Not pulled toward zero


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
