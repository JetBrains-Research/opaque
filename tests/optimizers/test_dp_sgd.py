"""Tests for DP-SGD optimizer."""

import pytest
import torch

from opaque.optimizers import DPOptimizerState, dp_sgd


class TestDPSGDCreation:
    """Tests for DP-SGD optimizer creation."""

    def test_creates_init_and_step_functions(self):
        """Test that dp_sgd returns init_fn and step_fn."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        assert callable(init_fn)
        assert callable(step_fn)

    def test_default_parameters(self):
        """Test DP-SGD with default parameters."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)
        assert state.step == 0

    def test_with_momentum(self):
        """Test DP-SGD with momentum."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
            momentum=0.9,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)

    def test_with_nesterov(self):
        """Test DP-SGD with Nesterov momentum."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)


class TestDPSGDOptimization:
    """Tests for DP-SGD optimization behavior."""

    def test_single_step(self):
        """Test a single DP-SGD optimization step."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
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
        """Test multiple DP-SGD steps."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
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
        """Test that DP-SGD can minimize a simple quadratic loss."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
            momentum=0.0,
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

    def test_momentum_accumulation(self):
        """Test that momentum accumulates across steps."""
        # High momentum
        init_fn1, step_fn1 = dp_sgd(
            learning_rate=0.1,
            momentum=0.9,  # High momentum
            l2_clip_norm=1.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        # No momentum
        init_fn2, step_fn2 = dp_sgd(
            learning_rate=0.1,
            momentum=0.0,  # No momentum
            l2_clip_norm=1.0,
            noise_multiplier=0.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        params1 = {"weight": torch.zeros(10)}
        params2 = {"weight": torch.zeros(10)}
        grad = {"weight": torch.ones(10) * 0.5}

        state1 = init_fn1(params1)
        state2 = init_fn2(params2)

        # Take 5 steps with constant gradient
        for _ in range(5):
            params1, state1, _ = step_fn1(params1, grad, state1)
            params2, state2, _ = step_fn2(params2, grad, state2)

        # With momentum, should move further (momentum builds up)
        change1 = torch.abs(params1["weight"]).mean()
        change2 = torch.abs(params2["weight"]).mean()

        # Momentum version should have moved more
        assert change1 > change2 * 1.5


class TestDPSGDPrivacy:
    """Tests for privacy accounting in DP-SGD."""

    def test_privacy_cost_increases(self):
        """Test that privacy cost increases with steps."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
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
        init_fn1, step_fn1 = dp_sgd(
            learning_rate=0.1,
            l2_clip_norm=1.0,
            noise_multiplier=0.5,  # Low noise
            sample_rate=0.01,
            target_delta=1e-5,
        )

        # High noise
        init_fn2, step_fn2 = dp_sgd(
            learning_rate=0.1,
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
        init_fn, _ = dp_sgd(
            learning_rate=0.1,
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
        init_fn, _ = dp_sgd(
            learning_rate=0.1,
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


class TestDPSGDReproducibility:
    """Tests for reproducibility in DP-SGD."""

    def test_seed_reproducibility(self):
        """Test that same seed gives reproducible results."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
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
        init_fn1, step_fn1 = dp_sgd(
            learning_rate=0.1,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        # Seed 43
        init_fn2, step_fn2 = dp_sgd(
            learning_rate=0.1,
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


class TestDPSGDPyTree:
    """Tests for PyTree parameter support."""

    def test_nested_pytree(self):
        """Test with nested PyTree parameters."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
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


class TestDPSGDEdgeCases:
    """Tests for edge cases."""

    def test_zero_gradient(self):
        """Test with zero gradients (only noise affects parameters)."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.zeros(10)}
        original_weight = params["weight"].clone()
        grads = {"weight": torch.zeros(10)}  # Zero gradient

        state = init_fn(params)
        new_params, _, _ = step_fn(params, grads, state)

        # With zero gradient, noise should still cause changes
        assert not torch.allclose(new_params["weight"], original_weight, atol=1e-6)

    def test_large_gradients(self):
        """Test with large gradients (should be clipped)."""
        init_fn, step_fn = dp_sgd(
            learning_rate=0.1,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
