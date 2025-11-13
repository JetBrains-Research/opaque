"""Tests for base DP optimizer infrastructure."""

import pytest
import torch
import torchopt

from opaque.optimizers.base import DPOptimizerState, make_dp_optimizer


class TestDPOptimizerState:
    """Tests for DPOptimizerState."""

    def test_state_creation(self):
        """Test creating a DPOptimizerState."""
        from opaque.accounting import RDPAccountant

        accountant = RDPAccountant()
        noise_gen = torch.Generator().manual_seed(42)
        opt_state = {"mu": torch.zeros(5)}

        state = DPOptimizerState(
            opt_state=opt_state,
            accountant=accountant,
            noise_gen=noise_gen,
            step=0,
        )

        assert state.opt_state == opt_state
        assert state.accountant is accountant
        assert state.noise_gen is noise_gen
        assert state.step == 0

    def test_state_immutability(self):
        """Test that state is a NamedTuple (immutable)."""
        from opaque.accounting import RDPAccountant

        accountant = RDPAccountant()
        noise_gen = torch.Generator().manual_seed(42)

        state = DPOptimizerState(
            opt_state={},
            accountant=accountant,
            noise_gen=noise_gen,
            step=0,
        )

        # NamedTuples are immutable
        with pytest.raises(AttributeError):
            state.step = 1


class TestMakeDPOptimizer:
    """Tests for make_dp_optimizer wrapper."""

    def test_creates_init_and_step_functions(self):
        """Test that make_dp_optimizer returns init_fn and step_fn."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        assert callable(init_fn)
        assert callable(step_fn)

    def test_init_creates_state(self):
        """Test that init_fn creates proper DPOptimizerState."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, _ = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        params = {"weight": torch.randn(5, 3), "bias": torch.randn(3)}
        state = init_fn(params)

        assert isinstance(state, DPOptimizerState)
        assert state.step == 0
        assert state.opt_state is not None
        assert state.accountant is not None
        assert state.noise_gen is not None

    def test_step_updates_parameters(self):
        """Test that step_fn updates parameters correctly."""
        base_opt = torchopt.sgd(lr=0.1, momentum=0.0)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.ones(5, 3)}
        original_weight = params["weight"].clone()  # Save original
        state = init_fn(params)

        # Simple gradient (clipped, before noise)
        grads = {"weight": torch.ones(5, 3) * 0.5}

        new_params, new_state, metrics = step_fn(params, grads, state)

        # Parameters should change (SGD step with noise)
        assert not torch.allclose(new_params["weight"], original_weight)

        # State should update
        assert new_state.step == 1
        assert metrics["step"] == 1

    def test_step_adds_noise(self):
        """Test that step_fn adds Gaussian noise to gradients."""
        base_opt = torchopt.sgd(lr=0.1, momentum=0.0)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        params = {"weight": torch.zeros(100)}
        original_weight = params["weight"].clone()  # Save original
        state = init_fn(params)

        # Zero gradients
        grads = {"weight": torch.zeros(100)}

        new_params, _, _ = step_fn(params, grads, state)

        # With zero gradients, any parameter change is from noise
        # Noise stddev = 1.1 * 1.0 = 1.1
        # With 100 dimensions, should see non-zero change
        assert not torch.allclose(new_params["weight"], original_weight, atol=1e-6)

        # Noise should have reasonable magnitude
        change = torch.abs(new_params["weight"] - original_weight)
        assert change.mean() > 0.05  # Should have noticeable noise

    def test_step_tracks_privacy(self):
        """Test that step_fn tracks privacy budget."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        state = init_fn(params)
        grads = {"weight": torch.randn(5, 3) * 0.5}

        # First step
        _, state, metrics1 = step_fn(params, grads, state)
        epsilon1 = metrics1["epsilon"]

        # Second step
        _, state, metrics2 = step_fn(params, grads, state)
        epsilon2 = metrics2["epsilon"]

        # Privacy cost should increase
        assert epsilon2 > epsilon1
        assert metrics1["delta"] == 1e-5
        assert metrics2["delta"] == 1e-5

    def test_rdp_accountant(self):
        """Test using RDP accountant."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, _ = make_dp_optimizer(
            base_opt,
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
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, _ = make_dp_optimizer(
            base_opt,
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

    def test_invalid_accountant_type(self):
        """Test that invalid accountant_type raises error."""
        base_opt = torchopt.sgd(lr=0.1)

        with pytest.raises(ValueError, match="Unknown accountant_type"):
            make_dp_optimizer(
                base_opt,
                l2_clip_norm=1.0,
                noise_multiplier=1.1,
                sample_rate=0.01,
                target_delta=1e-5,
                accountant_type="invalid",
            )

    def test_noise_scales_with_clip_norm(self):
        """Test that noise scales with clip norm."""
        # Use small LR to see noise effect without it dominating
        base_opt = torchopt.sgd(lr=0.01)

        # Low clip norm
        init_fn1, step_fn1 = make_dp_optimizer(
            base_opt,
            l2_clip_norm=0.1,  # Small C → stddev = 0.1
            noise_multiplier=1.0,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        # High clip norm
        init_fn2, step_fn2 = make_dp_optimizer(
            base_opt,
            l2_clip_norm=10.0,  # Large C → stddev = 10.0
            noise_multiplier=1.0,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        params1 = {"weight": torch.zeros(100)}
        params2 = {"weight": torch.zeros(100)}
        grads = {"weight": torch.zeros(100)}  # Zero gradients to isolate noise

        # Step with low clip norm
        state1 = init_fn1(params1)
        new_params1, _, _ = step_fn1(params1, grads, state1)
        change1 = torch.abs(new_params1["weight"]).mean()

        # Step with high clip norm
        state2 = init_fn2(params2)
        new_params2, _, _ = step_fn2(params2, grads, state2)
        change2 = torch.abs(new_params2["weight"]).mean()

        # Higher clip norm → higher noise
        # Noise scales linearly with clip norm (10x clip norm → ~10x noise)
        assert change2 > change1 * 5  # Should be much larger

    def test_reproducibility_with_seed(self):
        """Test that same seed gives reproducible results."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
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

    def test_different_seeds_give_different_results(self):
        """Test that different seeds give different noise."""
        base_opt = torchopt.sgd(lr=0.1)

        # Seed 42
        init_fn1, step_fn1 = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=42,
        )

        # Seed 43
        init_fn2, step_fn2 = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
            seed=43,
        )

        # Use separate param dicts to avoid mutation issues
        params1 = {"weight": torch.randn(5, 3)}
        params2 = {"weight": params1["weight"].clone()}  # Same initial values
        grads1 = {"weight": torch.randn(5, 3)}
        grads2 = {"weight": grads1["weight"].clone()}  # Same gradient

        # Run with different seeds
        state1 = init_fn1(params1)
        new_params1, _, _ = step_fn1(params1, grads1, state1)

        state2 = init_fn2(params2)
        new_params2, _, _ = step_fn2(params2, grads2, state2)

        # Should be different (due to different noise)
        assert not torch.allclose(new_params1["weight"], new_params2["weight"])


class TestMakeDPOptimizerPyTree:
    """Tests for PyTree parameter support."""

    def test_nested_pytree(self):
        """Test with nested PyTree parameters."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        # Nested structure
        params = {
            "layer1": {"weight": torch.randn(5, 3), "bias": torch.randn(3)},
            "layer2": {"weight": torch.randn(3, 2), "bias": torch.randn(2)},
        }
        original_weight = params["layer1"]["weight"].clone()  # Save original

        grads = {
            "layer1": {"weight": torch.randn(5, 3), "bias": torch.randn(3)},
            "layer2": {"weight": torch.randn(3, 2), "bias": torch.randn(2)},
        }

        state = init_fn(params)
        new_params, new_state, metrics = step_fn(params, grads, state)

        # Check structure preserved
        assert "layer1" in new_params
        assert "layer2" in new_params
        assert "weight" in new_params["layer1"]
        assert "bias" in new_params["layer1"]

        # Check updates applied
        assert not torch.allclose(new_params["layer1"]["weight"], original_weight)

    def test_single_tensor(self):
        """Test with single tensor (not dict)."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = torch.randn(5, 3)
        original_params = params.clone()  # Save original
        grads = torch.randn(5, 3)

        state = init_fn(params)
        new_params, new_state, metrics = step_fn(params, grads, state)

        assert isinstance(new_params, torch.Tensor)
        assert new_params.shape == params.shape
        assert not torch.allclose(new_params, original_params)


class TestMakeDPOptimizerMetrics:
    """Tests for metrics returned by step_fn."""

    def test_metrics_structure(self):
        """Test that metrics have correct structure."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}

        state = init_fn(params)
        _, _, metrics = step_fn(params, grads, state)

        assert "epsilon" in metrics
        assert "delta" in metrics
        assert "step" in metrics

    def test_epsilon_increases_over_time(self):
        """Test that epsilon increases with more steps."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
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

    def test_step_counter_increments(self):
        """Test that step counter increments correctly."""
        base_opt = torchopt.sgd(lr=0.1)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(5, 3)}
        grads = {"weight": torch.randn(5, 3)}

        state = init_fn(params)

        for expected_step in range(1, 11):
            params, state, metrics = step_fn(params, grads, state)
            assert state.step == expected_step
            assert metrics["step"] == expected_step


class TestMakeDPOptimizerIntegration:
    """Integration tests with different base optimizers."""

    def test_with_adam(self):
        """Test wrapping Adam optimizer."""
        base_opt = torchopt.adam(lr=0.001, betas=(0.9, 0.999))

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(10, 5)}
        original_weight = params["weight"].clone()  # Save original
        grads = {"weight": torch.randn(10, 5) * 0.5}

        state = init_fn(params)
        new_params, new_state, metrics = step_fn(params, grads, state)

        assert new_params["weight"].shape == params["weight"].shape
        assert not torch.allclose(new_params["weight"], original_weight)
        assert metrics["epsilon"] > 0

    def test_with_sgd_momentum(self):
        """Test wrapping SGD with momentum."""
        base_opt = torchopt.sgd(lr=0.1, momentum=0.9)

        init_fn, step_fn = make_dp_optimizer(
            base_opt,
            l2_clip_norm=1.0,
            noise_multiplier=1.1,
            sample_rate=0.01,
            target_delta=1e-5,
        )

        params = {"weight": torch.randn(10, 5)}
        grads = {"weight": torch.randn(10, 5) * 0.5}

        state = init_fn(params)

        # Multiple steps to see momentum effect
        for _ in range(5):
            params, state, metrics = step_fn(params, grads, state)

        assert state.step == 5
        assert metrics["epsilon"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
