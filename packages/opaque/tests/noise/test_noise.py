"""Unit tests for noise module - unified (noise_fn, state) API."""

import pytest
import scipy.stats
import torch

from opaque.noise import gaussian_noise
from opaque.noise.gaussian import GaussianNoiseState
from opaque.random import key


class TestGaussian:
    """Tests for gaussian_noise() function."""

    def test_returns_tuple(self):
        """gaussian_noise() should return (noise_fn, state) tuple."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))
        assert callable(noise_fn)
        assert isinstance(state, GaussianNoiseState)

    def test_adds_noise_to_tensor(self):
        """Noise function should add noise to a tensor."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))
        grad = torch.zeros(10, 5)
        noisy, state = noise_fn(grad, state)

        assert noisy.shape == grad.shape
        assert noisy.dtype == grad.dtype
        assert not torch.allclose(noisy, grad)

    def test_adds_noise_to_pytree(self):
        """Noise function should work with PyTrees."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))
        grads = {
            "weight": torch.zeros(10, 5),
            "bias": torch.zeros(10),
        }
        noisy, state = noise_fn(grads, state)

        assert set(noisy.keys()) == set(grads.keys())
        assert noisy["weight"].shape == grads["weight"].shape
        assert noisy["bias"].shape == grads["bias"].shape
        assert not torch.allclose(noisy["weight"], grads["weight"])

    def test_zero_stddev(self):
        """stddev=0 should return original gradients."""
        noise_fn, state = gaussian_noise(stddev=0.0, key=key(0))
        grad = torch.randn(5, 3)
        noisy, state = noise_fn(grad, state)
        assert torch.equal(noisy, grad)

    def test_negative_stddev_raises(self):
        """Negative stddev should raise ValueError."""
        with pytest.raises(ValueError, match="stddev must be non-negative"):
            gaussian_noise(stddev=-1.0, key=key(0))

    def test_dtype_preservation(self):
        """Noise function should preserve dtype."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))

        grad_f32 = torch.randn(5, 3, dtype=torch.float32)
        noisy_f32, state = noise_fn(grad_f32, state)
        assert noisy_f32.dtype == torch.float32

        grad_f64 = torch.randn(5, 3, dtype=torch.float64)
        noisy_f64, state = noise_fn(grad_f64, state)
        assert noisy_f64.dtype == torch.float64

    def test_device_preservation(self):
        """Noise function should preserve device."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))

        grad_cpu = torch.randn(5, 3)
        noisy_cpu, state = noise_fn(grad_cpu, state)
        assert noisy_cpu.device == torch.device("cpu")

        if torch.backends.mps.is_available():
            grad_mps = torch.randn(5, 3, device="mps")
            noisy_mps, state = noise_fn(grad_mps, state)
            assert noisy_mps.device.type == "mps"

    def test_noise_normality(self):
        """Noise should follow normal distribution."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
        zeros = torch.zeros(10000)
        noisy, state = noise_fn(zeros, state)

        _, p_value = scipy.stats.kstest(noisy.numpy(), "norm", args=(0, 1))
        assert p_value > 0.01

    def test_noise_stddev(self):
        """Noise should have correct stddev."""
        target_stddev = 2.5
        noise_fn, state = gaussian_noise(stddev=target_stddev, key=key(0))
        zeros = torch.zeros(10000)
        noisy, state = noise_fn(zeros, state)

        measured_stddev = noisy.std().item()
        assert abs(measured_stddev - target_stddev) < 0.1

    def test_uniqueness(self):
        """Successive calls should produce different noise."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))
        grad = torch.zeros(100)

        noisy1, state = noise_fn(grad, state)
        noisy2, state = noise_fn(grad, state)

        assert not torch.allclose(noisy1, noisy2)

    def test_nested_pytree(self):
        """Works with nested PyTree structures."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))
        grads = {
            "layer1": {"w": torch.zeros(10, 5), "b": torch.zeros(10)},
            "layer2": {"w": torch.zeros(5, 3), "b": torch.zeros(3)},
        }
        noisy, state = noise_fn(grads, state)

        assert set(noisy.keys()) == {"layer1", "layer2"}
        assert not torch.allclose(noisy["layer1"]["w"], grads["layer1"]["w"])

    def test_tuple_pytree(self):
        """Works with tuple PyTrees."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))
        grads = (torch.zeros(10, 5), torch.zeros(10))
        noisy, state = noise_fn(grads, state)

        assert len(noisy) == 2
        assert not torch.allclose(noisy[0], grads[0])

    def test_stddev_override(self):
        """Per-call stddev override should change noise scale."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))
        grad = torch.zeros(10000)

        # Override with smaller stddev
        noisy, state = noise_fn(grad, state, stddev=0.5)
        measured_stddev = noisy.std().item()
        assert abs(measured_stddev - 0.5) < 0.05

    def test_stddev_override_does_not_affect_default(self):
        """Per-call override should not change the default for next call."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
        grad = torch.zeros(10000)

        # Override with small stddev
        _, state = noise_fn(grad, state, stddev=0.1)

        # Default should still be 1.0
        noisy, state = noise_fn(grad, state)
        assert noisy.var().item() > 0.5  # should have variance ~1.0, not ~0.01


class TestGaussianKey:
    """Tests for required key parameter."""

    def test_key_required(self):
        """Missing key should raise TypeError."""
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
            gaussian_noise(stddev=1.0)

    def test_generator_int_reproducible(self):
        """key(42) should produce reproducible noise."""
        noise_fn1, state1 = gaussian_noise(stddev=1.0, key=key(42))
        noise_fn2, state2 = gaussian_noise(stddev=1.0, key=key(42))

        grad = torch.zeros(10, 10)
        noisy1, state1 = noise_fn1(grad, state1)
        noisy2, state2 = noise_fn2(grad, state2)

        assert torch.allclose(noisy1, noisy2)

    def test_generator_different_seeds(self):
        """Different keys should produce different noise."""
        noise_fn1, state1 = gaussian_noise(stddev=1.0, key=key(42))
        noise_fn2, state2 = gaussian_noise(stddev=1.0, key=key(43))

        grad = torch.zeros(10, 10)
        noisy1, _ = noise_fn1(grad, state1)
        noisy2, _ = noise_fn2(grad, state2)

        assert not torch.allclose(noisy1, noisy2)

    def test_seed_int_produces_reproducible_state(self):
        """key(42) should initialize RNG state deterministically."""
        noise_fn1, state1 = gaussian_noise(stddev=1.0, key=key(42))
        noise_fn2, state2 = gaussian_noise(stddev=1.0, key=key(42))

        # Both should produce same initial state
        grad = torch.zeros(10)
        noisy1, _ = noise_fn1(grad, state1)
        noisy2, _ = noise_fn2(grad, state2)
        assert torch.allclose(noisy1, noisy2)

    def test_state_evolution(self):
        """State should evolve, producing different noise each call."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))

        grad = torch.zeros(10)
        noisy1, state = noise_fn(grad, state)
        noisy2, state = noise_fn(grad, state)

        assert not torch.allclose(noisy1, noisy2)

    def test_saved_state_replay_is_deterministic(self):
        """Re-using the same immutable state should replay identical noise."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
        grad = torch.zeros(10)

        noisy1, _ = noise_fn(grad, state)
        noisy2, _ = noise_fn(grad, state)

        assert torch.allclose(noisy1, noisy2)

    def test_zero_stddev_with_seed(self):
        """stddev=0 should return original gradients even with explicit key."""
        noise_fn, state = gaussian_noise(stddev=0.0, key=key(42))
        grad = torch.randn(5, 3)
        noisy, state = noise_fn(grad, state)
        assert torch.equal(noisy, grad)

    def test_invalid_seed_type_raises(self):
        """Invalid key type should raise TypeError."""
        with pytest.raises(TypeError, match="key must be"):
            gaussian_noise(stddev=1.0, key="bad")
