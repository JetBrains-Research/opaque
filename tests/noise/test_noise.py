"""Unit tests for noise module - new functional API."""

import pytest
import scipy.stats
import torch

from opaque.noise import gaussian_noise, gaussian_noise_stateful


class TestGaussian:
    """Tests for gaussian_noise() function."""

    def test_returns_callable(self):
        """gaussian_noise() should return a callable."""
        noise_fn = gaussian_noise(stddev=1.0)
        assert callable(noise_fn)

    def test_adds_noise_to_tensor(self):
        """Noise function should add noise to a tensor."""
        noise_fn = gaussian_noise(stddev=1.0)
        grad = torch.zeros(10, 5)
        noisy = noise_fn(grad)

        assert noisy.shape == grad.shape
        assert noisy.dtype == grad.dtype
        assert not torch.allclose(noisy, grad)

    def test_adds_noise_to_pytree(self):
        """Noise function should work with PyTrees."""
        noise_fn = gaussian_noise(stddev=1.0)
        grads = {
            "weight": torch.zeros(10, 5),
            "bias": torch.zeros(10),
        }
        noisy = noise_fn(grads)

        assert set(noisy.keys()) == set(grads.keys())
        assert noisy["weight"].shape == grads["weight"].shape
        assert noisy["bias"].shape == grads["bias"].shape
        assert not torch.allclose(noisy["weight"], grads["weight"])

    def test_zero_stddev(self):
        """stddev=0 should return original gradients."""
        noise_fn = gaussian_noise(stddev=0.0)
        grad = torch.randn(5, 3)
        noisy = noise_fn(grad)
        assert torch.equal(noisy, grad)

    def test_negative_stddev_raises(self):
        """Negative stddev should raise ValueError."""
        with pytest.raises(ValueError, match="stddev must be non-negative"):
            gaussian_noise(stddev=-1.0)

    def test_dtype_preservation(self):
        """Noise function should preserve dtype."""
        noise_fn = gaussian_noise(stddev=1.0)

        # float32
        grad_f32 = torch.randn(5, 3, dtype=torch.float32)
        noisy_f32 = noise_fn(grad_f32)
        assert noisy_f32.dtype == torch.float32

        # float64
        grad_f64 = torch.randn(5, 3, dtype=torch.float64)
        noisy_f64 = noise_fn(grad_f64)
        assert noisy_f64.dtype == torch.float64

    def test_device_preservation(self):
        """Noise function should preserve device."""
        noise_fn = gaussian_noise(stddev=1.0)

        # CPU
        grad_cpu = torch.randn(5, 3)
        noisy_cpu = noise_fn(grad_cpu)
        assert noisy_cpu.device == torch.device("cpu")

        # MPS if available
        if torch.backends.mps.is_available():
            grad_mps = torch.randn(5, 3, device="mps")
            noisy_mps = noise_fn(grad_mps)
            assert noisy_mps.device.type == "mps"

    def test_noise_normality(self):
        """Noise should follow normal distribution."""
        noise_fn = gaussian_noise(stddev=1.0)
        zeros = torch.zeros(10000)
        noisy = noise_fn(zeros)

        # Test normality with Kolmogorov-Smirnov test
        _, p_value = scipy.stats.kstest(noisy.numpy(), "norm", args=(0, 1))
        assert p_value > 0.01  # Not rejecting null hypothesis

    def test_noise_stddev(self):
        """Noise should have correct stddev."""
        target_stddev = 2.5
        noise_fn = gaussian_noise(stddev=target_stddev)
        zeros = torch.zeros(10000)
        noisy = noise_fn(zeros)

        measured_stddev = noisy.std().item()
        assert abs(measured_stddev - target_stddev) < 0.1

    def test_uniqueness(self):
        """Successive calls should produce different noise."""
        noise_fn = gaussian_noise(stddev=1.0)
        grad = torch.zeros(100)

        noisy1 = noise_fn(grad)
        noisy2 = noise_fn(grad)

        assert not torch.allclose(noisy1, noisy2)

    def test_nested_pytree(self):
        """Works with nested PyTree structures."""
        noise_fn = gaussian_noise(stddev=1.0)
        grads = {
            "layer1": {"w": torch.zeros(10, 5), "b": torch.zeros(10)},
            "layer2": {"w": torch.zeros(5, 3), "b": torch.zeros(3)},
        }
        noisy = noise_fn(grads)

        assert set(noisy.keys()) == {"layer1", "layer2"}
        assert not torch.allclose(noisy["layer1"]["w"], grads["layer1"]["w"])

    def test_tuple_pytree(self):
        """Works with tuple PyTrees."""
        noise_fn = gaussian_noise(stddev=1.0)
        grads = (torch.zeros(10, 5), torch.zeros(10))
        noisy = noise_fn(grads)

        assert len(noisy) == 2
        assert not torch.allclose(noisy[0], grads[0])


class TestGaussianStateful:
    """Tests for gaussian_noise_stateful() function."""

    def test_returns_tuple(self):
        """gaussian_noise_stateful() should return (fn, state) tuple."""
        result = gaussian_noise_stateful(stddev=1.0, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

        noise_fn, state = result
        assert callable(noise_fn)
        assert isinstance(state, torch.Generator)

    def test_reproducibility(self):
        """Same seed should produce same noise."""
        noise_fn1, state1 = gaussian_noise_stateful(stddev=1.0, seed=42)
        noise_fn2, state2 = gaussian_noise_stateful(stddev=1.0, seed=42)

        grad = torch.zeros(10, 10)
        noisy1 = noise_fn1(grad, state1)
        noisy2 = noise_fn2(grad, state2)

        assert torch.allclose(noisy1, noisy2)

    def test_different_seeds(self):
        """Different seeds should produce different noise."""
        noise_fn1, state1 = gaussian_noise_stateful(stddev=1.0, seed=42)
        noise_fn2, state2 = gaussian_noise_stateful(stddev=1.0, seed=43)

        grad = torch.zeros(10, 10)
        noisy1 = noise_fn1(grad, state1)
        noisy2 = noise_fn2(grad, state2)

        assert not torch.allclose(noisy1, noisy2)

    def test_state_evolution(self):
        """State should evolve, producing different noise each call."""
        noise_fn, state = gaussian_noise_stateful(stddev=1.0, seed=42)

        grad = torch.zeros(10)
        noisy1 = noise_fn(grad, state)
        noisy2 = noise_fn(grad, state)

        assert not torch.allclose(noisy1, noisy2)

    def test_state_reset(self):
        """Resetting state should reproduce noise."""
        noise_fn, state = gaussian_noise_stateful(stddev=1.0, seed=42)

        grad = torch.zeros(10)
        noisy1 = noise_fn(grad, state)

        # Reset state
        state.manual_seed(42)
        noisy2 = noise_fn(grad, state)

        assert torch.allclose(noisy1, noisy2)

    def test_zero_stddev_stateful(self):
        """stddev=0 should return original gradients."""
        noise_fn, state = gaussian_noise_stateful(stddev=0.0, seed=42)
        grad = torch.randn(5, 3)
        noisy = noise_fn(grad, state)
        assert torch.equal(noisy, grad)

    def test_negative_stddev_raises_stateful(self):
        """Negative stddev should raise ValueError."""
        with pytest.raises(ValueError, match="stddev must be non-negative"):
            gaussian_noise_stateful(stddev=-1.0, seed=42)
