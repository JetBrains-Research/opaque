"""Unit tests for truncated Gaussian noise mechanism."""

import pytest
import scipy.stats
import torch

from opaque.noise import truncated_gaussian_noise
from opaque.noise.gaussian_noise import GaussianNoiseState
from opaque.random import key


class TestBoundedGaussian:
    """Tests for truncated_gaussian_noise() function."""

    def test_returns_tuple(self):
        """truncated_gaussian_noise() should return (noise_fn, state) tuple."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-3.0, 3.0), key=key(0)
        )
        assert callable(noise_fn)
        assert isinstance(state, GaussianNoiseState)

    def test_adds_noise_to_tensor(self):
        """Noise function should add noise to a tensor."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(0)
        )
        grad = torch.zeros(10, 5)
        noisy, state = noise_fn(grad, state)

        assert noisy.shape == grad.shape
        assert noisy.dtype == grad.dtype
        assert not torch.allclose(noisy, grad)

    def test_adds_noise_to_pytree(self):
        """Noise function should work with PyTrees."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(0)
        )
        grads = {
            "weight": torch.zeros(10, 5),
            "bias": torch.zeros(10),
        }
        noisy, state = noise_fn(grads, state)

        assert set(noisy.keys()) == set(grads.keys())
        assert noisy["weight"].shape == grads["weight"].shape
        assert noisy["bias"].shape == grads["bias"].shape
        assert not torch.allclose(noisy["weight"], grads["weight"])

    def test_output_within_bounds(self):
        """All outputs must lie within the specified bounds."""
        lower, upper = -2.0, 2.0
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(lower, upper), key=key(0)
        )
        grad = torch.zeros(10000)
        noisy, state = noise_fn(grad, state)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_output_within_bounds_nonzero_center(self):
        """Bounds are respected even when input values are nonzero."""
        lower, upper = -3.0, 3.0
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(lower, upper), key=key(0)
        )
        grad = torch.tensor([2.5, -2.5, 0.0, 1.0, -1.0]).repeat(2000)
        noisy, state = noise_fn(grad, state)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_output_within_tight_bounds(self):
        """Tight bounds are respected (high stddev relative to bound width)."""
        lower, upper = -0.5, 0.5
        noise_fn, state = truncated_gaussian_noise(
            stddev=5.0, bounds=(lower, upper), key=key(0)
        )
        grad = torch.zeros(10000)
        noisy, state = noise_fn(grad, state)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_zero_stddev(self):
        """stddev=0 should clamp to bounds without adding noise."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=0.0, bounds=(-1.0, 1.0), key=key(0)
        )
        grad = torch.tensor([0.5, -0.5, 0.0])
        noisy, state = noise_fn(grad, state)
        assert torch.equal(noisy, grad)

    def test_zero_stddev_clamps(self):
        """stddev=0 with out-of-bounds input should clamp."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=0.0, bounds=(-1.0, 1.0), key=key(0)
        )
        grad = torch.tensor([2.0, -2.0, 0.5])
        noisy, state = noise_fn(grad, state)
        expected = torch.tensor([1.0, -1.0, 0.5])
        assert torch.equal(noisy, expected)

    def test_negative_stddev_raises(self):
        """Negative stddev should raise ValueError."""
        with pytest.raises(ValueError, match="stddev must be non-negative"):
            truncated_gaussian_noise(stddev=-1.0, bounds=(-1.0, 1.0), key=key(0))

    def test_invalid_bounds_raises(self):
        """lower >= upper should raise ValueError."""
        with pytest.raises(ValueError, match="bounds must satisfy lower < upper"):
            truncated_gaussian_noise(stddev=1.0, bounds=(1.0, 1.0), key=key(0))
        with pytest.raises(ValueError, match="bounds must satisfy lower < upper"):
            truncated_gaussian_noise(stddev=1.0, bounds=(2.0, 1.0), key=key(0))

    def test_dtype_preservation(self):
        """Noise function should preserve dtype."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(0)
        )

        grad_f32 = torch.randn(5, 3, dtype=torch.float32)
        noisy_f32, state = noise_fn(grad_f32, state)
        assert noisy_f32.dtype == torch.float32

        grad_f64 = torch.randn(5, 3, dtype=torch.float64)
        noisy_f64, state = noise_fn(grad_f64, state)
        assert noisy_f64.dtype == torch.float64

    def test_device_preservation(self):
        """Noise function should preserve device."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(0)
        )

        grad_cpu = torch.randn(5, 3)
        noisy_cpu, state = noise_fn(grad_cpu, state)
        assert noisy_cpu.device == torch.device("cpu")

        if torch.backends.mps.is_available():
            grad_mps = torch.randn(5, 3, device="mps")
            noisy_mps, state = noise_fn(grad_mps, state)
            assert noisy_mps.device.type == "mps"

    def test_noise_distribution_truncated_normal(self):
        """Output should follow a truncated normal distribution."""
        stddev = 1.0
        lower, upper = -2.0, 2.0
        noise_fn, state = truncated_gaussian_noise(
            stddev=stddev, bounds=(lower, upper), key=key(42)
        )
        zeros = torch.zeros(50000)
        noisy, state = noise_fn(zeros, state)

        a_std = lower / stddev
        b_std = upper / stddev
        _, p_value = scipy.stats.kstest(
            noisy.numpy(),
            "truncnorm",
            args=(a_std, b_std, 0.0, stddev),
        )
        assert p_value > 0.01, f"KS test failed with p={p_value}"

    def test_noise_mean_approximately_zero(self):
        """For symmetric bounds and zero-centered input, mean should be ~0."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(0)
        )
        zeros = torch.zeros(50000)
        noisy, state = noise_fn(zeros, state)

        assert abs(noisy.mean().item()) < 0.05

    def test_variance_less_than_unbounded(self):
        """Truncated Gaussian should have lower variance than unbounded."""
        stddev = 1.0
        bounds = (-2.0, 2.0)
        noise_fn, state = truncated_gaussian_noise(
            stddev=stddev, bounds=bounds, key=key(0)
        )
        zeros = torch.zeros(50000)
        noisy, state = noise_fn(zeros, state)

        measured_var = noisy.var().item()
        assert measured_var < stddev**2

    def test_uniqueness(self):
        """Successive calls should produce different noise."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(0)
        )
        grad = torch.zeros(100)

        noisy1, state = noise_fn(grad, state)
        noisy2, state = noise_fn(grad, state)

        assert not torch.allclose(noisy1, noisy2)

    def test_nested_pytree(self):
        """Works with nested PyTree structures."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(0)
        )
        grads = {
            "layer1": {"w": torch.zeros(10, 5), "b": torch.zeros(10)},
            "layer2": {"w": torch.zeros(5, 3), "b": torch.zeros(3)},
        }
        noisy, state = noise_fn(grads, state)

        assert set(noisy.keys()) == {"layer1", "layer2"}
        assert not torch.allclose(noisy["layer1"]["w"], grads["layer1"]["w"])

    def test_tuple_pytree(self):
        """Works with tuple PyTrees."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(0)
        )
        grads = (torch.zeros(10, 5), torch.zeros(10))
        noisy, state = noise_fn(grads, state)

        assert len(noisy) == 2
        assert not torch.allclose(noisy[0], grads[0])

    def test_asymmetric_bounds(self):
        """Works with asymmetric bounds."""
        lower, upper = -1.0, 5.0
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(lower, upper), key=key(0)
        )
        grad = torch.ones(10000) * 2.0
        noisy, state = noise_fn(grad, state)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_input_at_boundary(self):
        """Input values at the boundary should produce valid outputs."""
        lower, upper = -2.0, 2.0
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(lower, upper), key=key(0)
        )
        grad = torch.tensor([lower, upper, lower, upper]).repeat(1000)
        noisy, state = noise_fn(grad, state)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper


class TestBoundedGaussianKey:
    """Tests for key parameter."""

    def test_reproducibility(self):
        """Same generator seed should produce same noise."""
        noise_fn1, state1 = truncated_gaussian_noise(
            stddev=1.0, bounds=(-3.0, 3.0), key=key(42)
        )
        noise_fn2, state2 = truncated_gaussian_noise(
            stddev=1.0, bounds=(-3.0, 3.0), key=key(42)
        )

        grad = torch.zeros(10, 10)
        noisy1, state1 = noise_fn1(grad, state1)
        noisy2, state2 = noise_fn2(grad, state2)

        assert torch.allclose(noisy1, noisy2)

    def test_different_seeds(self):
        """Different seeds should produce different noise."""
        noise_fn1, state1 = truncated_gaussian_noise(
            stddev=1.0, bounds=(-3.0, 3.0), key=key(42)
        )
        noise_fn2, state2 = truncated_gaussian_noise(
            stddev=1.0, bounds=(-3.0, 3.0), key=key(43)
        )

        grad = torch.zeros(10, 10)
        noisy1, _ = noise_fn1(grad, state1)
        noisy2, _ = noise_fn2(grad, state2)

        assert not torch.allclose(noisy1, noisy2)

    def test_output_within_bounds_with_generator(self):
        """Stateful version must also respect bounds."""
        lower, upper = -2.0, 2.0
        noise_fn, state = truncated_gaussian_noise(
            stddev=1.0, bounds=(lower, upper), key=key(42)
        )
        grad = torch.zeros(10000)
        noisy, state = noise_fn(grad, state)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_zero_stddev_with_generator(self):
        """stddev=0 should clamp to bounds without noise."""
        noise_fn, state = truncated_gaussian_noise(
            stddev=0.0, bounds=(-1.0, 1.0), key=key(42)
        )
        grad = torch.tensor([0.5, -0.5, 2.0, -2.0])
        noisy, state = noise_fn(grad, state)
        expected = torch.tensor([0.5, -0.5, 1.0, -1.0])
        assert torch.equal(noisy, expected)

    def test_negative_stddev_raises_with_generator(self):
        """Negative stddev should raise ValueError."""
        with pytest.raises(ValueError, match="stddev must be non-negative"):
            truncated_gaussian_noise(stddev=-1.0, bounds=(-1.0, 1.0), key=key(42))

    def test_invalid_bounds_raises_with_generator(self):
        """Invalid bounds should raise ValueError."""
        with pytest.raises(ValueError, match="bounds must satisfy lower < upper"):
            truncated_gaussian_noise(stddev=1.0, bounds=(1.0, -1.0), key=key(42))
