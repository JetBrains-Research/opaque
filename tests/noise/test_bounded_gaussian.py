"""Unit tests for bounded Gaussian noise mechanism."""

import pytest
import scipy.stats
import torch

from opaque.noise import bounded_gaussian, bounded_gaussian_stateful


class TestBoundedGaussian:
    """Tests for bounded_gaussian() function."""

    def test_returns_callable(self):
        """bounded_gaussian() should return a callable."""
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-3.0, 3.0))
        assert callable(noise_fn)

    def test_adds_noise_to_tensor(self):
        """Noise function should add noise to a tensor."""
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-5.0, 5.0))
        grad = torch.zeros(10, 5)
        noisy = noise_fn(grad)

        assert noisy.shape == grad.shape
        assert noisy.dtype == grad.dtype
        assert not torch.allclose(noisy, grad)

    def test_adds_noise_to_pytree(self):
        """Noise function should work with PyTrees."""
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-5.0, 5.0))
        grads = {
            "weight": torch.zeros(10, 5),
            "bias": torch.zeros(10),
        }
        noisy = noise_fn(grads)

        assert set(noisy.keys()) == set(grads.keys())
        assert noisy["weight"].shape == grads["weight"].shape
        assert noisy["bias"].shape == grads["bias"].shape
        assert not torch.allclose(noisy["weight"], grads["weight"])

    def test_output_within_bounds(self):
        """All outputs must lie within the specified bounds."""
        lower, upper = -2.0, 2.0
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(lower, upper))
        grad = torch.zeros(10000)
        noisy = noise_fn(grad)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_output_within_bounds_nonzero_center(self):
        """Bounds are respected even when input values are nonzero."""
        lower, upper = -3.0, 3.0
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(lower, upper))
        grad = torch.tensor([2.5, -2.5, 0.0, 1.0, -1.0]).repeat(2000)
        noisy = noise_fn(grad)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_output_within_tight_bounds(self):
        """Tight bounds are respected (high stddev relative to bound width)."""
        lower, upper = -0.5, 0.5
        noise_fn = bounded_gaussian(stddev=5.0, bounds=(lower, upper))
        grad = torch.zeros(10000)
        noisy = noise_fn(grad)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_zero_stddev(self):
        """stddev=0 should clamp to bounds without adding noise."""
        noise_fn = bounded_gaussian(stddev=0.0, bounds=(-1.0, 1.0))
        grad = torch.tensor([0.5, -0.5, 0.0])
        noisy = noise_fn(grad)
        assert torch.equal(noisy, grad)

    def test_zero_stddev_clamps(self):
        """stddev=0 with out-of-bounds input should clamp."""
        noise_fn = bounded_gaussian(stddev=0.0, bounds=(-1.0, 1.0))
        grad = torch.tensor([2.0, -2.0, 0.5])
        noisy = noise_fn(grad)
        expected = torch.tensor([1.0, -1.0, 0.5])
        assert torch.equal(noisy, expected)

    def test_negative_stddev_raises(self):
        """Negative stddev should raise ValueError."""
        with pytest.raises(ValueError, match="stddev must be non-negative"):
            bounded_gaussian(stddev=-1.0, bounds=(-1.0, 1.0))

    def test_invalid_bounds_raises(self):
        """lower >= upper should raise ValueError."""
        with pytest.raises(ValueError, match="bounds must satisfy lower < upper"):
            bounded_gaussian(stddev=1.0, bounds=(1.0, 1.0))
        with pytest.raises(ValueError, match="bounds must satisfy lower < upper"):
            bounded_gaussian(stddev=1.0, bounds=(2.0, 1.0))

    def test_dtype_preservation(self):
        """Noise function should preserve dtype."""
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-5.0, 5.0))

        grad_f32 = torch.randn(5, 3, dtype=torch.float32)
        noisy_f32 = noise_fn(grad_f32)
        assert noisy_f32.dtype == torch.float32

        grad_f64 = torch.randn(5, 3, dtype=torch.float64)
        noisy_f64 = noise_fn(grad_f64)
        assert noisy_f64.dtype == torch.float64

    def test_device_preservation(self):
        """Noise function should preserve device."""
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-5.0, 5.0))

        grad_cpu = torch.randn(5, 3)
        noisy_cpu = noise_fn(grad_cpu)
        assert noisy_cpu.device == torch.device("cpu")

        if torch.backends.mps.is_available():
            grad_mps = torch.randn(5, 3, device="mps")
            noisy_mps = noise_fn(grad_mps)
            assert noisy_mps.device.type == "mps"

    def test_noise_distribution_truncated_normal(self):
        """Output should follow a truncated normal distribution."""
        stddev = 1.0
        lower, upper = -2.0, 2.0
        noise_fn = bounded_gaussian(stddev=stddev, bounds=(lower, upper))
        zeros = torch.zeros(50000)
        noisy = noise_fn(zeros)

        # Compare against scipy truncated normal
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
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-5.0, 5.0))
        zeros = torch.zeros(50000)
        noisy = noise_fn(zeros)

        assert abs(noisy.mean().item()) < 0.05

    def test_variance_less_than_unbounded(self):
        """Bounded Gaussian should have lower variance than unbounded."""
        stddev = 1.0
        bounds = (-2.0, 2.0)
        bounded_fn = bounded_gaussian(stddev=stddev, bounds=bounds)
        zeros = torch.zeros(50000)
        bounded_noisy = bounded_fn(zeros)

        # Unbounded Gaussian variance is stddev^2 = 1.0
        # Truncated to [-2, 2] should have lower variance
        measured_var = bounded_noisy.var().item()
        assert measured_var < stddev**2

    def test_uniqueness(self):
        """Successive calls should produce different noise."""
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-5.0, 5.0))
        grad = torch.zeros(100)

        noisy1 = noise_fn(grad)
        noisy2 = noise_fn(grad)

        assert not torch.allclose(noisy1, noisy2)

    def test_nested_pytree(self):
        """Works with nested PyTree structures."""
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-5.0, 5.0))
        grads = {
            "layer1": {"w": torch.zeros(10, 5), "b": torch.zeros(10)},
            "layer2": {"w": torch.zeros(5, 3), "b": torch.zeros(3)},
        }
        noisy = noise_fn(grads)

        assert set(noisy.keys()) == {"layer1", "layer2"}
        assert not torch.allclose(noisy["layer1"]["w"], grads["layer1"]["w"])

    def test_tuple_pytree(self):
        """Works with tuple PyTrees."""
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(-5.0, 5.0))
        grads = (torch.zeros(10, 5), torch.zeros(10))
        noisy = noise_fn(grads)

        assert len(noisy) == 2
        assert not torch.allclose(noisy[0], grads[0])

    def test_asymmetric_bounds(self):
        """Works with asymmetric bounds."""
        lower, upper = -1.0, 5.0
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(lower, upper))
        grad = torch.ones(10000) * 2.0
        noisy = noise_fn(grad)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_input_at_boundary(self):
        """Input values at the boundary should produce valid outputs."""
        lower, upper = -2.0, 2.0
        noise_fn = bounded_gaussian(stddev=1.0, bounds=(lower, upper))
        grad = torch.tensor([lower, upper, lower, upper]).repeat(1000)
        noisy = noise_fn(grad)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper


class TestBoundedGaussianStateful:
    """Tests for bounded_gaussian_stateful() function."""

    def test_returns_tuple(self):
        """bounded_gaussian_stateful() should return (fn, state) tuple."""
        result = bounded_gaussian_stateful(stddev=1.0, bounds=(-3.0, 3.0), seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

        noise_fn, state = result
        assert callable(noise_fn)
        assert isinstance(state, torch.Generator)

    def test_reproducibility(self):
        """Same seed should produce same noise."""
        noise_fn1, state1 = bounded_gaussian_stateful(
            stddev=1.0, bounds=(-3.0, 3.0), seed=42
        )
        noise_fn2, state2 = bounded_gaussian_stateful(
            stddev=1.0, bounds=(-3.0, 3.0), seed=42
        )

        grad = torch.zeros(10, 10)
        noisy1 = noise_fn1(grad, state1)
        noisy2 = noise_fn2(grad, state2)

        assert torch.allclose(noisy1, noisy2)

    def test_different_seeds(self):
        """Different seeds should produce different noise."""
        noise_fn1, state1 = bounded_gaussian_stateful(
            stddev=1.0, bounds=(-3.0, 3.0), seed=42
        )
        noise_fn2, state2 = bounded_gaussian_stateful(
            stddev=1.0, bounds=(-3.0, 3.0), seed=43
        )

        grad = torch.zeros(10, 10)
        noisy1 = noise_fn1(grad, state1)
        noisy2 = noise_fn2(grad, state2)

        assert not torch.allclose(noisy1, noisy2)

    def test_state_evolution(self):
        """State should evolve, producing different noise each call."""
        noise_fn, state = bounded_gaussian_stateful(
            stddev=1.0, bounds=(-3.0, 3.0), seed=42
        )

        grad = torch.zeros(10)
        noisy1 = noise_fn(grad, state)
        noisy2 = noise_fn(grad, state)

        assert not torch.allclose(noisy1, noisy2)

    def test_state_reset(self):
        """Resetting state should reproduce noise."""
        noise_fn, state = bounded_gaussian_stateful(
            stddev=1.0, bounds=(-3.0, 3.0), seed=42
        )

        grad = torch.zeros(10)
        noisy1 = noise_fn(grad, state)

        # Reset state
        state.manual_seed(42)
        noisy2 = noise_fn(grad, state)

        assert torch.allclose(noisy1, noisy2)

    def test_output_within_bounds(self):
        """Stateful version must also respect bounds."""
        lower, upper = -2.0, 2.0
        noise_fn, state = bounded_gaussian_stateful(
            stddev=1.0, bounds=(lower, upper), seed=42
        )
        grad = torch.zeros(10000)
        noisy = noise_fn(grad, state)

        assert noisy.min().item() >= lower
        assert noisy.max().item() <= upper

    def test_zero_stddev_stateful(self):
        """stddev=0 should clamp to bounds without noise."""
        noise_fn, state = bounded_gaussian_stateful(
            stddev=0.0, bounds=(-1.0, 1.0), seed=42
        )
        grad = torch.tensor([0.5, -0.5, 2.0, -2.0])
        noisy = noise_fn(grad, state)
        expected = torch.tensor([0.5, -0.5, 1.0, -1.0])
        assert torch.equal(noisy, expected)

    def test_negative_stddev_raises_stateful(self):
        """Negative stddev should raise ValueError."""
        with pytest.raises(ValueError, match="stddev must be non-negative"):
            bounded_gaussian_stateful(stddev=-1.0, bounds=(-1.0, 1.0), seed=42)

    def test_invalid_bounds_raises_stateful(self):
        """Invalid bounds should raise ValueError."""
        with pytest.raises(ValueError, match="bounds must satisfy lower < upper"):
            bounded_gaussian_stateful(stddev=1.0, bounds=(1.0, -1.0), seed=42)
