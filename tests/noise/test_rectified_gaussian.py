"""Unit tests for rectified Gaussian noise mechanism."""

import pytest
import torch

from opaque.noise import rectified_gaussian_noise
from opaque.noise.gaussian_noise import GaussianNoiseState
from opaque.random import key


class TestRectifiedGaussianNoise:
    """Tests for rectified_gaussian_noise() function."""

    def test_returns_tuple(self):
        """rectified_gaussian_noise() should return (noise_fn, state) tuple."""
        noise_fn, state = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(0))
        assert callable(noise_fn)
        assert isinstance(state, GaussianNoiseState)

    def test_adds_noise_to_tensor(self):
        """Noise function should add noise to a tensor."""
        noise_fn, state = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(0))
        grad = torch.zeros(10, 5)
        noisy, state = noise_fn(grad, state)

        assert noisy.shape == grad.shape
        assert noisy.dtype == grad.dtype
        assert not torch.allclose(noisy, grad)

    def test_adds_noise_to_pytree(self):
        """Noise function should work with PyTrees."""
        noise_fn, state = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(0))
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
        """All noise values must lie within [-radius*stddev, radius*stddev]."""
        stddev, radius = 1.0, 5.0
        bound = stddev * radius
        noise_fn, state = rectified_gaussian_noise(
            stddev=stddev, radius=radius, key=key(0)
        )
        grad = torch.zeros(50_000)
        noisy, state = noise_fn(grad, state)

        assert noisy.min().item() >= -bound - 1e-6
        assert noisy.max().item() <= bound + 1e-6

    def test_output_within_tight_bounds(self):
        """Tight radius should clamp more aggressively."""
        stddev, radius = 1.0, 1.0  # clamp at ±1σ
        bound = stddev * radius
        noise_fn, state = rectified_gaussian_noise(
            stddev=stddev, radius=radius, key=key(0)
        )
        grad = torch.zeros(10_000)
        noisy, state = noise_fn(grad, state)

        assert noisy.min().item() >= -bound - 1e-6
        assert noisy.max().item() <= bound + 1e-6

    def test_zero_stddev(self):
        """stddev=0 should return input unchanged."""
        noise_fn, state = rectified_gaussian_noise(stddev=0.0, radius=5.0, key=key(0))
        grad = torch.tensor([0.5, -0.5, 0.0])
        noisy, state = noise_fn(grad, state)
        assert torch.equal(noisy, grad)

    def test_negative_stddev_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            rectified_gaussian_noise(stddev=-1.0, radius=5.0, key=key(0))

    def test_nonpositive_radius_raises(self):
        with pytest.raises(ValueError, match="positive"):
            rectified_gaussian_noise(stddev=1.0, radius=0.0, key=key(0))
        with pytest.raises(ValueError, match="positive"):
            rectified_gaussian_noise(stddev=1.0, radius=-1.0, key=key(0))

    def test_bad_key_raises(self):
        with pytest.raises(TypeError, match="RngKey"):
            rectified_gaussian_noise(stddev=1.0, radius=5.0, key=42)  # type: ignore[arg-type]

    def test_deterministic_same_key(self):
        """Same key should produce identical noise."""
        noise_fn1, state1 = rectified_gaussian_noise(
            stddev=1.0, radius=5.0, key=key(42)
        )
        noise_fn2, state2 = rectified_gaussian_noise(
            stddev=1.0, radius=5.0, key=key(42)
        )
        grad = torch.ones(100)

        noisy1, _ = noise_fn1(grad, state1)
        noisy2, _ = noise_fn2(grad, state2)
        assert torch.allclose(noisy1, noisy2)

    def test_different_keys_differ(self):
        """Different keys should produce different noise."""
        noise_fn1, state1 = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(0))
        noise_fn2, state2 = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(1))
        grad = torch.ones(100)

        noisy1, _ = noise_fn1(grad, state1)
        noisy2, _ = noise_fn2(grad, state2)
        assert not torch.allclose(noisy1, noisy2)

    def test_step_counter_increments(self):
        """Each call should increment the step counter."""
        noise_fn, state = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(0))
        assert state.step_counter == 0

        grad = torch.zeros(10)
        _, state = noise_fn(grad, state)
        assert state.step_counter == 1
        _, state = noise_fn(grad, state)
        assert state.step_counter == 2

    def test_state_is_immutable(self):
        """State should be frozen dataclass."""
        _, state = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(0))
        with pytest.raises(AttributeError):
            state.step_counter = 99  # type: ignore[misc]

    def test_large_radius_matches_gaussian(self):
        """With large radius, rectified noise ≈ Gaussian noise (no clamping)."""
        from opaque.noise import gaussian_noise

        stddev, radius = 1.0, 100.0  # effectively no clamping
        grad = torch.zeros(10_000)

        noise_fn_rect, state_rect = rectified_gaussian_noise(
            stddev=stddev, radius=radius, key=key(42)
        )
        noise_fn_gauss, state_gauss = gaussian_noise(stddev=stddev, key=key(42))

        noisy_rect, _ = noise_fn_rect(grad, state_rect)
        noisy_gauss, _ = noise_fn_gauss(grad, state_gauss)

        # With radius=100σ, virtually no samples are clamped
        assert torch.allclose(noisy_rect, noisy_gauss)
