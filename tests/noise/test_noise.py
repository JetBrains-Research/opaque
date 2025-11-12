"""Unit tests for noise module."""

import pytest
import scipy.stats
import torch

from opaque.noise import add_gaussian_noise


def test_add_noise_single_tensor():
    """Basic noise addition to single tensor."""
    grad = torch.randn(10, 5)
    stddev = 1.0
    generator = torch.Generator().manual_seed(42)

    noisy = add_gaussian_noise(grad, stddev, generator)

    assert noisy.shape == grad.shape
    assert noisy.dtype == grad.dtype
    assert not torch.allclose(noisy, grad)


def test_add_noise_pytree():
    """Noise addition to PyTree (dict of tensors)."""
    grads = {
        "weight": torch.randn(10, 5),
        "bias": torch.randn(10),
    }
    noisy = add_gaussian_noise(grads, stddev=1.0)

    assert set(noisy.keys()) == set(grads.keys())
    assert noisy["weight"].shape == grads["weight"].shape
    assert noisy["bias"].shape == grads["bias"].shape


def test_zero_stddev():
    """stddev=0 should return original gradients."""
    grad = torch.randn(5, 3)
    noisy = add_gaussian_noise(grad, stddev=0.0)
    assert torch.equal(noisy, grad)


def test_negative_stddev_raises():
    """Negative stddev should raise ValueError."""
    grad = torch.randn(5, 3)
    with pytest.raises(ValueError, match="stddev must be non-negative"):
        add_gaussian_noise(grad, stddev=-1.0)


def test_dtype_preservation():
    """Noise should preserve input dtype."""
    for dtype in [torch.float32, torch.float64]:
        grad = torch.randn(10, 5, dtype=dtype)
        noisy = add_gaussian_noise(grad, stddev=1.0)
        assert noisy.dtype == dtype


def test_device_preservation():
    """Noise should preserve input device."""
    grad = torch.randn(10, 5)
    noisy = add_gaussian_noise(grad, stddev=1.0)
    assert noisy.device == grad.device


def test_noise_normality():
    """Verify noise follows N(0, stddev²) using K-S test."""
    stddev = 1.5
    generator = torch.Generator().manual_seed(42)

    # Generate many samples
    n_samples = 10000
    grad = torch.zeros(n_samples)
    noisy = add_gaussian_noise(grad, stddev, generator)
    noise = (noisy - grad).numpy()

    # Kolmogorov-Smirnov test
    # H0: noise ~ N(0, stddev²)
    _, p_value = scipy.stats.kstest(noise, scipy.stats.norm(0, stddev).cdf)

    # Accept at 5% significance level
    assert p_value > 0.025


def test_noise_stddev():
    """Empirical stddev should match specified stddev."""
    stddev = 2.0
    generator = torch.Generator().manual_seed(123)

    # Many samples
    n_samples = 50000
    grad = torch.zeros(n_samples)
    noisy = add_gaussian_noise(grad, stddev, generator)
    noise = noisy - grad

    empirical_std = noise.std().item()

    # Within 5% (statistical variation)
    assert abs(empirical_std - stddev) / stddev < 0.05


def test_reproducibility():
    """Same seed produces same noise."""
    grad = torch.randn(100, 50)
    stddev = 1.0

    gen1 = torch.Generator().manual_seed(42)
    noisy1 = add_gaussian_noise(grad, stddev, gen1)

    gen2 = torch.Generator().manual_seed(42)
    noisy2 = add_gaussian_noise(grad, stddev, gen2)

    assert torch.equal(noisy1, noisy2)


def test_uniqueness():
    """Different calls produce different noise."""
    grad = torch.randn(100, 50)
    generator = torch.Generator().manual_seed(42)

    noisy1 = add_gaussian_noise(grad, 1.0, generator)
    noisy2 = add_gaussian_noise(grad, 1.0, generator)

    assert not torch.allclose(noisy1, noisy2, atol=1e-6)


def test_nested_pytree():
    """Test with nested PyTree structure."""
    grads = {
        "layer1": {"weight": torch.randn(10, 5), "bias": torch.randn(10)},
        "layer2": {"weight": torch.randn(5, 3), "bias": torch.randn(5)},
    }

    noisy = add_gaussian_noise(grads, stddev=1.0)

    assert "layer1" in noisy
    assert "layer2" in noisy
    assert noisy["layer1"]["weight"].shape == grads["layer1"]["weight"].shape
    assert noisy["layer2"]["bias"].shape == grads["layer2"]["bias"].shape


def test_tuple_pytree():
    """Test with tuple PyTree structure."""
    grads = (torch.randn(10, 5), torch.randn(10))
    noisy = add_gaussian_noise(grads, stddev=1.0)

    assert len(noisy) == len(grads)
    assert noisy[0].shape == grads[0].shape
    assert noisy[1].shape == grads[1].shape
