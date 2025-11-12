"""JAX validation tests for noise module.

These tests verify that Opaque's noise generation matches JAX-Privacy's behavior.
"""

import pytest
import torch

# Import JAX (skip if not available)
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from opaque.noise import add_gaussian_noise

pytestmark = [pytest.mark.jax_validation]


def test_gaussian_noise_single_tensor():
    """Test that Gaussian noise matches JAX for single tensors."""
    # Set up
    seed = 42
    stddev = 1.5
    shape = (100, 50)

    # JAX version
    jax_key = jax.random.PRNGKey(seed)
    jax_grad = jnp.zeros(shape, dtype=jnp.float32)
    jax_noise = jax.random.normal(jax_key, shape) * stddev
    jax_noisy = jax_grad + jax_noise

    # Opaque version
    torch_grad = torch.zeros(shape, dtype=torch.float32)
    torch_gen = torch.Generator().manual_seed(seed)
    torch_noisy = add_gaussian_noise(torch_grad, stddev, torch_gen)

    # Both should have same shape and dtype
    assert torch_noisy.shape == tuple(jax_noisy.shape)
    assert torch_noisy.dtype == torch.float32

    # Statistics should be similar (but not exactly equal due to different PRNG)
    torch_std = torch_noisy.std().item()
    jax_std = float(jax_noisy.std())

    # Within 10% due to random sampling
    assert abs(torch_std - jax_std) / jax_std < 0.1


def test_gaussian_noise_pytree():
    """Test that Gaussian noise works with PyTrees like JAX."""
    seed = 123
    stddev = 2.0

    # JAX PyTree
    jax_key = jax.random.PRNGKey(seed)
    jax_tree = {
        "weight": jnp.zeros((10, 5), dtype=jnp.float32),
        "bias": jnp.zeros((5,), dtype=jnp.float32),
    }

    # Add noise to each leaf
    jax_keys = jax.random.split(jax_key, 2)
    jax_noisy = {
        "weight": jax_tree["weight"] + jax.random.normal(jax_keys[0], (10, 5)) * stddev,
        "bias": jax_tree["bias"] + jax.random.normal(jax_keys[1], (5,)) * stddev,
    }

    # Opaque PyTree
    torch_tree = {
        "weight": torch.zeros((10, 5), dtype=torch.float32),
        "bias": torch.zeros((5,), dtype=torch.float32),
    }
    torch_gen = torch.Generator().manual_seed(seed)
    torch_noisy = add_gaussian_noise(torch_tree, stddev, torch_gen)

    # Check shapes match
    assert torch_noisy["weight"].shape == tuple(jax_noisy["weight"].shape)
    assert torch_noisy["bias"].shape == tuple(jax_noisy["bias"].shape)

    # Check dtypes match
    assert torch_noisy["weight"].dtype == torch.float32
    assert torch_noisy["bias"].dtype == torch.float32


def test_gaussian_noise_statistics():
    """Test that noise has correct statistical properties like JAX."""
    seed = 456
    stddev = 3.0
    n_samples = 10000

    # JAX noise
    jax_key = jax.random.PRNGKey(seed)
    jax_noise = jax.random.normal(jax_key, (n_samples,)) * stddev
    jax_mean = float(jax_noise.mean())
    jax_std = float(jax_noise.std())

    # Opaque noise
    torch_gen = torch.Generator().manual_seed(seed)
    torch_grad = torch.zeros(n_samples)
    torch_noisy = add_gaussian_noise(torch_grad, stddev, torch_gen)
    torch_noise = torch_noisy - torch_grad
    torch_mean = torch_noise.mean().item()
    torch_std = torch_noise.std().item()

    # Both should be approximately N(0, stddev)
    assert abs(jax_mean) < 0.1
    assert abs(torch_mean) < 0.1
    assert abs(jax_std - stddev) < 0.1
    assert abs(torch_std - stddev) < 0.1


def test_gaussian_noise_dtype_preservation():
    """Test that noise preserves dtypes like JAX."""
    seed = 789
    stddev = 1.0

    # Only test float32 (JAX requires JAX_ENABLE_X64 for float64)
    dtype_jax = jnp.float32
    dtype_torch = torch.float32

    # JAX
    jax_key = jax.random.PRNGKey(seed)
    jax_grad = jnp.zeros((10, 5), dtype=dtype_jax)
    jax_noise = jax.random.normal(jax_key, (10, 5)) * stddev
    jax_noisy = jax_grad + jax_noise.astype(dtype_jax)

    # Opaque
    torch_grad = torch.zeros((10, 5), dtype=dtype_torch)
    torch_gen = torch.Generator().manual_seed(seed)
    torch_noisy = add_gaussian_noise(torch_grad, stddev, torch_gen)

    # Check dtype preserved
    assert jax_noisy.dtype == dtype_jax
    assert torch_noisy.dtype == dtype_torch


def test_gaussian_noise_zero_stddev():
    """Test that zero stddev returns original values like JAX."""
    seed = 111
    stddev = 0.0

    # JAX
    jax_grad = jnp.ones((5, 3), dtype=jnp.float32)
    jax_key = jax.random.PRNGKey(seed)
    jax_noise = jax.random.normal(jax_key, (5, 3)) * stddev
    jax_noisy = jax_grad + jax_noise

    # Opaque
    torch_grad = torch.ones((5, 3), dtype=torch.float32)
    torch_noisy = add_gaussian_noise(torch_grad, stddev)

    # Both should equal original
    assert jnp.allclose(jax_noisy, jax_grad)
    assert torch.equal(torch_noisy, torch_grad)


def test_gaussian_noise_reproducibility():
    """Test that same seed gives same noise like JAX."""
    seed = 222
    stddev = 1.0
    shape = (20, 10)

    # JAX reproducibility
    jax_key1 = jax.random.PRNGKey(seed)
    jax_noise1 = jax.random.normal(jax_key1, shape) * stddev

    jax_key2 = jax.random.PRNGKey(seed)
    jax_noise2 = jax.random.normal(jax_key2, shape) * stddev

    assert jnp.array_equal(jax_noise1, jax_noise2)

    # Opaque reproducibility
    torch_grad = torch.zeros(shape)

    torch_gen1 = torch.Generator().manual_seed(seed)
    torch_noisy1 = add_gaussian_noise(torch_grad, stddev, torch_gen1)

    torch_gen2 = torch.Generator().manual_seed(seed)
    torch_noisy2 = add_gaussian_noise(torch_grad, stddev, torch_gen2)

    assert torch.equal(torch_noisy1, torch_noisy2)


def test_gaussian_noise_nested_pytree():
    """Test noise with nested PyTrees like JAX."""
    seed = 333
    stddev = 1.5

    # JAX nested PyTree
    jax_key = jax.random.PRNGKey(seed)
    jax_tree = {
        "layer1": {"w": jnp.zeros((5, 3)), "b": jnp.zeros((3,))},
        "layer2": jnp.zeros((10,)),
    }

    # For JAX, we'd need to flatten and add noise
    # Simplified comparison - just check structure

    # Opaque nested PyTree
    torch_tree = {
        "layer1": {"w": torch.zeros((5, 3)), "b": torch.zeros((3,))},
        "layer2": torch.zeros((10,)),
    }
    torch_gen = torch.Generator().manual_seed(seed)
    torch_noisy = add_gaussian_noise(torch_tree, stddev, torch_gen)

    # Check structure preserved
    assert "layer1" in torch_noisy
    assert "w" in torch_noisy["layer1"]
    assert "b" in torch_noisy["layer1"]
    assert "layer2" in torch_noisy

    # Check shapes preserved
    assert torch_noisy["layer1"]["w"].shape == (5, 3)
    assert torch_noisy["layer1"]["b"].shape == (3,)
    assert torch_noisy["layer2"].shape == (10,)
