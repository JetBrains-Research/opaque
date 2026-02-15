"""Tests for matrix factorization noise addition (DP-FTRL noise)."""

import pytest
import torch

from opaque.matrix_factorization.streaming_matrix import identity, prefix_sum
from opaque.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)
from opaque.noise.matrix_factorization import (
    MFNoiseState,
    matrix_factorization_noise,
)


class TestDenseMatrixFactorizationNoise:
    def test_identity_matrix(self):
        """Identity noising matrix = standard Gaussian."""
        noising = torch.eye(5, dtype=torch.float64)
        init_fn, noise_fn = matrix_factorization_noise(noising, stddev=1.0, seed=42)
        grad = torch.zeros(10, dtype=torch.float64)
        state = init_fn(grad)
        noisy, state = noise_fn(grad, state)
        assert noisy.shape == grad.shape

    def test_stepping(self):
        """State advances through matrix rows."""
        noising = torch.eye(3, dtype=torch.float64) * 2.0
        init_fn, noise_fn = matrix_factorization_noise(noising, stddev=1.0, seed=42)
        grad = torch.zeros(5, dtype=torch.float64)
        state = init_fn(grad)
        assert state.inner_state.item() == 0
        _, state = noise_fn(grad, state)
        assert state.inner_state.item() == 1
        _, state = noise_fn(grad, state)
        assert state.inner_state.item() == 2

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="2D"):
            matrix_factorization_noise(torch.ones(5), stddev=1.0)


class TestStreamingMatrixFactorizationNoise:
    def test_identity_streaming(self):
        """Identity StreamingMatrix = standard Gaussian."""
        noising = identity()
        init_fn, noise_fn = matrix_factorization_noise(noising, stddev=1.0, seed=42)
        grad = torch.zeros(10, dtype=torch.float32)
        state = init_fn(grad)
        noisy, state = noise_fn(grad, state)
        assert noisy.shape == grad.shape

    def test_adds_noise(self):
        """Noise is actually added to gradients."""
        init_fn, noise_fn = matrix_factorization_noise(
            identity(), stddev=1.0, seed=42
        )
        grad = torch.zeros(10, dtype=torch.float32)
        state = init_fn(grad)
        noisy, _ = noise_fn(grad, state)
        assert not torch.allclose(noisy, grad)

    def test_stateful(self):
        """Successive calls produce different noise."""
        init_fn, noise_fn = matrix_factorization_noise(
            identity(), stddev=1.0, seed=42
        )
        grad = torch.zeros(10, dtype=torch.float32)
        state = init_fn(grad)
        noisy1, state = noise_fn(grad, state)
        noisy2, state = noise_fn(grad, state)
        assert not torch.allclose(noisy1, noisy2)

    def test_noise_scale(self):
        """Noise scale should be proportional to stddev."""
        grad = torch.zeros(1000, dtype=torch.float32)

        # Large stddev
        init_fn, noise_fn = matrix_factorization_noise(
            identity(), stddev=100.0, seed=0
        )
        state = init_fn(grad)
        noisy, _ = noise_fn(grad, state)
        large_std = noisy.std().item()

        # Small stddev
        init_fn, noise_fn = matrix_factorization_noise(
            identity(), stddev=1.0, seed=1
        )
        state = init_fn(grad)
        noisy, _ = noise_fn(grad, state)
        small_std = noisy.std().item()

        assert large_std > small_std * 10

    def test_toeplitz_noising(self):
        """Test with a BandMF-style Toeplitz noising matrix."""
        coefs = optimal_max_error_strategy_coefs(5)
        noising = inverse_as_streaming_matrix(coefs)
        init_fn, noise_fn = matrix_factorization_noise(noising, stddev=1.0, seed=42)
        grad = torch.zeros(10, dtype=torch.float32)
        state = init_fn(grad)
        for _ in range(5):
            noisy, state = noise_fn(grad, state)
            assert noisy.shape == grad.shape

    def test_pytree_grads(self):
        """Test with dict-structured gradients."""
        noising = identity()
        init_fn, noise_fn = matrix_factorization_noise(noising, stddev=1.0, seed=42)
        grad = {
            "weight": torch.zeros(5, 3, dtype=torch.float32),
            "bias": torch.zeros(3, dtype=torch.float32),
        }
        state = init_fn(grad)
        noisy, state = noise_fn(grad, state)
        assert isinstance(noisy, dict)
        assert noisy["weight"].shape == (5, 3)
        assert noisy["bias"].shape == (3,)

    def test_noise_is_correlated(self):
        """With prefix sum noising, noise should accumulate."""
        noising = prefix_sum()
        init_fn, noise_fn = matrix_factorization_noise(noising, stddev=1.0, seed=42)
        grad = torch.zeros(50, dtype=torch.float64)
        state = init_fn(grad)

        variances = []
        for _ in range(20):
            noisy, state = noise_fn(grad, state)
            variances.append(noisy.var().item())

        # With prefix sum, variance should increase
        assert variances[-1] > variances[0]

    def test_init_returns_mf_noise_state(self):
        """init_fn returns MFNoiseState."""
        init_fn, _ = matrix_factorization_noise(identity(), stddev=1.0, seed=42)
        state = init_fn(torch.zeros(10))
        assert isinstance(state, MFNoiseState)
