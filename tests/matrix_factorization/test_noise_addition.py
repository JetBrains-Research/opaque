"""Tests for matrix factorization noise addition (DP-FTRL privatizer)."""

import pytest
import torch

from opaque.matrix_factorization.streaming_matrix import identity, prefix_sum
from opaque.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)
from opaque.noise.matrix_factorization import (
    Privatizer,
    PrivatizerState,
    gaussian_privatizer,
    matrix_factorization_privatizer,
)


class TestGaussianPrivatizer:
    def test_basic(self):
        privatizer = gaussian_privatizer(stddev=1.0, seed=42)
        params = torch.zeros(10, dtype=torch.float32)
        state = privatizer.init(params)
        assert isinstance(state, PrivatizerState)

    def test_adds_noise(self):
        privatizer = gaussian_privatizer(stddev=1.0, seed=42)
        params = torch.zeros(10, dtype=torch.float32)
        state = privatizer.init(params)
        grad = torch.zeros(10, dtype=torch.float32)
        noisy_grad, new_state = privatizer.update(grad, state)
        # Should have noise added
        assert not torch.allclose(noisy_grad, grad)

    def test_stateful(self):
        """Successive calls produce different noise."""
        privatizer = gaussian_privatizer(stddev=1.0, seed=42)
        params = torch.zeros(10, dtype=torch.float32)
        state = privatizer.init(params)

        grad = torch.zeros(10, dtype=torch.float32)
        noisy1, state = privatizer.update(grad, state)
        noisy2, state = privatizer.update(grad, state)
        assert not torch.allclose(noisy1, noisy2)

    def test_noise_scale(self):
        """Noise scale should be proportional to stddev."""
        torch.manual_seed(42)
        # Large stddev
        privatizer_large = gaussian_privatizer(stddev=100.0, seed=0)
        params = torch.zeros(1000, dtype=torch.float32)
        state = privatizer_large.init(params)
        grad = torch.zeros(1000, dtype=torch.float32)
        noisy, _ = privatizer_large.update(grad, state)
        large_std = noisy.std().item()

        # Small stddev
        privatizer_small = gaussian_privatizer(stddev=1.0, seed=1)
        state = privatizer_small.init(params)
        noisy, _ = privatizer_small.update(grad, state)
        small_std = noisy.std().item()

        assert large_std > small_std * 10


class TestDenseMatrixFactorizationPrivatizer:
    def test_identity_matrix(self):
        """Identity noising matrix = standard Gaussian."""
        noising = torch.eye(5, dtype=torch.float64)
        privatizer = matrix_factorization_privatizer(noising, stddev=1.0, seed=42)
        grad = torch.zeros(10, dtype=torch.float64)
        state = privatizer.init(grad)
        noisy, state = privatizer.update(grad, state)
        assert noisy.shape == grad.shape

    def test_stepping(self):
        """State advances through matrix rows."""
        noising = torch.eye(3, dtype=torch.float64) * 2.0
        privatizer = matrix_factorization_privatizer(noising, stddev=1.0, seed=42)
        grad = torch.zeros(5, dtype=torch.float64)
        state = privatizer.init(grad)
        assert state.inner_state.item() == 0
        _, state = privatizer.update(grad, state)
        assert state.inner_state.item() == 1
        _, state = privatizer.update(grad, state)
        assert state.inner_state.item() == 2

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="2D"):
            matrix_factorization_privatizer(torch.ones(5), stddev=1.0)


class TestStreamingMatrixFactorizationPrivatizer:
    def test_identity_streaming(self):
        """Identity StreamingMatrix = standard Gaussian."""
        noising = identity()
        privatizer = matrix_factorization_privatizer(noising, stddev=1.0, seed=42)
        grad = torch.zeros(10, dtype=torch.float32)
        state = privatizer.init(grad)
        noisy, state = privatizer.update(grad, state)
        assert noisy.shape == grad.shape

    def test_toeplitz_noising(self):
        """Test with a BandMF-style Toeplitz noising matrix."""
        coefs = optimal_max_error_strategy_coefs(5)
        noising = inverse_as_streaming_matrix(coefs)
        privatizer = matrix_factorization_privatizer(noising, stddev=1.0, seed=42)
        grad = torch.zeros(10, dtype=torch.float32)
        state = privatizer.init(grad)
        for _ in range(5):
            noisy, state = privatizer.update(grad, state)
            assert noisy.shape == grad.shape

    def test_pytree_grads(self):
        """Test with dict-structured gradients."""
        noising = identity()
        privatizer = matrix_factorization_privatizer(noising, stddev=1.0, seed=42)
        grad = {
            "weight": torch.zeros(5, 3, dtype=torch.float32),
            "bias": torch.zeros(3, dtype=torch.float32),
        }
        state = privatizer.init(grad)
        noisy, state = privatizer.update(grad, state)
        assert isinstance(noisy, dict)
        assert noisy["weight"].shape == (5, 3)
        assert noisy["bias"].shape == (3,)

    def test_noise_is_correlated(self):
        """With prefix sum noising, noise should accumulate."""
        noising = prefix_sum()
        privatizer = matrix_factorization_privatizer(noising, stddev=1.0, seed=42)
        grad = torch.zeros(50, dtype=torch.float64)
        state = privatizer.init(grad)

        variances = []
        for _ in range(20):
            noisy, state = privatizer.update(grad, state)
            variances.append(noisy.var().item())

        # With prefix sum, variance should increase
        assert variances[-1] > variances[0]
