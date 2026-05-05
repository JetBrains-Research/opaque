"""Tests for matrix factorization noise addition (DP-FTRL noise)."""

import pytest
import torch

from opaque.dpftrl.noise._engine import MFNoiseState, _matrix_factorization_noise
from opaque.dpftrl.noise._streaming_matrix import identity, prefix_sum
from opaque.dpftrl.noise._toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)
from opaque.random import key


class TestDenseMatrixFactorizationNoise:
    def test_identity_matrix(self):
        """Identity noising matrix = standard Gaussian."""
        noising = torch.eye(5, dtype=torch.float64)
        grad = torch.zeros(10, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        noisy, state = noise_fn(grad, state, stddev=1.0)
        assert noisy.shape == grad.shape

    def test_stepping(self):
        """State advances through matrix rows."""
        noising = torch.eye(3, dtype=torch.float64) * 2.0
        grad = torch.zeros(5, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        assert state._inner_state.item() == 0
        _, state = noise_fn(grad, state, stddev=1.0)
        assert state._inner_state.item() == 1
        _, state = noise_fn(grad, state, stddev=1.0)
        assert state._inner_state.item() == 2

    def test_invalid_ndim(self):
        grad = torch.zeros(10)
        with pytest.raises(ValueError, match="2D"):
            _matrix_factorization_noise(grad, torch.ones(5), key=key(42))


class TestStreamingMatrixFactorizationNoise:
    def test_identity_streaming(self):
        """Identity StreamingMatrix = standard Gaussian."""
        noising = identity()
        grad = torch.zeros(10, dtype=torch.float32)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        noisy, state = noise_fn(grad, state, stddev=1.0)
        assert noisy.shape == grad.shape

    def test_adds_noise(self):
        """Noise is actually added to gradients."""
        grad = torch.zeros(10, dtype=torch.float32)
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(42))
        noisy, _ = noise_fn(grad, state, stddev=1.0)
        assert not torch.allclose(noisy, grad)

    def test_stateful(self):
        """Successive calls produce different noise."""
        grad = torch.zeros(10, dtype=torch.float32)
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(42))
        noisy1, state = noise_fn(grad, state, stddev=1.0)
        noisy2, state = noise_fn(grad, state, stddev=1.0)
        assert not torch.allclose(noisy1, noisy2)

    def test_noise_scale(self):
        """Noise scale should be proportional to stddev."""
        grad = torch.zeros(1000, dtype=torch.float32)

        # Large stddev
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(0))
        noisy, _ = noise_fn(grad, state, stddev=100.0)
        large_std = noisy.std().item()

        # Small stddev
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(1))
        noisy, _ = noise_fn(grad, state, stddev=1.0)
        small_std = noisy.std().item()

        assert large_std > small_std * 10

    def test_toeplitz_noising(self):
        """Test with a BandMF-style Toeplitz noising matrix."""
        coefs = optimal_max_error_strategy_coefs(5)
        noising = inverse_as_streaming_matrix(coefs)
        grad = torch.zeros(10, dtype=torch.float32)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        for _ in range(5):
            noisy, state = noise_fn(grad, state, stddev=1.0)
            assert noisy.shape == grad.shape

    def test_pytree_grads(self):
        """Test with dict-structured gradients."""
        noising = identity()
        grad = {
            "weight": torch.zeros(5, 3, dtype=torch.float32),
            "bias": torch.zeros(3, dtype=torch.float32),
        }
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))
        noisy, state = noise_fn(grad, state, stddev=1.0)
        assert isinstance(noisy, dict)
        assert noisy["weight"].shape == (5, 3)
        assert noisy["bias"].shape == (3,)

    def test_noise_is_correlated(self):
        """With prefix sum noising, noise should accumulate."""
        noising = prefix_sum()
        grad = torch.zeros(50, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(grad, noising, key=key(42))

        variances = []
        for _ in range(20):
            noisy, state = noise_fn(grad, state, stddev=1.0)
            variances.append(noisy.var().item())

        # With prefix sum, variance should increase
        assert variances[-1] > variances[0]

    def test_returns_mf_noise_state(self):
        """_matrix_factorization_noise returns MFNoiseState."""
        grad = torch.zeros(10)
        noise_fn, state = _matrix_factorization_noise(grad, identity(), key=key(42))
        assert isinstance(state, MFNoiseState)
