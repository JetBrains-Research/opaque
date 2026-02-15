"""Tests for dense matrix factorization operations."""

import pytest
import torch

from opaque.matrix_factorization.dense import (
    get_orthogonal_mask,
    max_error,
    mean_error,
    per_query_error,
    strategy_from_X,
)


class TestPerQueryError:
    def test_identity_strategy(self):
        """Identity strategy: B = A, error per row = [1, 2, 3, ...]."""
        C = torch.eye(4, dtype=torch.float64)
        error = per_query_error(strategy_matrix=C)
        expected = torch.arange(1, 5, dtype=torch.float64)
        torch.testing.assert_close(error, expected)

    def test_identity_noising(self):
        """Identity noising matrix: same as identity strategy."""
        C_inv = torch.eye(4, dtype=torch.float64)
        error = per_query_error(noising_matrix=C_inv)
        expected = torch.arange(1, 5, dtype=torch.float64)
        torch.testing.assert_close(error, expected)

    def test_both_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            per_query_error(
                strategy_matrix=torch.eye(3),
                noising_matrix=torch.eye(3),
            )

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            per_query_error()


class TestMaxMeanError:
    def test_max_error(self):
        C = torch.eye(4, dtype=torch.float64)
        error = max_error(strategy_matrix=C)
        assert error == pytest.approx(4.0)

    def test_mean_error(self):
        C = torch.eye(4, dtype=torch.float64)
        error = mean_error(strategy_matrix=C)
        assert error == pytest.approx(2.5)  # mean([1,2,3,4])


class TestStrategyFromX:
    def test_identity(self):
        X = torch.eye(3, dtype=torch.float64)
        C = strategy_from_X(X)
        # C.T @ C = I => C = I (up to sign)
        torch.testing.assert_close(C.T @ C, X, atol=1e-10, rtol=1e-10)

    def test_lower_triangular(self):
        M = torch.tril(torch.randn(4, 4, dtype=torch.float64))
        X = M.T @ M
        C = strategy_from_X(X)
        torch.testing.assert_close(C.T @ C, X, atol=1e-8, rtol=1e-8)
        # Verify lower triangular
        torch.testing.assert_close(C, torch.tril(C), atol=1e-10, rtol=1e-10)


class TestOrthogonalMask:
    def test_single_epoch(self):
        mask = get_orthogonal_mask(4, epochs=1)
        expected = torch.ones(4, 4, dtype=torch.float64)
        torch.testing.assert_close(mask, expected)

    def test_two_epochs(self):
        mask = get_orthogonal_mask(4, epochs=2)
        assert mask.shape == (4, 4)
        # Diagonal entries are always 1
        assert mask[0, 0] == 1.0
        # Same within-epoch position across epochs => orthogonality enforced (0)
        assert mask[0, 2] == 0.0
        # Different within-epoch position => free to correlate (1)
        assert mask[0, 1] == 1.0
