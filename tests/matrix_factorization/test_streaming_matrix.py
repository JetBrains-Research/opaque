"""Tests for the StreamingMatrix interface."""

import pytest
import torch

from opaque.noise.matrix_factorization.streaming_matrix import (
    StreamingMatrix,
    diagonal,
    identity,
    multiply_array,
    multiply_streaming_matrices,
    prefix_sum,
)


class TestIdentity:
    def test_identity_preserves_input(self):
        ident = identity()
        x = torch.randn(5, 3, dtype=torch.float64)
        result = multiply_array(ident, x)
        torch.testing.assert_close(result, x)

    def test_identity_materialize(self):
        ident = identity()
        M = ident.materialize(4)
        torch.testing.assert_close(M, torch.eye(4, dtype=torch.float64))


class TestPrefixSum:
    def test_prefix_sum_basic(self):
        P = prefix_sum()
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
        result = multiply_array(P, x)
        expected = torch.tensor([[1.0], [3.0], [6.0], [10.0]], dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_prefix_sum_materialize(self):
        P = prefix_sum()
        M = P.materialize(4)
        expected = torch.tril(torch.ones(4, 4, dtype=torch.float64))
        torch.testing.assert_close(M, expected)

    def test_prefix_sum_row_norms(self):
        P = prefix_sum()
        norms = P.row_norms_squared(4)
        # Row i of tri(4) has (i+1) ones => L2^2 = i+1
        expected = torch.arange(1, 5, dtype=torch.float64)
        torch.testing.assert_close(norms, expected)


class TestDiagonal:
    def test_diagonal_scales(self):
        d = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        D = diagonal(d)
        x = torch.ones(3, 1, dtype=torch.float64)
        result = multiply_array(D, x)
        expected = d.unsqueeze(1)
        torch.testing.assert_close(result, expected)


class TestComposition:
    def test_identity_composition(self):
        ident = identity()
        P = prefix_sum()
        IP = multiply_streaming_matrices(ident, P)
        x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        result1 = multiply_array(P, x)
        result2 = multiply_array(IP, x)
        torch.testing.assert_close(result1, result2)

    def test_prefix_sum_composition(self):
        P1 = prefix_sum()
        P2 = prefix_sum()
        PP = multiply_streaming_matrices(P1, P2)
        M = PP.materialize(4)
        expected = torch.tril(torch.ones(4, 4, dtype=torch.float64))
        expected = expected @ expected
        torch.testing.assert_close(M, expected)


class TestMatMul:
    def test_matmul_streaming(self):
        P = prefix_sum()
        ident = identity()
        result = P @ ident
        M = result.materialize(3)
        expected = torch.tril(torch.ones(3, 3, dtype=torch.float64))
        torch.testing.assert_close(M, expected)

    def test_matmul_tensor(self):
        P = prefix_sum()
        x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        result = P @ x
        expected = torch.tensor([[1.0], [3.0], [6.0]], dtype=torch.float64)
        torch.testing.assert_close(result, expected)


class TestFromArrayImplementation:
    def test_custom_streaming_matrix(self):
        """Test creating a custom StreamingMatrix (doubling matrix)."""

        def init(abstract_value):
            return torch.zeros_like(abstract_value)

        def next_fn(value, state):
            return value * 2, state

        M = StreamingMatrix.from_array_implementation(init, next_fn)
        x = torch.ones(3, 1, dtype=torch.float64)
        result = multiply_array(M, x)
        expected = torch.ones(3, 1, dtype=torch.float64) * 2
        torch.testing.assert_close(result, expected)
