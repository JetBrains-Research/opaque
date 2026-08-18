"""Tests for the StreamingMatrix interface."""

import numpy as np

from opaque.api.dpftrl.noise._streaming_matrix import (
    StreamingMatrix,
    diagonal,
    identity,
    momentum_sgd_matrix,
    multiply_array,
    multiply_streaming_matrices,
    prefix_sum,
)


class TestIdentity:
    def test_identity_preserves_input(self):
        ident = identity()
        x = np.arange(15, dtype=np.float64).reshape(5, 3)
        result = multiply_array(ident, x)
        np.testing.assert_allclose(result, x)

    def test_identity_materialize(self):
        ident = identity()
        M = ident.materialize(4)
        np.testing.assert_allclose(M, np.eye(4, dtype=np.float64))


class TestPrefixSum:
    def test_prefix_sum_basic(self):
        P = prefix_sum()
        x = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
        result = multiply_array(P, x)
        expected = np.array([[1.0], [3.0], [6.0], [10.0]], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_prefix_sum_materialize(self):
        P = prefix_sum()
        M = P.materialize(4)
        expected = np.tril(np.ones((4, 4), dtype=np.float64))
        np.testing.assert_allclose(M, expected)

    def test_prefix_sum_row_norms(self):
        P = prefix_sum()
        norms = P.row_norms_squared(4)
        # Row i of tri(4) has (i+1) ones => L2^2 = i+1
        expected = np.arange(1, 5, dtype=np.float64)
        np.testing.assert_allclose(norms, expected)


class TestDiagonal:
    def test_diagonal_scales(self):
        d = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        D = diagonal(d)
        x = np.ones((3, 1), dtype=np.float64)
        result = multiply_array(D, x)
        expected = np.expand_dims(d, axis=1)
        np.testing.assert_allclose(result, expected)


class TestMomentum:
    def test_momentum_and_learning_rate_schedule(self):
        matrix = momentum_sgd_matrix(
            momentum=0.5,
            learning_rates=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        result = multiply_array(matrix, np.ones((3, 1), dtype=np.float64))
        expected = np.array([[1.0], [4.0], [9.25]], dtype=np.float64)
        np.testing.assert_allclose(result, expected)


class TestComposition:
    def test_identity_composition(self):
        ident = identity()
        P = prefix_sum()
        IP = multiply_streaming_matrices(ident, P)
        x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        result1 = multiply_array(P, x)
        result2 = multiply_array(IP, x)
        np.testing.assert_allclose(result1, result2)

    def test_prefix_sum_composition(self):
        P1 = prefix_sum()
        P2 = prefix_sum()
        PP = multiply_streaming_matrices(P1, P2)
        M = PP.materialize(4)
        expected = np.tril(np.ones((4, 4), dtype=np.float64))
        expected = expected @ expected
        np.testing.assert_allclose(M, expected)


class TestMatMul:
    def test_matmul_streaming(self):
        P = prefix_sum()
        ident = identity()
        result = P @ ident
        M = result.materialize(3)
        expected = np.tril(np.ones((3, 3), dtype=np.float64))
        np.testing.assert_allclose(M, expected)

    def test_matmul_tensor(self):
        P = prefix_sum()
        x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        result = P @ x
        expected = np.array([[1.0], [3.0], [6.0]], dtype=np.float64)
        np.testing.assert_allclose(result, expected)


class TestFromArrayImplementation:
    def test_custom_streaming_matrix(self):
        """Test creating a custom StreamingMatrix (doubling matrix)."""

        def init(abstract_value):
            return np.zeros_like(abstract_value)

        def next_fn(value, state):
            return value * 2, state

        M = StreamingMatrix.from_array_implementation(init, next_fn)
        x = np.ones((3, 1), dtype=np.float64)
        result = multiply_array(M, x)
        expected = np.ones((3, 1), dtype=np.float64) * 2
        np.testing.assert_allclose(result, expected)
