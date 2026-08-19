"""Tests for matrix factorization input validation (checks.py)."""

import numpy as np
import pytest

from opaque.api.dpftrl.noise._checks import (
    check,
    check_exactly_one,
    check_finite,
    check_lower_triangular,
    check_square,
    check_symmetric,
)


class TestCheckLowerTriangular:
    def test_valid_lower_triangular(self):
        M = np.tril(np.ones((3, 3)))
        check_lower_triangular(M)  # Should not raise

    def test_upper_triangular_fails(self):
        M = np.triu(np.ones((3, 3)))
        M[0, 1] = 1.0
        with pytest.raises(ValueError, match="lower-triangular"):
            check_lower_triangular(M)

    def test_identity_is_valid(self):
        check_lower_triangular(np.eye(4))

    def test_named_error_message(self):
        M = np.ones((2, 2))
        with pytest.raises(ValueError, match="Matrix A"):
            check_lower_triangular(M, "A")


class TestCheckSquare:
    def test_square(self):
        check_square(np.zeros((3, 3)))

    def test_non_square_fails(self):
        with pytest.raises(ValueError, match="square"):
            check_square(np.zeros((3, 4)))

    def test_1d_fails(self):
        with pytest.raises(ValueError, match="square"):
            check_square(np.zeros(3))


class TestCheckFinite:
    def test_finite(self):
        check_finite(np.ones((3, 3)), "M")

    def test_inf_fails(self):
        M = np.ones((3, 3))
        M[1, 1] = float("inf")
        with pytest.raises(ValueError, match="not finite"):
            check_finite(M, "M")

    def test_nan_fails(self):
        M = np.ones((3, 3))
        M[0, 0] = float("nan")
        with pytest.raises(ValueError, match="not finite"):
            check_finite(M, "M")


class TestCheckSymmetric:
    def test_symmetric(self):
        M = np.array([[1.0, 2.0], [2.0, 3.0]])
        check_symmetric(M, "M")

    def test_asymmetric_fails(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="symmetric"):
            check_symmetric(M, "M")


class TestCheckExactlyOne:
    def test_one_provided(self):
        result = check_exactly_one(a=1, b=None)
        assert result == "a"

    def test_none_provided(self):
        with pytest.raises(ValueError, match="exactly one"):
            check_exactly_one(a=None, b=None)

    def test_both_provided(self):
        with pytest.raises(ValueError, match="exactly one"):
            check_exactly_one(a=1, b=2)

    def test_returns_name(self):
        result = check_exactly_one(strategy_coef=None, noising_coef="value")
        assert result == "noising_coef"


class TestCheck:
    def test_valid_A(self):
        A = np.tril(np.ones((3, 3)))
        check(A=A)

    def test_valid_C(self):
        C = np.tril(np.random.default_rng(0).normal(size=(3, 3)))
        check(C=C)

    def test_valid_X(self):
        M = np.random.default_rng(0).normal(size=(3, 3))
        X = M.T @ M
        check(X=X)

    def test_shape_mismatch_B_C(self):
        rng = np.random.default_rng(0)
        B = rng.normal(size=(3, 4))
        C = rng.normal(size=(5, 3))  # B.shape[1]=4, C.shape[0]=5 mismatch
        with pytest.raises(ValueError, match="shapes do not match"):
            check(B=B, C=C)

    def test_empty_check_ok(self):
        check()  # No matrices, should not raise
