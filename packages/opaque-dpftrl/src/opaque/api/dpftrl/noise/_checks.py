"""Utilities for validating matrix factorization inputs.

Provides checks for matrices used in the factorization A = B @ C and X = C.T @ C,
ensuring they satisfy expected properties (lower-triangular, symmetric, finite, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _pad(s: str) -> str:
    return s + " " if s else ""


def _array(M: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(M, dtype=np.float64)


def check_lower_triangular(M: ArrayLike, name: str = "", **allclose_kwargs) -> None:
    """Check that M is lower-triangular."""
    M = _array(M)
    if not np.allclose(M, np.tril(M), **allclose_kwargs):
        raise ValueError(f"Matrix {_pad(name)}should be lower-triangular, found\n{M}")


def check_is_matrix(M: ArrayLike, name: str = "") -> None:
    """Check that M is a 2D tensor."""
    M = _array(M)
    if M.ndim != 2:
        raise ValueError(f"Matrix {_pad(name)}has unexpected shape {M.shape}")


def check_square(M: ArrayLike, name: str = "") -> None:
    """Check that M is a square matrix."""
    M = _array(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Matrix {_pad(name)}should be square, found shape {M.shape}")


def check_finite(M: ArrayLike, name: str = "") -> None:
    """Check that all elements of M are finite."""
    M = _array(M)
    if not np.all(np.isfinite(M)):
        raise ValueError(f"Matrix {_pad(name)}is not finite, found\n{M}")


def check_symmetric(M: ArrayLike, name: str = "", **allclose_kwargs) -> None:
    """Check that M is symmetric."""
    M = _array(M)
    if not np.allclose(M, M.T, **allclose_kwargs):
        raise ValueError(f"Matrix {_pad(name)}should be symmetric, found\n{M}")


def check_exactly_one(**kwargs) -> str:
    """Check that exactly one keyword argument is not None.

    Returns the name of the provided argument. Raises ``ValueError`` if
    zero or more than one argument is not None.

    Example:
        >>> which = check_exactly_one(strategy_coef=coef, noising_coef=None)
        >>> assert which == "strategy_coef"
    """
    provided = [k for k, v in kwargs.items() if v is not None]
    if len(provided) != 1:
        names = ", ".join(kwargs.keys())
        raise ValueError(
            f"Specify exactly one of: {names}. "
            f"Got: {', '.join(provided) if provided else 'none'}"
        )
    return provided[0]


def check(
    *,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    C: ArrayLike | None = None,
    X: ArrayLike | None = None,
    **allclose_kwargs,
) -> None:
    """Apply checks to matrices A = B @ C and X = C.T @ C.

    Any subset of the matrices can be provided.

    Args:
        A: The workload matrix.
        B: The decoder matrix, such that A = B @ C.
        C: The encoder (strategy) matrix.
        X: Symmetric Gram matrix C.T @ C.
        **allclose_kwargs: kwargs to pass to numpy.allclose.

    Raises:
        ValueError: if matrices do not satisfy expected properties.
    """
    not_none: dict[str, NDArray[np.float64]] = {}
    n: int | None = None

    if A is not None:
        A = _array(A)
        check_finite(A, "A")
        check_square(A, "A")
        check_lower_triangular(A, "A", **allclose_kwargs)
        not_none["A"] = A
        n = A.shape[0]

    if B is not None:
        B = _array(B)
        check_finite(B, "B")
        check_is_matrix(B, "B")
        if B.shape[0] == B.shape[1]:
            check_lower_triangular(B, "B", **allclose_kwargs)
        not_none["B"] = B
        n = B.shape[0]

    if C is not None:
        C = _array(C)
        check_finite(C, "C")
        check_is_matrix(C, "C")
        if C.shape[0] == C.shape[1]:
            check_lower_triangular(C, "C", **allclose_kwargs)
        not_none["C"] = C
        n = C.shape[1]

    if X is not None:
        X = _array(X)
        check_finite(X, "X")
        check_square(X, "X")
        check_symmetric(X, "X", **allclose_kwargs)
        not_none["X"] = X
        n = X.shape[0]

    if B is not None and C is not None and B.shape[1] != C.shape[0]:
        raise ValueError(
            "B and C shapes do not match. Expected "
            f"B.shape[1] == C.shape[0], but found "
            f"B.shape={B.shape} and C.shape={C.shape}"
        )

    expected_shapes = {
        "A": ("n", "n"),
        "B": ("n", "k"),
        "C": ("k", "n"),
        "X": ("n", "n"),
    }
    correct_shapes = True
    for name_key, m in not_none.items():
        for i, letter in enumerate(expected_shapes[name_key]):
            if (letter == "n") and n is not None and (m.shape[i] != n):
                correct_shapes = False

    if not correct_shapes:
        shapes = {k: v.shape for k, v in not_none.items()}
        raise ValueError(
            f"Expected matrix shapes to match {expected_shapes}, "
            f"but found shapes:\n{shapes}"
        )
