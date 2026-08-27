"""Library for computing sensitivity under multiple participation patterns.

Implements sensitivity computations for matrix factorization mechanisms
under various participation schemas:
- Single participation (one gradient per user)
- Min-sep participation (minimum separation between participations)

References:
    - Algorithm 3 (VecSens): https://arxiv.org/abs/2306.08153
    - Algorithm 4 (Efficient sensitivity UB): https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _checks as checks

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "get_min_sep_sensitivity_upper_bound",
    "get_sensitivity_banded",
    "max_participation_for_linear_fn",
    "minsep_true_max_participations",
    "single_participation_sensitivity",
]


def single_participation_sensitivity(C: ArrayLike) -> float:
    """Returns the L2 sensitivity of a matrix with single participation.

    Args:
        C: The encoder (strategy) matrix, shape (k, n).

    Returns:
        The maximum L2 column norm of C.
    """
    C = np.asarray(C, dtype=np.float64)
    checks.check(C=C)
    return float(np.linalg.norm(C, axis=0).max())


def _ceil_div(x: int, y: int) -> int:
    """Integer division, rounding up."""
    return -(x // -y)


def minsep_true_max_participations(
    n: int, min_sep: int, max_participations: int | None = None
) -> int:
    """Returns the maximum number of participations for a min_sep pattern.

    This may be less than ``max_participations`` if n is too small.

    Args:
        n: Number of rounds.
        min_sep: Minimum separation between participations (min_sep=1 means
            adjacent indices can be selected).
        max_participations: Optional upper bound on participations.

    Returns:
        The largest possible number of participations.
    """
    max_part_ub = _ceil_div(n, min_sep)
    if max_participations is None:
        return max_part_ub
    return min(max_participations, max_part_ub)


def max_participation_for_linear_fn(
    x: ArrayLike,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Returns max_u <x, u> where u respects the participation pattern.

    Solves the optimization problem using dynamic programming with
    running time O(len(x) * max_participations).

    Reference: Algorithm 3 (VecSens) from https://arxiv.org/abs/2306.08153

    Args:
        x: A vector of values to optimize over.
        min_sep: Minimum separation between selected indices.
        max_participations: Maximum number of participations.

    Returns:
        The optimal value.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    max_participations = minsep_true_max_participations(n, min_sep, max_participations)

    # f = F[:, k], with extra padding for boundary conditions
    f = np.zeros(n + min_sep, dtype=np.float64)
    for _ in range(max_participations):
        # f[i] = x[i] + f[i + min_sep] (selecting x[i])
        f[:n] = x + f[min_sep : min_sep + n]
        # Accumulate max right-to-left: cummax in reverse
        f = np.maximum.accumulate(f[::-1])[::-1]
    return float(f[0])


def banded_lower_triangular_mask(n: int, num_bands: int) -> NDArray[np.int32]:
    """Returns n x n lower-triangular {0, 1} mask with b bands of 1s.

    Args:
        n: Matrix size.
        num_bands: Number of bands (b).

    Returns:
        An integer tensor with 1s in the first b lower-triangular bands.
    """
    if num_bands < 1:
        raise ValueError(f"num_bands must be >= 1, found {num_bands}")
    ones = np.ones((n, n), dtype=np.int32)
    return np.tril(ones) - np.tril(ones, k=-num_bands)


def banded_symmetric_mask(n: int, num_bands: int) -> NDArray[np.int32]:
    """Returns n x n symmetric {0, 1} mask with 2b - 1 bands of 1s.

    Args:
        n: Matrix size.
        num_bands: Number of bands (b).

    Returns:
        An integer tensor with 1s in a symmetric band of width 2b-1.
    """
    if num_bands < 1:
        raise ValueError(f"num_bands must be >= 1, found {num_bands}")
    ones = np.ones((n, n), dtype=np.int32)
    return np.tril(ones, k=num_bands - 1) - np.tril(ones, k=-num_bands)


def get_min_sep_sensitivity_upper_bound_for_X(
    X: ArrayLike,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Computes an upper bound on the min_sep sensitivity of X.

    Unlike get_sensitivity_banded_for_X, this does not require X to be banded.

    Reference: Algorithm 4 from https://arxiv.org/abs/2306.08153

    Args:
        X: The Gram matrix C.T @ C.
        min_sep: Minimum separation between participations.
        max_participations: Maximum number of participations.

    Returns:
        An upper bound on the L2 sensitivity.
    """
    X = np.asarray(X, dtype=np.float64)
    checks.check(X=X)
    # Stage 1: For each row, find max participation value
    row_max = np.zeros(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        row_max[i] = max_participation_for_linear_fn(
            np.abs(X[i]),
            min_sep=min_sep,
            max_participations=max_participations,
        )
    # Stage 2: Find max participation over these row maxima
    result = max_participation_for_linear_fn(row_max, min_sep, max_participations)
    return float(np.sqrt(result))


def get_min_sep_sensitivity_upper_bound(
    C: ArrayLike,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Like get_min_sep_sensitivity_upper_bound_for_X, but takes encoder C."""
    C = np.asarray(C, dtype=np.float64)
    checks.check(C=C)
    return get_min_sep_sensitivity_upper_bound_for_X(
        C.T @ C, min_sep, max_participations
    )


def get_sensitivity_banded_for_X(
    X: ArrayLike,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Computes the exact sensitivity of X when X is min_sep-banded.

    Args:
        X: The Gram matrix C.T @ C, which must be min_sep-banded.
        min_sep: Minimum separation between participations.
        max_participations: Maximum number of participations.

    Returns:
        The L2 sensitivity.

    Raises:
        ValueError: If X is not properly banded.
    """
    X = np.asarray(X, dtype=np.float64)
    checks.check(X=X)
    n = X.shape[0]
    if min_sep < 1 or min_sep > n:
        raise ValueError(f"min_sep must be in the range [1, {n}], found {min_sep}.")

    expected_zeros = ~banded_symmetric_mask(n, min_sep).astype(bool)
    if not np.all(X[expected_zeros] == 0):
        raise ValueError(
            "X must be min_sep-banded: entries with |i - j| >= min_sep must be zero."
        )

    x = np.diag(X)
    value = max_participation_for_linear_fn(x, min_sep, max_participations)
    return float(np.sqrt(value))


def get_sensitivity_banded(
    C: ArrayLike,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Like get_sensitivity_banded_for_X, but takes encoder C."""
    C = np.asarray(C, dtype=np.float64)
    checks.check(C=C)
    return get_sensitivity_banded_for_X(C.T @ C, min_sep, max_participations)
