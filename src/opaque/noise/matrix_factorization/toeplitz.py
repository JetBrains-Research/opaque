"""Library for working with lower-triangular Toeplitz matrices.

Toeplitz matrices are constant along diagonals. For DP matrix factorization,
we work with lower-triangular banded Toeplitz strategy matrices, which
enable efficient streaming noise generation with correlated noise.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - Fichtenberger et al.: https://arxiv.org/abs/2202.11205
    - Scaling BandMF: https://arxiv.org/abs/2405.15913
"""

from __future__ import annotations

import functools
from typing import Protocol

import torch
from scipy.linalg import toeplitz as scipy_toeplitz

from . import checks, optimization, sensitivity, streaming_matrix


def _l2_norm_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.dot(x, x)


def _reconcile(coef: torch.Tensor, n: int | None = None) -> tuple[torch.Tensor, int]:
    """Reconcile Toeplitz coefficients with matrix size."""
    n = n or len(coef)
    coef = torch.as_tensor(coef, dtype=torch.float64)[:n]
    return coef, n


def pad_coefs_to_n(coef: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Materialize length-n Toeplitz coefficients (zero-padded)."""
    coef, n = _reconcile(coef, n)
    result = torch.zeros(n, dtype=coef.dtype)
    result[: len(coef)] = coef
    return result


def inverse_as_streaming_matrix(
    coef: torch.Tensor,
    column_normalize_for_n: int | None = None,
) -> streaming_matrix.StreamingMatrix:
    """Create C^{-1} as a StreamingMatrix.

    If ``column_normalize_for_n`` is None, returns C^{-1} for an arbitrarily
    large banded Toeplitz C. Otherwise, C is column-normalized for size n.

    This implements Algorithm 9 from https://arxiv.org/abs/2306.08153.

    Args:
        coef: Toeplitz coefficients of the strategy matrix.
        column_normalize_for_n: If set, column-normalize C for this size.

    Returns:
        A StreamingMatrix representing C^{-1}.
    """
    coef, _ = _reconcile(coef, column_normalize_for_n)
    bands = coef.shape[0]

    def init(abstract_yi):
        dtype = torch.promote_types(abstract_yi.dtype, coef.dtype)
        zero = torch.zeros_like(abstract_yi, dtype=dtype)
        return zero.unsqueeze(0).expand(bands - 1, *zero.shape).clone()

    def _next(yi, state):
        if bands == 1:
            return yi / coef[0], state
        inner = torch.tensordot(coef[1:], state, dims=1)
        xi = (yi - inner) / coef[0]
        new_state = torch.roll(state, 1, dims=0)
        new_state[0] = xi
        return xi, new_state

    Cinv = streaming_matrix.StreamingMatrix.from_array_implementation(init, _next)

    if column_normalize_for_n is not None:
        full_coef = pad_coefs_to_n(coef, column_normalize_for_n)
        col_norms = torch.sqrt(torch.cumsum(full_coef**2, dim=0)).flip(0)
        Cinv = streaming_matrix.scale_rows_and_columns(Cinv, row_scale=col_norms)

    return Cinv


def optimal_max_error_strategy_coefs(n: int) -> torch.Tensor:
    """Returns optimal Toeplitz strategy coefficients for max error.

    From Fichtenberger, Henzinger, and Upadhyay:
    https://arxiv.org/abs/2202.11205

    Args:
        n: Number of coefficients.

    Returns:
        Tensor of Toeplitz coefficients.
    """
    k = torch.arange(n, dtype=torch.float64)
    ratios = (2 * k - 1) / (2 * k)
    ratios[0] = 1.0
    return torch.cumprod(ratios, dim=0)


def optimal_max_error_noising_coefs(n: int) -> torch.Tensor:
    """Returns optimal Toeplitz noising coefficients for max error.

    Args:
        n: Number of coefficients.

    Returns:
        Coefficients of C^{-1}.
    """
    c = optimal_max_error_strategy_coefs(n)
    result = c.clone()
    result[1:n] = c[1:n] - c[: n - 1]
    return result


def materialize_lower_triangular(
    coef: torch.Tensor, n: int | None = None
) -> torch.Tensor:
    """Create a lower-triangular Toeplitz matrix from coefficients.

    Example: coef=[a,b,c], n=4 gives::

        [[a 0 0 0]
         [b a 0 0]
         [c b a 0]
         [0 c b a]]

    Args:
        coef: Nonzero coefficients of the first column.
        n: Optional matrix size (defaults to len(coef)).

    Returns:
        Lower-triangular Toeplitz matrix.
    """
    full_coef = pad_coefs_to_n(coef, n)
    n_actual = len(full_coef)
    original_device = full_coef.device
    col = full_coef.detach().cpu().numpy()
    row = torch.zeros(n_actual, dtype=full_coef.dtype).numpy()
    row[0] = col[0]
    toeplitz_np = scipy_toeplitz(col, row)
    return torch.from_numpy(toeplitz_np).to(
        device=original_device, dtype=full_coef.dtype
    )


def solve_banded(coef: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve T_{coef} x = rhs for banded Toeplitz T.

    Args:
        coef: Toeplitz coefficients.
        rhs: Right-hand side vector.

    Returns:
        Solution x.
    """
    return (
        streaming_matrix.multiply_array(
            inverse_as_streaming_matrix(coef),
            rhs.unsqueeze(1) if rhs.ndim == 1 else rhs,
        ).squeeze(-1)
        if rhs.ndim == 1
        else streaming_matrix.multiply_array(inverse_as_streaming_matrix(coef), rhs)
    )


def multiply(
    lhs_coef: torch.Tensor,
    rhs_coef: torch.Tensor,
    n: int | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Multiply two lower-triangular Toeplitz matrices (via convolution).

    Args:
        lhs_coef: Coefficients of the left matrix.
        rhs_coef: Coefficients of the right matrix.
        n: Optional matrix size.
        skip_checks: Skip input validation.

    Returns:
        Coefficients of the product matrix.
    """
    if not skip_checks:
        if n is None and len(lhs_coef) != len(rhs_coef):
            raise ValueError(
                "If n is not specified, lhs_coef and rhs_coef must have "
                f"the same length, found {len(lhs_coef)} and {len(rhs_coef)}."
            )
    lhs_coef, n = _reconcile(lhs_coef, n)
    rhs_coef, _ = _reconcile(rhs_coef, n)

    # Convolution of Toeplitz coefficients
    # Use numpy for convolution since torch doesn't have a direct 1D convolve
    import numpy as np

    conv = np.convolve(
        lhs_coef.detach().cpu().numpy(), rhs_coef.detach().cpu().numpy()
    )[:n]
    result = torch.as_tensor(conv, dtype=lhs_coef.dtype)
    return result.to(lhs_coef.device)


def inverse_coef(coef: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Find the inverse coefficients of a Toeplitz matrix.

    Args:
        coef: Toeplitz coefficients of C.
        n: Optional matrix size.

    Returns:
        Toeplitz coefficients of C^{-1}.
    """
    coef, n = _reconcile(coef, n)
    e0 = torch.zeros(n, dtype=coef.dtype)
    e0[0] = 1.0
    return solve_banded(coef, e0)


def sensitivity_squared(coef: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Sensitivity^2 under single participation."""
    coef, _ = _reconcile(coef, n)
    return _l2_norm_squared(coef)


def minsep_sensitivity_squared(
    strategy_coef: torch.Tensor,
    min_sep: int,
    max_participations: int | None = None,
    n: int | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Returns the sensitivity squared of the Toeplitz matrix under min-sep.

    Reference: https://arxiv.org/abs/2405.13763, Theorem 2.

    Args:
        strategy_coef: Toeplitz coefficients of C.
        min_sep: Minimum separation between participations.
        max_participations: Maximum participations.
        n: Optional matrix size.
        skip_checks: Skip input checks.

    Returns:
        The sensitivity squared.
    """
    coef, n = _reconcile(strategy_coef, n)

    if not skip_checks:
        if not torch.all(coef >= 0):
            raise ValueError(
                f"coef must be non-negative, found min={coef.min().item()}"
            )
        if len(coef) > 1:
            incr = coef[1:] - coef[:-1]
            max_incr = incr.max()
            if max_incr > 0:
                raise ValueError(
                    f"coef must be non-increasing, found increase "
                    f"{max_incr.item()} at index {incr.argmax().item()}"
                )
        if min_sep <= 0:
            raise ValueError("min_sep must be positive")

    k = sensitivity.minsep_true_max_participations(
        n=n, min_sep=min_sep, max_participations=max_participations
    )

    padding = (min_sep - n) % min_sep
    full_coef = pad_coefs_to_n(coef, n + padding)
    vector = full_coef.reshape(-1, min_sep).cumsum(dim=0).flatten()
    if min_sep * k < len(vector):
        vector[min_sep * k :] = (
            vector[min_sep * k :] - vector[: len(vector) - min_sep * k]
        )
    return torch.dot(vector[:n], vector[:n])


def per_query_error(
    *,
    strategy_coef: torch.Tensor | None = None,
    noising_coef: torch.Tensor | None = None,
    n: int | None = None,
    workload_coef: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Expected per-query squared error for a (banded) Toeplitz mechanism.

    Exactly one of ``strategy_coef`` and ``noising_coef`` must be provided.

    Args:
        strategy_coef: Toeplitz coefficients of the strategy matrix.
        noising_coef: Toeplitz coefficients of the noising matrix.
        n: Matrix size.
        workload_coef: Workload matrix coefficients (defaults to all ones).
        skip_checks: Skip input validation.

    Returns:
        Per-query expected squared error, tensor of length n.
    """
    if not skip_checks:
        checks.check_exactly_one(strategy_coef=strategy_coef, noising_coef=noising_coef)

    if strategy_coef is not None:
        strategy_coef, n = _reconcile(strategy_coef, n)
        if workload_coef is not None:
            workload_coef = pad_coefs_to_n(workload_coef, n)
        else:
            workload_coef = torch.ones(n, dtype=strategy_coef.dtype)
        B_coef = solve_banded(strategy_coef, workload_coef)
    else:
        assert noising_coef is not None
        noising_coef, n = _reconcile(noising_coef, n)
        if workload_coef is None:
            B_coef = torch.cumsum(noising_coef, dim=0)
        else:
            B_coef = multiply(workload_coef, noising_coef, n=n, skip_checks=skip_checks)

    return torch.cumsum(B_coef**2, dim=0)


def max_error(
    *,
    strategy_coef: torch.Tensor | None = None,
    noising_coef: torch.Tensor | None = None,
    n: int | None = None,
    workload_coef: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Max-over-iterations squared error for a Toeplitz mechanism."""
    return per_query_error(
        strategy_coef=strategy_coef,
        noising_coef=noising_coef,
        n=n,
        workload_coef=workload_coef,
        skip_checks=skip_checks,
    )[-1]


def mean_error(
    *,
    strategy_coef: torch.Tensor | None = None,
    noising_coef: torch.Tensor | None = None,
    n: int | None = None,
    workload_coef: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Mean-over-iterations squared error for a Toeplitz mechanism."""
    return per_query_error(
        strategy_coef=strategy_coef,
        noising_coef=noising_coef,
        n=n,
        workload_coef=workload_coef,
        skip_checks=skip_checks,
    ).mean()


class ErrorOrLossFn(Protocol):
    """Protocol for error functions."""

    def __call__(
        self, *, strategy_coef: torch.Tensor, n: int | None = None
    ) -> torch.Tensor: ...


def loss(
    strategy_coef: torch.Tensor,
    n: int | None = None,
    error_fn: ErrorOrLossFn = mean_error,
) -> torch.Tensor:
    """Error of C on prefix workload under single participation.

    Returns error * sensitivity_squared.

    Args:
        strategy_coef: Toeplitz coefficients of C.
        n: Matrix size.
        error_fn: Error function (mean_error or max_error).

    Returns:
        Total squared error times sensitivity.
    """
    strategy_coef, n = _reconcile(strategy_coef, n)
    error = error_fn(strategy_coef=strategy_coef, n=n)
    sens_sq = sensitivity_squared(strategy_coef, n)
    return error * sens_sq


mean_loss = functools.partial(loss, error_fn=mean_error)
max_loss = functools.partial(loss, error_fn=max_error)


def optimize_banded_toeplitz(
    n: int,
    bands: int,
    strategy_coef: torch.Tensor | None = None,
    max_optimizer_steps: int = 250,
    loss_fn=mean_loss,
) -> torch.Tensor:
    """Optimize banded Toeplitz strategy on a Prefix workload.

    The resulting strategy can be used for single- and multi-participation
    settings, as long as the minimum separation >= number of bands.

    Args:
        n: Number of iterations.
        bands: Number of bands in the Toeplitz matrix.
        strategy_coef: Optional initial coefficients.
        max_optimizer_steps: Maximum L-BFGS iterations.
        loss_fn: Loss function (default: mean_loss).

    Returns:
        Optimized coefficients with L2 norm 1.
    """
    partial_loss = functools.partial(loss_fn, n=n)

    if strategy_coef is None:
        strategy_coef = optimal_max_error_strategy_coefs(bands)
    if strategy_coef.shape[0] != bands:
        raise ValueError(f"{strategy_coef.shape=} != {bands=}")

    params = optimization.optimize(
        partial_loss,
        strategy_coef,
        max_optimizer_steps=max_optimizer_steps,
    )
    return params / torch.linalg.norm(params)
