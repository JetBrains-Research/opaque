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

import dataclasses
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar

import numpy as np
from scipy.linalg import toeplitz as scipy_toeplitz
from scipy.optimize import minimize

from opaque.api.engine import ops

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

from . import (
    _checks as checks,
)
from . import (
    _sensitivity as sensitivity,
)
from . import (
    _streaming_matrix as streaming_matrix,
)

# ---------------------------------------------------------------------------
# Optimization (L-BFGS wrapper, formerly optimization.py)
# ---------------------------------------------------------------------------

_ParamT = TypeVar("_ParamT")
_CallbackFnType: TypeAlias = Callable[["_OptimCallbackArgs"], bool | None]


@dataclasses.dataclass
class _OptimCallbackArgs:
    """Information passed to the callback on each optimization step."""

    step: int
    loss: float
    grad: Any | None
    params: Any
    state: Any


class _EarlyStopException(Exception):
    """Internal exception for early stopping in scipy optimization."""

    def __init__(self, params):
        self.params = params
        super().__init__("Early stop")


def _refine_stalled_analytic_optimization(
    loss_fn: Callable,
    initial_params: NDArray[np.float64],
    optimized_params: NDArray[np.float64],
    bounds: list[tuple[float | None, float | None]] | None,
    max_optimizer_steps: int,
    *,
    force: bool = False,
) -> NDArray[np.float64]:
    """Recover a feasible descent step when L-BFGS-B stops at its initializer.

    Rational BLT parameterizations have narrow feasible regions.  L-BFGS-B
    occasionally tests only infeasible trial points and incorrectly reports
    relative-reduction convergence at the initial point.  A normalized,
    bounded Armijo step is scale independent and supplies a conservative
    fallback for analytic objectives only.
    """
    initial_value, _ = loss_fn(initial_params)
    optimized_value, _ = loss_fn(optimized_params)
    if not force and optimized_value < initial_value * (1.0 - 1e-12):
        return optimized_params

    if bounds is None:
        lower = np.full_like(initial_params, -np.inf)
        upper = np.full_like(initial_params, np.inf)
    else:
        lower = np.asarray(
            [-np.inf if lower is None else lower for lower, _ in bounds],
            dtype=np.float64,
        )
        upper = np.asarray(
            [np.inf if upper is None else upper for _, upper in bounds],
            dtype=np.float64,
        )

    params = optimized_params.copy()
    for _ in range(min(max_optimizer_steps, 100)):
        value, gradient = loss_fn(params)
        gradient = np.asarray(gradient, dtype=np.float64)
        gradient_norm = float(np.linalg.norm(gradient))
        if (
            not np.isfinite(value)
            or not np.isfinite(gradient_norm)
            or gradient_norm == 0
        ):
            break
        step = min(1e-2, 1.0 / gradient_norm)
        direction = gradient / gradient_norm
        accepted = False
        for _ in range(40):
            candidate = np.clip(params - step * direction, lower, upper)
            candidate_value, _ = loss_fn(candidate)
            if (
                np.isfinite(candidate_value)
                and candidate_value < value - 1e-4 * step * gradient_norm
            ):
                params = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return params


def _lbfgs_optimize(
    loss_fn: Callable,
    params: ArrayLike,
    *,
    max_optimizer_steps: int = 250,
    grad: bool = False,
    callback: _CallbackFnType = lambda _: None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    restart_stalled_analytic_optimization: bool = False,
) -> NDArray[np.float64]:
    """Optimize a scalar loss function using L-BFGS-B.

    When ``grad`` is true, ``loss_fn`` returns ``(value, gradient)`` and the
    analytic gradient is passed through to SciPy. Otherwise SciPy numerically
    differentiates the scalar objective. Parameters are canonical NumPy
    float64 arrays throughout the host-side optimization.

    ``restart_stalled_analytic_optimization`` is an opt-in recovery for
    rational objectives with narrow feasible regions.  It takes one bounded
    Armijo step before restarting L-BFGS-B when the initial run stalls.
    """
    params_np = np.asarray(params, dtype=np.float64).copy()

    step_counter = [0]

    def scipy_loss(x):
        x = np.asarray(x, dtype=np.float64)
        result = loss_fn(x)
        if grad:
            loss_value, gradient = result
            loss_np = float(loss_value)
            gradient_np = np.asarray(gradient, dtype=np.float64)
            if gradient_np.shape != x.shape:
                raise ValueError(
                    "Analytic gradient must have the same shape as parameters: "
                    f"{gradient_np.shape} != {x.shape}"
                )
        else:
            loss_np = float(result)
            gradient_np = None

        cb_result = callback(
            _OptimCallbackArgs(
                step=step_counter[0],
                loss=loss_np,
                grad=gradient_np,
                params=x.copy(),
                state=None,
            )
        )
        step_counter[0] += 1

        if cb_result:
            raise _EarlyStopException(x)

        return (loss_np, gradient_np) if grad else loss_np

    try:
        result = minimize(
            scipy_loss,
            params_np,
            method="L-BFGS-B",
            jac=grad,
            bounds=bounds,
            options={"maxiter": max_optimizer_steps, "ftol": 1e-15, "gtol": 1e-10},
        )
        optimal_params = np.asarray(result.x, dtype=np.float64)
        if grad and restart_stalled_analytic_optimization:
            recovered_params = _refine_stalled_analytic_optimization(
                loss_fn,
                params_np,
                optimal_params,
                bounds,
                1,
            )
            recovered_value, _ = loss_fn(recovered_params)
            optimal_value, _ = loss_fn(optimal_params)
            if recovered_value < optimal_value * (1.0 - 1e-12):
                restarted = minimize(
                    scipy_loss,
                    recovered_params,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={
                        "maxiter": max_optimizer_steps,
                        "ftol": 1e-15,
                        "gtol": 1e-10,
                    },
                )
                restarted_params = np.asarray(restarted.x, dtype=np.float64)
                restarted_value, _ = loss_fn(restarted_params)
                if restarted_value < recovered_value * (1.0 - 1e-12):
                    optimal_params = restarted_params
                else:
                    optimal_params = _refine_stalled_analytic_optimization(
                        loss_fn,
                        recovered_params,
                        recovered_params,
                        bounds,
                        max_optimizer_steps - 1,
                        force=True,
                    )
            else:
                optimal_params = recovered_params
        elif grad:
            optimal_params = _refine_stalled_analytic_optimization(
                loss_fn,
                params_np,
                optimal_params,
                bounds,
                max_optimizer_steps,
            )
    except _EarlyStopException as e:
        optimal_params = np.asarray(e.params, dtype=np.float64)

    return optimal_params


# ---------------------------------------------------------------------------
# Toeplitz matrix library
# ---------------------------------------------------------------------------

__all__ = [
    "inverse_as_streaming_matrix",
    "materialize_lower_triangular",
    "max_error",
    "mean_error",
    "minsep_sensitivity_squared",
    "optimal_max_error_strategy_coefs",
    "optimize",
    "per_query_error",
    "sensitivity_squared",
]


def _l2_norm_squared(x: NDArray[np.float64]) -> np.float64:
    return np.dot(x, x)


def _reconcile(
    coef: ArrayLike, n: int | None = None
) -> tuple[NDArray[np.float64], int]:
    """Reconcile Toeplitz coefficients with matrix size."""
    n = n or len(coef)
    coef = np.asarray(coef, dtype=np.float64)[:n]
    return coef, n


def pad_coefs_to_n(coef: ArrayLike, n: int | None = None) -> NDArray[np.float64]:
    """Materialize length-n Toeplitz coefficients (zero-padded)."""
    coef, n = _reconcile(coef, n)
    result = np.zeros(n, dtype=np.float64)
    result[: len(coef)] = coef
    return result


def inverse_as_streaming_matrix(
    coef: ArrayLike,
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
        return tuple(
            streaming_matrix._zeros_like(abstract_yi) for _ in range(bands - 1)
        )

    def _next(yi, state):
        if bands == 1:
            divisor = streaming_matrix._scalar_like(coef[0], yi)
            if isinstance(yi, np.ndarray):
                return yi / divisor, state
            return ops.divide(yi, divisor), state

        inner = streaming_matrix._zeros_like(yi)
        for coefficient, previous_xi in zip(coef[1:], state, strict=True):
            scaled = streaming_matrix._multiply(
                streaming_matrix._scalar_like(coefficient, previous_xi),
                previous_xi,
            )
            inner = streaming_matrix._add(inner, scaled)

        divisor = streaming_matrix._scalar_like(coef[0], yi)
        if isinstance(yi, np.ndarray):
            xi = (yi - inner) / divisor
        else:
            xi = ops.divide(ops.subtract(yi, inner), divisor)
        return xi, (xi, *state[:-1])

    Cinv = streaming_matrix.StreamingMatrix.from_array_implementation(init, _next)

    if column_normalize_for_n is not None:
        full_coef = pad_coefs_to_n(coef, column_normalize_for_n)
        col_norms = np.sqrt(np.cumsum(full_coef**2))[::-1].copy()
        Cinv = streaming_matrix.scale_rows_and_columns(Cinv, row_scale=col_norms)

    return Cinv


def optimal_max_error_strategy_coefs(n: int) -> NDArray[np.float64]:
    """Returns optimal Toeplitz strategy coefficients for max error.

    From Fichtenberger, Henzinger, and Upadhyay:
    https://arxiv.org/abs/2202.11205

    Args:
        n: Number of coefficients.

    Returns:
        Tensor of Toeplitz coefficients.
    """
    k = np.arange(n, dtype=np.float64)
    ratios = np.ones(n, dtype=np.float64)
    ratios[1:] = (2 * k[1:] - 1) / (2 * k[1:])
    return np.cumprod(ratios)


def optimal_max_error_noising_coefs(n: int) -> NDArray[np.float64]:
    """Returns optimal Toeplitz noising coefficients for max error.

    Args:
        n: Number of coefficients.

    Returns:
        Coefficients of C^{-1}.
    """
    c = optimal_max_error_strategy_coefs(n)
    result = c.copy()
    result[1:n] = c[1:n] - c[: n - 1]
    return result


def materialize_lower_triangular(
    coef: ArrayLike, n: int | None = None
) -> NDArray[np.float64]:
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
    col = full_coef
    row = np.zeros(n_actual, dtype=np.float64)
    row[0] = col[0]
    toeplitz_np = scipy_toeplitz(col, row)
    return np.asarray(toeplitz_np, dtype=np.float64)


def solve_banded(coef: ArrayLike, rhs: ArrayLike) -> NDArray[np.float64]:
    """Solve T_{coef} x = rhs for banded Toeplitz T.

    Args:
        coef: Toeplitz coefficients.
        rhs: Right-hand side vector.

    Returns:
        Solution x.
    """
    coef = np.asarray(coef, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    result = np.empty_like(rhs, dtype=np.float64)
    for i in range(len(rhs)):
        width = min(i, len(coef) - 1)
        inner = np.tensordot(coef[1 : width + 1], result[i - width : i][::-1], axes=1)
        result[i] = (rhs[i] - inner) / coef[0]
    return result


def multiply(
    lhs_coef: ArrayLike,
    rhs_coef: ArrayLike,
    n: int | None = None,
    skip_checks: bool = False,
) -> NDArray[np.float64]:
    """Multiply two lower-triangular Toeplitz matrices (via convolution).

    Args:
        lhs_coef: Coefficients of the left matrix.
        rhs_coef: Coefficients of the right matrix.
        n: Optional matrix size.
        skip_checks: Skip input validation.

    Returns:
        Coefficients of the product matrix.
    """
    if not skip_checks and n is None and len(lhs_coef) != len(rhs_coef):
        raise ValueError(
            "If n is not specified, lhs_coef and rhs_coef must have "
            f"the same length, found {len(lhs_coef)} and {len(rhs_coef)}."
        )
    lhs_coef, n = _reconcile(lhs_coef, n)
    rhs_coef, _ = _reconcile(rhs_coef, n)

    return np.convolve(lhs_coef, rhs_coef)[:n]


def inverse_coef(coef: ArrayLike, n: int | None = None) -> NDArray[np.float64]:
    """Find the inverse coefficients of a Toeplitz matrix.

    Args:
        coef: Toeplitz coefficients of C.
        n: Optional matrix size.

    Returns:
        Toeplitz coefficients of C^{-1}.
    """
    coef, n = _reconcile(coef, n)
    e0 = np.zeros(n, dtype=np.float64)
    e0[0] = 1.0
    return solve_banded(coef, e0)


def sensitivity_squared(coef: ArrayLike, n: int | None = None) -> np.float64:
    """Sensitivity^2 under single participation."""
    coef, _ = _reconcile(coef, n)
    return _l2_norm_squared(coef)


def minsep_sensitivity_squared(
    strategy_coef: ArrayLike,
    min_sep: int,
    max_participations: int | None = None,
    n: int | None = None,
    skip_checks: bool = False,
) -> np.float64:
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
        if not np.all(coef >= 0):
            raise ValueError(f"coef must be non-negative, found min={coef.min()}")
        if len(coef) > 1:
            incr = coef[1:] - coef[:-1]
            max_incr = incr.max()
            if max_incr > 0:
                raise ValueError(
                    f"coef must be non-increasing, found increase "
                    f"{max_incr} at index {incr.argmax()}"
                )
        if min_sep <= 0:
            raise ValueError("min_sep must be positive")

    k = sensitivity.minsep_true_max_participations(
        n=n, min_sep=min_sep, max_participations=max_participations
    )

    padding = (min_sep - n) % min_sep
    full_coef = pad_coefs_to_n(coef, n + padding)
    vector = full_coef.reshape(-1, min_sep).cumsum(axis=0).flatten()
    if min_sep * k < len(vector):
        vector[min_sep * k :] = (
            vector[min_sep * k :] - vector[: len(vector) - min_sep * k]
        )
    return np.dot(vector[:n], vector[:n])


def per_query_error(
    *,
    strategy_coef: ArrayLike | None = None,
    noising_coef: ArrayLike | None = None,
    n: int | None = None,
    workload_coef: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
    skip_checks: bool = False,
) -> NDArray[np.float64]:
    """Expected per-query squared error for a (banded) Toeplitz mechanism.

    Exactly one of ``strategy_coef`` and ``noising_coef`` must be provided.

    Args:
        strategy_coef: Toeplitz coefficients of the strategy matrix.
        noising_coef: Toeplitz coefficients of the noising matrix.
        n: Matrix size.
        workload_coef: Workload matrix coefficients (defaults to all ones).
        query_weights: Per-query workload row weights. Each returned squared
            error is multiplied by the corresponding weight squared.
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
            workload_coef = np.ones(n, dtype=np.float64)
        B_coef = solve_banded(strategy_coef, workload_coef)
    else:
        assert noising_coef is not None
        noising_coef, n = _reconcile(noising_coef, n)
        if workload_coef is None:
            B_coef = np.cumsum(noising_coef)
        else:
            B_coef = multiply(workload_coef, noising_coef, n=n, skip_checks=skip_checks)

    error = np.cumsum(B_coef**2)
    if query_weights is None:
        return error

    query_weights = np.asarray(query_weights, dtype=np.float64)
    if query_weights.ndim != 1 or query_weights.shape[0] != n:
        raise ValueError(
            f"query_weights must have shape ({n},), got {tuple(query_weights.shape)}"
        )
    return error * np.square(query_weights)


def max_error(
    *,
    strategy_coef: ArrayLike | None = None,
    noising_coef: ArrayLike | None = None,
    n: int | None = None,
    workload_coef: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
    skip_checks: bool = False,
) -> np.float64:
    """Max-over-iterations squared error for a Toeplitz mechanism."""
    return per_query_error(
        strategy_coef=strategy_coef,
        noising_coef=noising_coef,
        n=n,
        workload_coef=workload_coef,
        query_weights=query_weights,
        skip_checks=skip_checks,
    ).max()


def mean_error(
    *,
    strategy_coef: ArrayLike | None = None,
    noising_coef: ArrayLike | None = None,
    n: int | None = None,
    workload_coef: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
    skip_checks: bool = False,
) -> np.float64:
    """Mean-over-iterations squared error for a Toeplitz mechanism."""
    return per_query_error(
        strategy_coef=strategy_coef,
        noising_coef=noising_coef,
        n=n,
        workload_coef=workload_coef,
        query_weights=query_weights,
        skip_checks=skip_checks,
    ).mean()


class ErrorOrLossFn(Protocol):
    """Protocol for error functions."""

    def __call__(
        self,
        *,
        strategy_coef: NDArray[np.float64],
        n: int | None = None,
        workload_coef: ArrayLike | None = None,
        query_weights: ArrayLike | None = None,
    ) -> float:
        raise NotImplementedError


def loss(
    strategy_coef: ArrayLike,
    n: int | None = None,
    error_fn: ErrorOrLossFn = mean_error,
    workload_coef: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
) -> float:
    """Error of C on a workload under single participation.

    Returns error * sensitivity_squared.

    Args:
        strategy_coef: Toeplitz coefficients of C.
        n: Matrix size.
        error_fn: Error function (mean_error or max_error).
        workload_coef: Toeplitz coefficients of the workload matrix.
            Defaults to ``None`` (prefix-sum workload, i.e. all ones).
        query_weights: Per-query workload row weights.

    Returns:
        Total squared error times sensitivity.
    """
    strategy_coef, n = _reconcile(strategy_coef, n)
    error_kwargs: dict[str, ArrayLike | int] = {
        "strategy_coef": strategy_coef,
        "n": n,
    }
    if workload_coef is not None:
        error_kwargs["workload_coef"] = workload_coef
    if query_weights is not None:
        error_kwargs["query_weights"] = query_weights
    error = error_fn(**error_kwargs)
    sens_sq = sensitivity_squared(strategy_coef, n)
    return error * sens_sq


mean_loss = functools.partial(loss, error_fn=mean_error)
max_loss = functools.partial(loss, error_fn=max_error)


def _mean_loss_and_gradient(
    strategy_coef: ArrayLike,
    *,
    n: int,
    workload_coef: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Return the default BandMF objective and its exact host Jacobian."""
    coef = np.asarray(strategy_coef, dtype=np.float64)
    if coef.ndim != 1 or coef.size == 0 or coef[0] == 0.0:
        return 1e100, np.zeros_like(coef)
    if workload_coef is None:
        workload = np.ones(n, dtype=np.float64)
    else:
        workload = pad_coefs_to_n(workload_coef, n)

    b_coef = np.empty(n, dtype=np.float64)
    b_jacobian = np.empty((n, len(coef)), dtype=np.float64)
    for index in range(n):
        width = min(index, len(coef) - 1)
        previous = b_coef[index - width : index][::-1]
        previous_jacobian = b_jacobian[index - width : index][::-1]
        numerator = workload[index] - np.dot(coef[1 : width + 1], previous)
        b_coef[index] = numerator / coef[0]
        numerator_jacobian = -np.sum(
            coef[1 : width + 1, np.newaxis] * previous_jacobian,
            axis=0,
        )
        numerator_jacobian[1 : width + 1] -= previous
        b_jacobian[index] = numerator_jacobian / coef[0]
        b_jacobian[index, 0] -= b_coef[index] / coef[0]

    per_query_error = np.cumsum(b_coef**2)
    per_query_error_jacobian = np.cumsum(
        2.0 * b_coef[:, np.newaxis] * b_jacobian,
        axis=0,
    )
    if query_weights is not None:
        weights_squared = np.square(np.asarray(query_weights, dtype=np.float64))
        per_query_error *= weights_squared
        per_query_error_jacobian *= weights_squared[:, np.newaxis]

    error = float(per_query_error.mean())
    error_jacobian = per_query_error_jacobian.mean(axis=0)
    sensitivity = float(np.dot(coef, coef))
    return (
        error * sensitivity,
        error_jacobian * sensitivity + error * 2.0 * coef,
    )


def optimize(
    n: int,
    bands: int,
    strategy_coef: ArrayLike | None = None,
    max_optimizer_steps: int = 250,
    loss_fn=mean_loss,
    workload_coef: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Optimize banded Toeplitz strategy for a given workload.

    The resulting strategy can be used for single- and multi-participation
    settings, as long as the minimum separation >= number of bands.

    Args:
        n: Number of iterations.
        bands: Number of bands in the Toeplitz matrix.
        strategy_coef: Optional initial coefficients.
        max_optimizer_steps: Maximum L-BFGS iterations.
        loss_fn: Loss function (default: mean_loss).
        workload_coef: Toeplitz coefficients of the workload matrix.
            Defaults to ``None`` (prefix-sum workload).  For momentum-SGD
            with coefficient β, pass ``[1, β, β², ...]``.
        query_weights: Per-query workload row weights, such as a
            learning-rate schedule materialized at each training step.

    Returns:
        Optimized coefficients with L2 norm 1.
    """
    loss_kwargs: dict[str, ArrayLike | int] = {"n": n}
    if workload_coef is not None:
        loss_kwargs["workload_coef"] = workload_coef
    if query_weights is not None:
        loss_kwargs["query_weights"] = query_weights
    partial_loss = functools.partial(loss_fn, **loss_kwargs)

    if strategy_coef is None:
        strategy_coef = optimal_max_error_strategy_coefs(bands)
    if strategy_coef.shape[0] != bands:
        raise ValueError(f"{strategy_coef.shape=} != {bands=}")

    loss_and_gradient = (
        functools.partial(
            _mean_loss_and_gradient,
            n=n,
            workload_coef=workload_coef,
            query_weights=query_weights,
        )
        if loss_fn is mean_loss
        else None
    )
    params = _lbfgs_optimize(
        loss_and_gradient if loss_and_gradient is not None else partial_loss,
        strategy_coef,
        max_optimizer_steps=max_optimizer_steps,
        grad=loss_and_gradient is not None,
    )
    return params / np.linalg.norm(params)
