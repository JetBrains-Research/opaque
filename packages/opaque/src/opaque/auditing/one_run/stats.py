"""Statistical helpers for one-run privacy auditing.

Likelihood-ratio p-value computation and epsilon binary search
for the one-run estimator (Steinke et al. 2023).
"""

from __future__ import annotations

import numpy as np
import scipy.special
import scipy.stats

__all__ = ["epsilon_one_run_search", "log_sub", "one_run_p_value"]


def log_sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stable computation of log(exp(x) - exp(y))."""
    if np.any(y > x):
        raise ValueError(f"y must be <= x, got y={y} and x={x}")
    with np.errstate(divide="ignore"):
        return x + np.log1p(-np.exp(y - x))


def one_run_p_value(
    m: int, n_guess: int, n_correct: int, eps: float, delta: float
) -> float:
    """P-value for one-shot privacy audit (Steinke et al. 2023)."""
    q = scipy.special.expit(eps)
    beta = scipy.stats.binom.sf(n_correct - 1, n_guess, q)

    if delta == 0:
        return beta

    i_vals = np.arange(1, n_correct + 1)
    cum_sums = scipy.stats.binom.sf(n_correct - i_vals - 1, n_guess, q) - beta
    alpha = np.max(cum_sums / i_vals, initial=0)

    return min(beta + alpha * delta * 2 * m, 1.0)


def validate_significance(significance: float) -> None:
    if not 0 < significance < 0.5:
        raise ValueError(f"significance must be in (0, 0.5), got {significance}")


def validate_delta(delta: float) -> None:
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got {delta}")


def epsilon_one_run_search(
    n_guess: int,
    n_correct: int,
    m: int,
    significance: float,
    delta: float,
    eps_max: float,
    tol: float,
) -> float:
    """One-run epsilon via binary search."""
    if n_guess == 0 or n_correct == 0:
        return 0.0

    eps_lo, eps_hi = 0.0, eps_max
    while eps_hi - eps_lo > tol:
        eps_mid = (eps_lo + eps_hi) / 2
        p_val = one_run_p_value(m, n_guess, n_correct, eps_mid, delta)
        if p_val < significance:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid

    return eps_lo
