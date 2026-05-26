"""Statistical helpers shared across one-run audit methods.

Includes argument validation, the closed-form ε ceiling used as the
binary-search upper bracket, and the legacy Steinke likelihood-ratio
search (preserved for the private ``_epsilon_at_steinke`` regression path).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special
import scipy.stats

__all__ = [
    "epsilon_one_run_search",
    "one_run_p_value",
    "search_ceiling",
    "validate_delta",
    "validate_significance",
]


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


def search_ceiling(m: int, delta: float, significance: float) -> float:
    """Maximum ε any one-run audit at ``(m, delta, significance)`` can return.

    Derived from the perfect-attack p-value
    ``sigmoid(ε) ** n_eff = significance`` where
    ``n_eff = m - round(m * delta)``.

    A safety pad of 1.1× and a floor of 1.0 keep the binary search well-posed
    when ``n_eff`` is too small to certify any positive ε.
    """
    n_eff = max(m - round(m * delta), 1)
    eps_exact = -math.log(significance ** (-1.0 / n_eff) - 1.0)
    return max(eps_exact * 1.1, 1.0)


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

    while one_run_p_value(m, n_guess, n_correct, eps_max, delta) < significance:
        eps_max *= 2

    eps_lo, eps_hi = 0.0, eps_max
    while eps_hi - eps_lo > tol:
        eps_mid = (eps_lo + eps_hi) / 2
        p_val = one_run_p_value(m, n_guess, n_correct, eps_mid, delta)
        if p_val < significance:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid

    return eps_lo
