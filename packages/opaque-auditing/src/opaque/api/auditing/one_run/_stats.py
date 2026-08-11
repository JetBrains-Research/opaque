"""Shared statistical helpers for one-run audit methods.

Argument validation and the closed-form ε ceiling used as the binary-search
upper bracket.
"""

from __future__ import annotations

import math


def validate_significance(significance: float) -> None:
    if not 0 < significance < 0.5:
        raise ValueError(f"significance must be in (0, 0.5), got {significance}")


def validate_delta(delta: float) -> None:
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got {delta}")


def corrected_significance(
    significance: float,
    *,
    threshold: float | None,
    num_scores: int,
) -> float:
    """Return the per-test significance for a one-run audit query.

    A supplied threshold is pre-specified, so it is one test.  The default
    threshold is selected from score labels, so Bonferroni correct over the
    at-most ``num_scores + 1`` score thresholds (including ``inf``).
    """
    validate_significance(significance)
    if threshold is not None:
        return significance
    return significance / (num_scores + 1)


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
