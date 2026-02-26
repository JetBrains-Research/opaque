"""Internal helper functions for privacy auditing.

Low-level utilities for Pareto frontiers, p-value computation,
and other mathematical operations. All functions are prefixed with ``_``
to indicate they are internal.
"""

from __future__ import annotations

import numpy as np
import scipy.special
import scipy.stats


def _log_sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stable computation of log(exp(x) - exp(y))."""
    if np.any(y > x):
        raise ValueError(f"y must be <= x, got y={y} and x={x}")
    with np.errstate(divide="ignore"):
        return x + np.log1p(-np.exp(y - x))


def _pareto_frontier(points: np.ndarray) -> np.ndarray:
    """Compute indices of Pareto frontier for a piecewise linear function."""
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError(f"Expected at least two 2D points, got shape {points.shape}")
    if not np.all(points[:-1, 0] <= points[1:, 0]):
        raise ValueError("Expected points to be sorted by x-coordinate")

    indices = np.arange(points.shape[0])
    while True:
        if len(indices) <= 2:
            break
        diff = np.diff(points[indices], axis=0)
        cross_product = diff[:-1, 1] * diff[1:, 0] - diff[1:, 1] * diff[:-1, 0]
        dominated_mask = cross_product <= 0
        if not np.any(dominated_mask):
            break
        keep_mask = np.r_[True, ~dominated_mask, True]
        indices = indices[keep_mask]
    return indices


def _get_tn_fn_counts(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute TN/FN counts at each threshold, filtered to Pareto frontier."""
    in_scores = np.asarray(in_scores)
    out_scores = np.asarray(out_scores)

    if in_scores.size == 0 and out_scores.size == 0:
        raise ValueError("At least one of the canary score arrays must be non-empty")

    unique_scores_sorted = np.union1d(in_scores, out_scores)
    thresholds = np.concatenate((unique_scores_sorted, [np.inf]))

    in_sorted = np.sort(in_scores)
    out_sorted = np.sort(out_scores)

    fn_counts = np.searchsorted(in_sorted, thresholds, side="left")
    tn_counts = np.searchsorted(out_sorted, thresholds, side="left")

    counts = np.stack([fn_counts, tn_counts], axis=1)
    indices = _pareto_frontier(counts)

    return thresholds[indices], tn_counts[indices], fn_counts[indices]


def _tpr_at_given_fpr(
    fpr: np.ndarray | float,
    tp_counts: np.ndarray,
    fp_counts: np.ndarray,
) -> np.ndarray | float:
    """Maximum TPR at a given FPR, with linear interpolation."""
    fpr_arr = np.asarray(fpr)
    if not np.all((0 <= fpr_arr) & (fpr_arr <= 1)):
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")

    n_pos = tp_counts[-1]
    n_neg = fp_counts[-1]
    target_fp_count = n_neg * fpr_arr

    threshold = np.minimum(
        np.searchsorted(fp_counts, target_fp_count, side="right"),
        np.size(fp_counts) - 1,
    )

    fp_left = fp_counts[threshold - 1]
    fp_right = fp_counts[threshold]
    q = (target_fp_count - fp_left) / (fp_right - fp_left)

    tp_left = tp_counts[threshold - 1]
    tp_right = tp_counts[threshold]
    result = (tp_left + q * (tp_right - tp_left)) / n_pos

    return float(result) if np.isscalar(fpr) else result


def _one_run_p_value(
    m: int, n_guess: int, n_correct: int, eps: float, delta: float
) -> float:
    """P-value for one-shot privacy audit (Nasr et al. 2023)."""
    q = scipy.special.expit(eps)
    beta = scipy.stats.binom.sf(n_correct - 1, n_guess, q)

    if delta == 0:
        return beta

    i_vals = np.arange(1, n_correct + 1)
    cum_sums = scipy.stats.binom.sf(n_correct - i_vals - 1, n_guess, q) - beta
    alpha = np.max(cum_sums / i_vals, initial=0)

    return min(beta + alpha * delta * 2 * m, 1.0)
