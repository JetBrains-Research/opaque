"""ROC curve helpers for one-run privacy auditing.

Raw empirical ROC (TN/FN counts at every threshold) with an optional
Pareto-frontier (hull) restriction, plus TPR/FPR interpolation, used by the
one-run estimator.
"""

from __future__ import annotations

import numpy as np

__all__ = ["get_tn_fn_counts", "pareto_frontier", "tpr_at_given_fpr"]


def pareto_frontier(points: np.ndarray) -> np.ndarray:
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


def get_tn_fn_counts(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    *,
    hull: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute TN/FN counts at each threshold along the empirical ROC.

    Returns the **raw** empirical ROC by default.  ``hull=True`` restricts the
    points to the Pareto frontier (upper-left convex hull); that is useful for
    optimal-threshold audit statistics but a biased basis for AUC / coverage
    (the hull's area sits systematically above 0.5 under the null), so raw is
    the default (#378).
    """
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

    # The terminal (reject-all) threshold must count EVERY score, including any
    # ``+inf`` that is not strictly ``< inf`` under ``side="left"``.  Pin it to
    # the totals so TPR/FPR/AUC denominators are the true ``n_in`` / ``n_out``
    # rather than the finite-only counts (#378).
    fn_counts[-1] = in_sorted.size
    tn_counts[-1] = out_sorted.size

    if hull:
        indices = pareto_frontier(np.stack([fn_counts, tn_counts], axis=1))
        return thresholds[indices], tn_counts[indices], fn_counts[indices]
    return thresholds, tn_counts, fn_counts


def tpr_at_given_fpr(
    fpr: np.ndarray | float,
    tp_counts: np.ndarray,
    fp_counts: np.ndarray,
) -> np.ndarray | float:
    """TPR at a given FPR along the empirical ROC (linear interpolation)."""
    fpr_arr = np.asarray(fpr)
    if not np.all((fpr_arr >= 0) & (fpr_arr <= 1)):
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
