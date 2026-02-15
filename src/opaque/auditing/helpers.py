"""Helper functions for privacy auditing computations.

This module provides low-level utilities for computing binomial confidence bounds,
Pareto frontiers, and other mathematical operations needed by the auditor.
"""

from collections.abc import Sequence

import numpy as np
import scipy.special
import scipy.stats


def log_sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stable computation of log(exp(x) - exp(y)).

    Args:
        x: First value (in log space)
        y: Second value (in log space), must satisfy y <= x

    Returns:
        log(exp(x) - exp(y)) computed stably

    Raises:
        ValueError: If any y > x
    """
    if np.any(y > x):
        raise ValueError(f"y must be <= x, got y={y} and x={x}")

    # Use log1p for stability when x ≈ y
    with np.errstate(divide="ignore"):  # OK to return -inf if x == y
        return x + np.log1p(-np.exp(y - x))


def clopper_pearson_upper(
    k: int | np.ndarray, n: int, significance: float
) -> np.ndarray | float:
    """Compute Clopper-Pearson one-sided upper binomial confidence interval.

    Given k successes in n Bernoulli trials, computes a value p such that the
    probability of observing k or fewer successes with success probability p
    is approximately `significance`.

    This provides a conservative upper bound on the true success probability.

    Args:
        k: Number of successes (can be array for vectorized computation)
        n: Number of trials
        significance: Allowed probability of failure (1 - confidence)

    Returns:
        Upper confidence bound on success probability
    """
    k_arr = np.asarray(k)
    result = np.where(
        k_arr < n,
        scipy.stats.beta.ppf(1 - significance, k_arr + 1, n - k_arr),
        1.0,
    )
    # Return scalar if input was scalar
    return float(result) if np.isscalar(k) else result


def pareto_frontier(points: np.ndarray) -> np.ndarray:
    """Compute indices of Pareto frontier for a piecewise linear function.

    Given a piecewise linear function defined by points, computes the set of
    points that are not weakly linearly dominated by any pair of outer points.

    Formally: retain only points (x_i, y_i) for which there do not exist j < i
    and k > i and a in [0, 1] such that:
        x_i = (1-a)*x_j + a*x_k  and  y_i <= (1-a)*y_j + a*y_k

    Uses iterative vectorized operations. Complexity is typically O(N) but can be
    O(N^2) in pathological cases.

    Args:
        points: Array of shape (N, 2) with N >= 2 containing vertices defining
            the piecewise linear function, sorted by x-coordinate.

    Returns:
        Array of indices (length M with 2 <= M <= N) of points on the Pareto
        frontier.

    Raises:
        ValueError: If points has wrong shape or is not sorted by x-coordinate.
    """
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError(
            f"Expected at least two 2D points, got shape {points.shape}"
        )

    if not np.all(points[:-1, 0] <= points[1:, 0]):
        raise ValueError("Expected points to be sorted by x-coordinate")

    indices = np.arange(points.shape[0])

    while True:
        if len(indices) <= 2:
            break

        # Compute cross products to find dominated points
        diff = np.diff(points[indices], axis=0)
        cross_product = diff[:-1, 1] * diff[1:, 0] - diff[1:, 1] * diff[:-1, 0]
        dominated_mask = cross_product <= 0

        # If no dominated points in this pass, we're done
        if not np.any(dominated_mask):
            break

        # Keep first, last, and non-dominated interior points
        keep_mask = np.r_[True, ~dominated_mask, True]
        indices = indices[keep_mask]

    return indices


def get_tn_fn_counts(
    in_canary_scores: Sequence[float],
    out_canary_scores: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute true negative and false negative counts at each threshold.

    For each possible threshold (unique score value), computes:
    - True negatives: held-out canaries with score < threshold (correctly rejected)
    - False negatives: held-in canaries with score < threshold (missed detections)

    Only returns points on the Pareto frontier to reduce redundancy.

    Args:
        in_canary_scores: Attack scores of held-in canaries (higher is more suspicious)
        out_canary_scores: Attack scores of held-out canaries (should be lower)

    Returns:
        Tuple of (thresholds, tn_counts, fn_counts), each a 1D array. Only includes
        Pareto-optimal points.

    Raises:
        ValueError: If both score arrays are empty.
    """
    in_scores = np.asarray(in_canary_scores)
    out_scores = np.asarray(out_canary_scores)

    if in_scores.size == 0 and out_scores.size == 0:
        raise ValueError(
            "At least one of the canary score arrays must be non-empty"
        )

    # Get unique sorted thresholds from both arrays
    unique_scores_sorted = np.union1d(in_scores, out_scores)

    # Append infinity to consider thresholds > max_score
    thresholds = np.concatenate((unique_scores_sorted, [np.inf]))

    # Sort for efficient searching
    in_sorted = np.sort(in_scores)
    out_sorted = np.sort(out_scores)

    # Count scores below each threshold using binary search
    # side='left' gives index of first value >= threshold
    fn_counts = np.searchsorted(in_sorted, thresholds, side="left")
    tn_counts = np.searchsorted(out_sorted, thresholds, side="left")

    # Keep only Pareto-optimal points
    counts = np.stack([fn_counts, tn_counts], axis=1)
    indices = pareto_frontier(counts)

    return thresholds[indices], tn_counts[indices], fn_counts[indices]


def tpr_at_given_fpr(
    fpr: np.ndarray | float,
    tp_counts: np.ndarray,
    fp_counts: np.ndarray,
) -> np.ndarray | float:
    """Compute maximum TPR achievable at a given FPR.

    Given true positive and false positive counts at each threshold, computes
    the maximum true positive rate (TPR) achievable while maintaining false
    positive rate (FPR) at most the specified value.

    Uses linear interpolation between thresholds.

    Args:
        fpr: Desired false positive rate(s) in [0, 1]
        tp_counts: True positive counts at each threshold (non-decreasing)
        fp_counts: False positive counts at each threshold (non-decreasing)

    Returns:
        Maximum TPR(s) at the given FPR(s)

    Raises:
        ValueError: If fpr is outside [0, 1]
    """
    fpr_arr = np.asarray(fpr)

    if not np.all((0 <= fpr_arr) & (fpr_arr <= 1)):
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")

    n_pos = tp_counts[-1]
    n_neg = fp_counts[-1]

    target_fp_count = n_neg * fpr_arr

    # Find threshold where FP count just exceeds target
    threshold = np.minimum(
        np.searchsorted(fp_counts, target_fp_count, side="right"),
        np.size(fp_counts) - 1,
    )

    # Interpolate between adjacent thresholds
    fp_left = fp_counts[threshold - 1]
    fp_right = fp_counts[threshold]
    q = (target_fp_count - fp_left) / (fp_right - fp_left)

    tp_left = tp_counts[threshold - 1]
    tp_right = tp_counts[threshold]
    result = (tp_left + q * (tp_right - tp_left)) / n_pos

    # Return scalar if input was scalar
    return float(result) if np.isscalar(fpr) else result


def epsilon_raw_counts_helper(
    tp_counts: np.ndarray,
    fp_counts: np.ndarray,
    min_count: int,
    delta: float,
) -> float:
    """Estimate epsilon given TP/FP counts at each threshold.

    Computes the maximum epsilon for which the observed true positive and false
    positive counts are consistent with (epsilon, delta)-DP, using the raw counts
    method.

    Args:
        tp_counts: True positive counts at each threshold (non-decreasing)
        fp_counts: False positive counts at each threshold (non-decreasing)
        min_count: Minimum count to consider (for statistical significance)
        delta: DP delta parameter

    Returns:
        Estimated epsilon lower bound
    """
    n_pos = tp_counts[-1]
    n_neg = fp_counts[-1]

    if min_count >= n_neg:
        return 0.0

    min_fpr = min_count / n_neg
    tpr_at_min_fpr = tpr_at_given_fpr(min_fpr, tp_counts, fp_counts)

    if delta == 0:
        return np.log(tpr_at_min_fpr / min_fpr)

    # Compute epsilon at the minimum FPR point
    if tpr_at_min_fpr > delta:
        initial_eps = max(0, np.log(tpr_at_min_fpr - delta) - np.log(min_fpr))
    else:
        initial_eps = 0.0

    # Compute epsilon at all valid thresholds
    tpr = tp_counts / n_pos
    fpr = fp_counts / n_neg
    valid = (fp_counts >= min_count) & (tpr > delta)
    eps = np.log(tpr[valid] - delta) - np.log(fpr[valid])

    return float(np.max(eps, initial=initial_eps))


def random_partition(
    scores: np.ndarray,
    rng: np.random.Generator,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split a score array into two parts.

    Args:
        scores: Array to split
        rng: NumPy random number generator
        p: Fraction for first part (in (0, 1))

    Returns:
        Tuple of (first_part, second_part) where first_part has approximately
        p * len(scores) elements.

    Raises:
        ValueError: If p is not in (0, 1)
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p}")

    perm = rng.permutation(len(scores))
    split_idx = int(len(scores) * p)
    return scores[perm[:split_idx]], scores[perm[split_idx:]]


def one_run_p_value(
    m: int, n_guess: int, n_correct: int, eps: float, delta: float
) -> float:
    """Compute p-value for one-shot privacy audit.

    Based on Nasr et al. (2023), https://arxiv.org/pdf/2305.08846

    Args:
        m: Number of canaries
        n_guess: Number of guesses (canaries predicted as held-in)
        n_correct: Number of correct guesses
        eps: Epsilon value to test
        delta: Delta value to test

    Returns:
        P-value for rejecting the (epsilon, delta)-DP hypothesis
    """
    q = scipy.special.expit(eps)  # logistic function
    beta = scipy.stats.binom.sf(n_correct - 1, n_guess, q)

    if delta == 0:
        return beta

    i_vals = np.arange(1, n_correct + 1)
    cum_sums = scipy.stats.binom.sf(n_correct - i_vals - 1, n_guess, q) - beta
    alpha = np.max(cum_sums / i_vals, initial=0)

    return min(beta + alpha * delta * 2 * m, 1.0)


__all__ = [
    "log_sub",
    "clopper_pearson_upper",
    "pareto_frontier",
    "get_tn_fn_counts",
    "tpr_at_given_fpr",
    "epsilon_raw_counts_helper",
    "random_partition",
    "one_run_p_value",
]
