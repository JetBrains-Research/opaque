"""Epsilon estimation functions for privacy auditing.

Provides lower bounds on epsilon using different statistical methods:
- Clopper-Pearson: Conservative binomial confidence intervals
- One-run: Likelihood-ratio test from Nasr et al. (2023)
- Raw counts: Direct computation (less conservative)

References:
    - Nasr et al. (2023), https://arxiv.org/pdf/2305.08846
    - Carlini et al. (2022), https://arxiv.org/pdf/2112.03570
"""

from __future__ import annotations

import numpy as np

from opaque.auditing.helpers import (
    _clopper_pearson_upper,
    _epsilon_raw_counts_helper,
    _get_tn_fn_counts,
    _one_run_p_value,
)

__all__ = [
    "epsilon_clopper_pearson",
    "epsilon_one_run",
    "epsilon_raw_counts",
]


def epsilon_clopper_pearson(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    significance: float = 0.05,
    delta: float = 0.0,
    *,
    threshold: float | None = None,
) -> float:
    """Estimate epsilon using Clopper-Pearson confidence intervals.

    Constructs conservative binomial confidence intervals for TPR/FPR and
    uses them to bound epsilon. Provides formal statistical guarantees.

    Uses Bonferroni correction over all thresholds unless an explicit
    threshold is provided.

    Args:
        in_scores: Attack scores for held-in canaries (training set).
        out_scores: Attack scores for held-out canaries (test set).
        significance: Allowed failure probability (1 - confidence). Default: 0.05.
        delta: DP delta parameter. Default: 0 (pure DP).
        threshold: If provided, use this specific threshold instead of
            searching over all thresholds with Bonferroni correction.

    Returns:
        Epsilon lower bound at the specified confidence level.

    Example:
        >>> eps = epsilon_clopper_pearson(in_scores, out_scores, significance=0.05)
        >>> print(f"Epsilon lower bound: {eps:.2f}")
    """
    if not 0 < significance < 0.5:
        raise ValueError(f"significance must be in (0, 0.5), got {significance}")
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got {delta}")

    in_arr = np.asarray(in_scores)
    out_arr = np.asarray(out_scores)
    n_in, n_out = len(in_arr), len(out_arr)

    if threshold is not None:
        return _epsilon_cp_at_threshold(
            in_arr, out_arr, n_in, n_out, threshold, significance, delta
        )

    # Bonferroni correction over all thresholds
    thresholds, _, _ = _get_tn_fn_counts(in_arr, out_arr)
    sig_corrected = significance / len(thresholds)

    return max(
        _epsilon_cp_at_threshold(in_arr, out_arr, n_in, n_out, t, sig_corrected, delta)
        for t in thresholds
    )


def _epsilon_cp_at_threshold(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    n_in: int,
    n_out: int,
    threshold: float,
    significance: float,
    delta: float,
) -> float:
    """Compute Clopper-Pearson epsilon at a specific threshold."""
    fn = np.sum(in_arr < threshold)
    fp = np.sum(out_arr >= threshold)

    fnr_ub = _clopper_pearson_upper(fn, n_in, significance / 2)
    fpr_ub = _clopper_pearson_upper(fp, n_out, significance / 2)

    tpr_lb = 1 - fnr_ub
    if tpr_lb <= delta:
        return 0.0

    return max(0.0, float(np.log(tpr_lb - delta) - np.log(fpr_ub)))


def epsilon_one_run(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    significance: float = 0.05,
    delta: float = 0.0,
    *,
    threshold: float | None = None,
    eps_max: float = 20.0,
    tol: float = 1e-4,
) -> float:
    """Estimate epsilon using the one-run method from Nasr et al. (2023).

    Uses a likelihood-ratio test tailored for DP auditing. Generally less
    conservative than Clopper-Pearson for the same sample size.

    Uses Bonferroni correction over all thresholds unless an explicit
    threshold is provided.

    Args:
        in_scores: Attack scores for held-in canaries (training set).
        out_scores: Attack scores for held-out canaries (test set).
        significance: Allowed failure probability (1 - confidence). Default: 0.05.
        delta: DP delta parameter. Default: 0 (pure DP).
        threshold: If provided, use this specific threshold.
        eps_max: Maximum epsilon to search. Default: 20.0.
        tol: Binary search tolerance. Default: 1e-4.

    Returns:
        Epsilon lower bound at the specified confidence level.

    Reference:
        Nasr et al. (2023), https://arxiv.org/pdf/2305.08846
    """
    if not 0 < significance < 0.5:
        raise ValueError(f"significance must be in (0, 0.5), got {significance}")
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got {delta}")

    in_arr = np.asarray(in_scores)
    out_arr = np.asarray(out_scores)
    m = len(in_arr) + len(out_arr)

    if threshold is not None:
        return _epsilon_one_run_at_threshold(
            in_arr, out_arr, m, threshold, significance, delta, eps_max, tol
        )

    # Bonferroni correction over all thresholds
    thresholds, _, _ = _get_tn_fn_counts(in_arr, out_arr)
    sig_corrected = significance / len(thresholds)

    return max(
        _epsilon_one_run_at_threshold(
            in_arr, out_arr, m, t, sig_corrected, delta, eps_max, tol
        )
        for t in thresholds
    )


def _epsilon_one_run_at_threshold(
    in_arr: np.ndarray,
    out_arr: np.ndarray,
    m: int,
    threshold: float,
    significance: float,
    delta: float,
    eps_max: float,
    tol: float,
) -> float:
    """Compute one-run epsilon at a specific threshold via binary search."""
    tp = np.sum(in_arr >= threshold)
    fp = np.sum(out_arr >= threshold)

    n_guess = tp + fp
    n_correct = tp

    if n_guess == 0 or n_correct == 0:
        return 0.0

    # Binary search for largest epsilon where p-value > significance
    eps_lo, eps_hi = 0.0, eps_max
    while eps_hi - eps_lo > tol:
        eps_mid = (eps_lo + eps_hi) / 2
        p_val = _one_run_p_value(m, n_guess, n_correct, eps_mid, delta)
        if p_val < significance:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid

    return eps_lo


def epsilon_raw_counts(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    min_count: int = 50,
    delta: float = 0.0,
) -> float:
    """Estimate epsilon from raw TPR/FPR counts.

    Direct computation without confidence intervals. Less conservative but
    higher variance than Clopper-Pearson.

    Args:
        in_scores: Attack scores for held-in canaries (training set).
        out_scores: Attack scores for held-out canaries (test set).
        min_count: Minimum FP count to consider a threshold. Default: 50.
        delta: DP delta parameter. Default: 0 (pure DP).

    Returns:
        Epsilon estimate.
    """
    if min_count < 1:
        raise ValueError(f"min_count must be positive, got {min_count}")
    if not 0 <= delta <= 1:
        raise ValueError(f"delta must be in [0, 1], got {delta}")

    _, tn_counts, fn_counts = _get_tn_fn_counts(in_scores, out_scores)

    tp_counts = (fn_counts[-1] - fn_counts)[::-1]
    fp_counts = (tn_counts[-1] - tn_counts)[::-1]

    return max(0.0, _epsilon_raw_counts_helper(tp_counts, fp_counts, min_count, delta))
