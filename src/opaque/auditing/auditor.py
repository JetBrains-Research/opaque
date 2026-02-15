"""Empirical privacy auditing for differential privacy.

This module provides functional tools for empirically auditing privacy guarantees
using membership inference attacks on canary examples.

The API follows the functional patterns in the rest of Opaque:
- Pure functions that take data and return results
- No mutable state
- Composable building blocks

Example:
    >>> from opaque.auditing import epsilon_clopper_pearson, attack_auroc
    >>> eps = epsilon_clopper_pearson(in_scores, out_scores, significance=0.05)
    >>> auroc = attack_auroc(in_scores, out_scores)
"""

from collections import namedtuple
from collections.abc import Sequence
from concurrent import futures

import numpy as np
import scipy.stats

from opaque.auditing.bootstrap import BootstrapParams
from opaque.auditing.helpers import (
    clopper_pearson_upper,
    epsilon_raw_counts_helper,
    get_tn_fn_counts,
    one_run_p_value,
    random_partition,
    tpr_at_given_fpr as _tpr_at_given_fpr,
)

# Type alias for scores
Scores = Sequence[float] | np.ndarray

# Named tuple for comprehensive audit results
AuditResult = namedtuple(
    "AuditResult",
    ["epsilon", "auroc", "tpr_at_low_fpr", "max_accuracy"],
)
"""Results from a comprehensive privacy audit.

Fields:
    epsilon: Estimated epsilon lower bound at the specified significance level.
    auroc: Area under the ROC curve for the membership inference attack.
    tpr_at_low_fpr: True positive rate at 1% false positive rate.
    max_accuracy: Maximum classification accuracy achievable.
"""


# =============================================================================
# Epsilon estimation functions
# =============================================================================


def epsilon_clopper_pearson(
    in_scores: Scores,
    out_scores: Scores,
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
    thresholds, _, _ = get_tn_fn_counts(in_arr, out_arr)
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

    fnr_ub = clopper_pearson_upper(fn, n_in, significance / 2)
    fpr_ub = clopper_pearson_upper(fp, n_out, significance / 2)

    tpr_lb = 1 - fnr_ub
    if tpr_lb <= delta:
        return 0.0

    return max(0.0, float(np.log(tpr_lb - delta) - np.log(fpr_ub)))


def epsilon_one_run(
    in_scores: Scores,
    out_scores: Scores,
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
    thresholds, _, _ = get_tn_fn_counts(in_arr, out_arr)
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
        p_val = one_run_p_value(m, n_guess, n_correct, eps_mid, delta)
        if p_val < significance:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid

    return eps_lo


def epsilon_raw_counts(
    in_scores: Scores,
    out_scores: Scores,
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

    _, tn_counts, fn_counts = get_tn_fn_counts(in_scores, out_scores)

    tp_counts = (fn_counts[-1] - fn_counts)[::-1]
    fp_counts = (tn_counts[-1] - tn_counts)[::-1]

    return max(0.0, epsilon_raw_counts_helper(tp_counts, fp_counts, min_count, delta))


# =============================================================================
# Utility metrics
# =============================================================================


def attack_auroc(in_scores: Scores, out_scores: Scores) -> float:
    """Area under ROC curve for the membership inference attack.

    AUROC = 0.5 means random guessing, AUROC = 1.0 means perfect attack.

    Args:
        in_scores: Attack scores for held-in canaries (training set).
        out_scores: Attack scores for held-out canaries (test set).

    Returns:
        AUROC value in [0, 1].

    Example:
        >>> auroc = attack_auroc(in_scores, out_scores)
        >>> print(f"Attack AUROC: {auroc:.3f}")
    """
    _, tn_counts, fn_counts = get_tn_fn_counts(in_scores, out_scores)

    tnr = tn_counts / tn_counts[-1]
    fnr = fn_counts / fn_counts[-1]
    return float(0.5 * np.dot(tnr[:-1] + tnr[1:], fnr[1:] - fnr[:-1]))


def tpr_at_fpr(
    in_scores: Scores,
    out_scores: Scores,
    fpr: float | np.ndarray,
) -> float | np.ndarray:
    """True positive rate at a given false positive rate.

    Args:
        in_scores: Attack scores for held-in canaries (training set).
        out_scores: Attack scores for held-out canaries (test set).
        fpr: Target false positive rate(s) in [0, 1].

    Returns:
        TPR value(s) at the specified FPR(s).

    Example:
        >>> tpr = tpr_at_fpr(in_scores, out_scores, fpr=0.01)
        >>> print(f"TPR at 1% FPR: {tpr:.3f}")
    """
    fpr_arr = np.asarray(fpr)
    if not np.all((0 <= fpr_arr) & (fpr_arr <= 1)):
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")

    _, tn_counts, fn_counts = get_tn_fn_counts(in_scores, out_scores)

    tp_counts = (fn_counts[-1] - fn_counts)[::-1]
    fp_counts = (tn_counts[-1] - tn_counts)[::-1]

    return _tpr_at_given_fpr(fpr, tp_counts, fp_counts)


def max_accuracy(
    in_scores: Scores,
    out_scores: Scores,
    *,
    prevalence: float | None = None,
) -> float:
    """Maximum classification accuracy achievable.

    Args:
        in_scores: Attack scores for held-in canaries (training set).
        out_scores: Attack scores for held-out canaries (test set).
        prevalence: Fraction of positives in population. Default: use sample ratio.

    Returns:
        Maximum accuracy across all thresholds.

    Example:
        >>> acc = max_accuracy(in_scores, out_scores)
        >>> print(f"Max accuracy: {acc:.3f}")
    """
    _, tn_counts, fn_counts = get_tn_fn_counts(in_scores, out_scores)

    n_pos = fn_counts[-1]
    n_neg = tn_counts[-1]

    if prevalence is None:
        prevalence = n_pos / (n_pos + n_neg)

    tp_counts = n_pos - fn_counts
    tnr = tn_counts / n_neg
    tpr = tp_counts / n_pos

    return float(np.max(tpr * prevalence + tnr * (1 - prevalence)))


# =============================================================================
# Convenience function
# =============================================================================


def audit(
    in_scores: Scores,
    out_scores: Scores,
    significance: float = 0.05,
    delta: float = 0.0,
    *,
    method: str = "clopper_pearson",
) -> AuditResult:
    """Run a comprehensive privacy audit.

    Convenience function that computes epsilon and all utility metrics.

    Args:
        in_scores: Attack scores for held-in canaries (training set).
        out_scores: Attack scores for held-out canaries (test set).
        significance: Allowed failure probability. Default: 0.05.
        delta: DP delta parameter. Default: 0 (pure DP).
        method: Epsilon estimation method. One of:
            - "clopper_pearson" (default): Conservative statistical bounds
            - "raw_counts": Direct computation (less conservative)
            - "one_run": Likelihood-ratio method from Nasr et al.

    Returns:
        AuditResult with epsilon, auroc, tpr_at_low_fpr, max_accuracy.

    Example:
        >>> result = audit(in_scores, out_scores, significance=0.05, delta=1e-5)
        >>> print(f"Epsilon: {result.epsilon:.2f}, AUROC: {result.auroc:.3f}")
    """
    match method:
        case "clopper_pearson":
            eps = epsilon_clopper_pearson(in_scores, out_scores, significance, delta)
        case "raw_counts":
            eps = epsilon_raw_counts(in_scores, out_scores, delta=delta)
        case "one_run":
            eps = epsilon_one_run(in_scores, out_scores, significance, delta)
        case _:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Must be one of: 'clopper_pearson', 'raw_counts', 'one_run'"
            )

    return AuditResult(
        epsilon=eps,
        auroc=attack_auroc(in_scores, out_scores),
        tpr_at_low_fpr=tpr_at_fpr(in_scores, out_scores, 0.01),
        max_accuracy=max_accuracy(in_scores, out_scores),
    )


# =============================================================================
# Bootstrap wrapper
# =============================================================================


def bootstrap(
    fn,
    in_scores: Scores,
    out_scores: Scores,
    params: BootstrapParams,
) -> np.ndarray:
    """Compute bootstrapped quantiles for any auditing function.

    Args:
        fn: Function with signature fn(in_scores, out_scores, ...) -> float
        in_scores: Attack scores for held-in canaries.
        out_scores: Attack scores for held-out canaries.
        params: Bootstrap parameters.

    Returns:
        Array of quantiles specified in params.quantiles.

    Example:
        >>> params = BootstrapParams(num_samples=1000, seed=42)
        >>> auroc_ci = bootstrap(attack_auroc, in_scores, out_scores, params)
        >>> print(f"AUROC 95% CI: [{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]")
    """
    in_arr = np.asarray(in_scores)
    out_arr = np.asarray(out_scores)
    n_in, n_out = len(in_arr), len(out_arr)

    rng = np.random.default_rng(seed=params.seed)
    seeds = rng.integers(np.iinfo(np.int64).max, size=params.num_samples)

    def get_value(seed):
        inner_rng = np.random.default_rng(seed=seed)
        in_sample = inner_rng.choice(in_arr, size=n_in)
        out_sample = inner_rng.choice(out_arr, size=n_out)
        return fn(in_sample, out_sample)

    with futures.ThreadPoolExecutor() as pool:
        values = np.array(list(pool.map(get_value, seeds)))

    if not params.bias_correction:
        return np.quantile(values, params.quantiles, method="linear")

    # Bias-corrected bootstrap (BCa)
    full_estimate = fn(in_arr, out_arr)
    prop_less = (np.sum(values < full_estimate) + 1) / (params.num_samples + 2)
    z0 = scipy.stats.norm.ppf(prop_less)

    if params.acceleration:
        # Jackknife for acceleration
        def delete_in(i):
            return fn(np.delete(in_arr, i), out_arr)

        def delete_out(i):
            return fn(in_arr, np.delete(out_arr, i))

        with futures.ThreadPoolExecutor() as pool:
            jk = list(pool.map(delete_in, range(n_in)))
            jk.extend(pool.map(delete_out, range(n_out)))

        jk_mean = np.mean(jk)
        num = np.sum((jk_mean - np.array(jk)) ** 3)
        denom = 6 * np.sum((jk_mean - np.array(jk)) ** 2) ** 1.5
        accel = 0.0 if denom == 0 else num / denom
    else:
        accel = 0.0

    z_q = scipy.stats.norm.ppf(params.quantiles)
    corrected = scipy.stats.norm.cdf(z0 + (z0 + z_q) / (1 - accel * (z0 + z_q)))

    return np.quantile(values, corrected, method="linear")


__all__ = [
    # Epsilon estimation
    "epsilon_clopper_pearson",
    "epsilon_one_run",
    "epsilon_raw_counts",
    # Utility metrics
    "attack_auroc",
    "tpr_at_fpr",
    "max_accuracy",
    # Convenience
    "audit",
    "AuditResult",
    # Bootstrap
    "bootstrap",
]
