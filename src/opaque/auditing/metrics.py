"""Attack utility metrics for privacy auditing.

Provides metrics for evaluating membership inference attacks:
- AUROC: Area under the ROC curve
- TPR at FPR: True positive rate at a given false positive rate
- Max accuracy: Maximum classification accuracy
"""

from __future__ import annotations

import numpy as np

from opaque.auditing.helpers import _get_tn_fn_counts, _tpr_at_given_fpr

__all__ = ["attack_auroc", "tpr_at_fpr", "max_accuracy"]


def attack_auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """Area under ROC curve for the membership inference attack.

    AUROC = 0.5 means random guessing, AUROC = 1.0 means perfect attack.

    Args:
        in_scores: Attack scores for held-in canaries (training set).
        out_scores: Attack scores for held-out canaries (test set).

    Returns:
        AUROC value in [0, 1].

    Raises:
        ValueError: If either in_scores or out_scores is empty.

    Example:
        >>> auroc = attack_auroc(in_scores, out_scores)
        >>> print(f"Attack AUROC: {auroc:.3f}")
    """
    in_arr = np.asarray(in_scores)
    out_arr = np.asarray(out_scores)
    if len(in_arr) == 0 or len(out_arr) == 0:
        raise ValueError("in_scores and out_scores must be non-empty")

    _, tn_counts, fn_counts = _get_tn_fn_counts(in_arr, out_arr)

    tnr = tn_counts / tn_counts[-1]
    fnr = fn_counts / fn_counts[-1]
    return float(0.5 * np.dot(tnr[:-1] + tnr[1:], fnr[1:] - fnr[:-1]))


def tpr_at_fpr(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
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

    _, tn_counts, fn_counts = _get_tn_fn_counts(in_scores, out_scores)

    tp_counts = (fn_counts[-1] - fn_counts)[::-1]
    fp_counts = (tn_counts[-1] - tn_counts)[::-1]

    return _tpr_at_given_fpr(fpr, tp_counts, fp_counts)


def max_accuracy(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
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

    Raises:
        ValueError: If prevalence is not in [0, 1].

    Example:
        >>> acc = max_accuracy(in_scores, out_scores)
        >>> print(f"Max accuracy: {acc:.3f}")
    """
    if prevalence is not None and not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")

    _, tn_counts, fn_counts = _get_tn_fn_counts(in_scores, out_scores)

    n_pos = fn_counts[-1]
    n_neg = tn_counts[-1]

    if prevalence is None:
        prevalence = n_pos / (n_pos + n_neg)

    tp_counts = n_pos - fn_counts
    tnr = tn_counts / n_neg
    tpr = tp_counts / n_pos

    return float(np.max(tpr * prevalence + tnr * (1 - prevalence)))
