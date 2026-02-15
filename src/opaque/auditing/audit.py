"""Convenience function and result type for privacy auditing."""

from __future__ import annotations

import dataclasses

import numpy as np

from opaque.auditing.epsilon import (
    epsilon_clopper_pearson,
    epsilon_one_run,
    epsilon_raw_counts,
)
from opaque.auditing.metrics import attack_auroc, max_accuracy, tpr_at_fpr

__all__ = ["AuditResult", "audit"]


@dataclasses.dataclass(frozen=True)
class AuditResult:
    """Results from a comprehensive privacy audit.

    Attributes:
        epsilon: Estimated epsilon lower bound at the specified significance level.
        auroc: Area under the ROC curve for the membership inference attack.
        tpr_at_low_fpr: True positive rate at 1% false positive rate.
        max_accuracy: Maximum classification accuracy achievable.
    """

    epsilon: float
    auroc: float
    tpr_at_low_fpr: float
    max_accuracy: float


def audit(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
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
