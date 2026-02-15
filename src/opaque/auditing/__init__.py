"""Empirical privacy auditing for differential privacy.

Functional tools for empirically validating privacy guarantees using
membership inference attacks on canary examples.

Example:
    >>> from opaque.auditing import audit
    >>> result = audit(in_scores, out_scores, significance=0.05, delta=1e-5)
    >>> print(f"Epsilon: {result.epsilon:.2f}, AUROC: {result.auroc:.3f}")

References:
    - Nasr et al. (2023), https://arxiv.org/pdf/2305.08846
    - Carlini et al. (2022), https://arxiv.org/pdf/2112.03570
"""

from opaque.auditing.audit import AuditResult, audit
from opaque.auditing.bootstrap import BootstrapParams, bootstrap
from opaque.auditing.epsilon import (
    epsilon_clopper_pearson,
    epsilon_one_run,
    epsilon_raw_counts,
)
from opaque.auditing.metrics import attack_auroc, max_accuracy, tpr_at_fpr

__all__ = [
    # Convenience
    "audit",
    "AuditResult",
    # Epsilon estimation
    "epsilon_clopper_pearson",
    "epsilon_one_run",
    "epsilon_raw_counts",
    # Metrics
    "attack_auroc",
    "tpr_at_fpr",
    "max_accuracy",
    # Bootstrap
    "bootstrap",
    "BootstrapParams",
]
