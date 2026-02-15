"""Empirical privacy auditing for differential privacy.

This module provides functional tools for empirically validating privacy guarantees
using membership inference attacks on canary examples.

Functional API:
    >>> from opaque.auditing import epsilon_clopper_pearson, attack_auroc
    >>> eps = epsilon_clopper_pearson(in_scores, out_scores, significance=0.05)
    >>> auroc = attack_auroc(in_scores, out_scores)

Convenience function for full audit:
    >>> from opaque.auditing import audit
    >>> result = audit(in_scores, out_scores, significance=0.05, delta=1e-5)
    >>> print(f"Epsilon: {result.epsilon:.2f}, AUROC: {result.auroc:.3f}")

Bootstrap for confidence intervals:
    >>> from opaque.auditing import bootstrap, attack_auroc, BootstrapParams
    >>> params = BootstrapParams(num_samples=1000, seed=42)
    >>> auroc_ci = bootstrap(attack_auroc, in_scores, out_scores, params)

References:
    - Nasr et al. (2023), https://arxiv.org/pdf/2305.08846
    - Carlini et al. (2022), https://arxiv.org/pdf/2112.03570
"""

from opaque.auditing.auditor import (
    AuditResult,
    attack_auroc,
    audit,
    bootstrap,
    epsilon_clopper_pearson,
    epsilon_one_run,
    epsilon_raw_counts,
    max_accuracy,
    tpr_at_fpr,
)
from opaque.auditing.bootstrap import BootstrapParams

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
    "BootstrapParams",
]
