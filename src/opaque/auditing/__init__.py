"""Empirical privacy auditing for differential privacy.

Construct an :class:`AuditResult` from membership-inference canary scores,
then query epsilon bounds, AUROC, and other metrics as methods.

Example:
    >>> from opaque.auditing import AuditResult
    >>> result = AuditResult(in_scores, out_scores)
    >>> print(f"Epsilon: {result.epsilon_clopper_pearson():.2f}")
    >>> print(f"AUROC: {result.auroc():.3f}")

References:
    - Nasr et al. (2023), https://arxiv.org/pdf/2305.08846
    - Carlini et al. (2022), https://arxiv.org/pdf/2112.03570
"""

from opaque.auditing.audit import AuditResult, CoinFlipExperiment
from opaque.auditing.bootstrap import BootstrapParams

__all__ = [
    "AuditResult",
    "BootstrapParams",
    "CoinFlipExperiment",
]
