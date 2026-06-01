"""Per-example KL estimator ``log pi - log pi_ref``.

A pure metric: the per-example difference of policy and reference sequence
log-probabilities, optionally detached and lower-clamped. The ``detach`` flag
matters when the estimate is consumed as a batch aggregate that re-enters a
per-example loss — detaching keeps it out of the released gradient (DP-purity).
"""

from __future__ import annotations

import torch

__all__ = ["kl_estimator"]


def kl_estimator(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    *,
    detach: bool = True,
    clamp_min: float = 0.0,
) -> torch.Tensor:
    """Estimate the per-example KL term ``log pi - log pi_ref``.

    Args:
        policy_logp: Per-example policy sequence log-probabilities.
        ref_logp: Per-example reference sequence log-probabilities.
        detach: When ``True`` (default), detach the estimate from the autograd
            graph so it cannot backpropagate.
        clamp_min: Lower bound applied to the estimate (a KL term is
            non-negative, so the default ``0.0`` keeps it so).

    Returns:
        The per-example KL estimate, clamped at ``clamp_min`` (and detached
        when ``detach`` is ``True``).
    """
    kl = policy_logp - ref_logp
    if detach:
        kl = kl.detach()
    return kl.clamp(min=clamp_min)
