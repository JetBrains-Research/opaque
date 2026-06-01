"""Per-example KL estimator for KTO-family losses.

Computes the per-example KL term ``log pi - log pi_ref`` consumed by KTO's
Tier-2 detached batch-mean aggregate (§3.3, §7.8). The ``detach`` flag is
load-bearing for DP-purity: the KTO aggregate must be detached before it
re-enters the per-example loss (KTO paper Eq. 8 / TRL ``kto_trainer.py``:
"we do not backpropagate through z_0"). Callers that want the aggregate to
contribute gradient (rare) may pass ``detach=False``.
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
            graph so it cannot backpropagate (DP-purity for the KTO aggregate).
        clamp_min: Lower bound applied to the estimate; the KTO KL term is
            clamped at ``0`` so it stays non-negative.

    Returns:
        The per-example KL estimate, clamped at ``clamp_min`` (and detached
        when ``detach`` is ``True``).
    """
    kl = policy_logp - ref_logp
    if detach:
        kl = kl.detach()
    return kl.clamp(min=clamp_min)
