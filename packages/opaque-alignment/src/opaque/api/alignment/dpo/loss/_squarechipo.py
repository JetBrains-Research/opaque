"""SquareChiPO (Square Chi-squared Preference Optimisation) loss.

Implements the first optimal-rate DP-DPO loss from:

    arXiv:2505.21395 — "SquareChiPO: Optimal Differentially Private
    Direct Preference Optimization via Chi-Squared Divergence".

SquareChiPO is derived by minimising the χ²-divergence between the
policy and the optimal reward-weighted reference policy.  The resulting
per-example loss has a squared-sigmoid form that yields provably
optimal privacy-utility trade-offs for DP-DPO training.

Formula::

    L = 0.5 · (σ(β·Δ) − 1)²

where ``Δ = chosen_logratio − rejected_logratio`` and ``σ`` is the
logistic sigmoid.
"""

from __future__ import annotations

import torch

__all__ = ["squarechipo_loss"]


def squarechipo_loss(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """SquareChiPO per-example loss (arXiv:2505.21395).

    Computes the squared deviation of the sigmoid policy ratio from 1::

        Δ = chosen_logratio - rejected_logratio
        L = 0.5 · (σ(β·Δ) − 1)²

    At ``Δ = 0`` (equally preferred pair) this evaluates to
    ``0.5 · (0.5 − 1)² = 0.125``, and the gradient is non-zero,
    providing a learning signal even at the decision boundary.

    Args:
        chosen_logratio: Per-example scalar log-ratio for the chosen
            completion, ``log π(y_w | x) − log π_ref(y_w | x)``.
            May be 0-dim (single example) or ``(B,)`` (batch).
        rejected_logratio: Per-example scalar log-ratio for the rejected
            completion, ``log π(y_l | x) − log π_ref(y_l | x)``.
            Same shape as *chosen_logratio*.
        beta: KL-regularisation temperature (DPO β).  Positive float;
            typical values in [0.01, 0.5].

    Returns:
        Per-example scalar loss tensor with the same shape as the inputs.
    """
    logits = chosen_logratio - rejected_logratio
    return 0.5 * (torch.sigmoid(beta * logits) - 1.0) ** 2
