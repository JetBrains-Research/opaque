# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""WPO per-example weighting — Weighted Preference Optimization.

Implements the policy-probability reweighting from:

    Zhou, W., Agrawal, R., Zhang, S., Indurthi, S. R., Zhao, S., Song, K.,
    Xu, S., & Zhu, C. (2024). WPO: Enhancing RLHF with Weighted Preference
    Optimization. arXiv:2406.11827.

WPO simulates on-policy preference learning under an off-policy dataset by
weighting each preference pair by how likely the *current* policy is to have
produced the completion.  The weight is the policy's average per-token
probability on the completion::

    avg_logp = (per_token_logps * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    weight   = avg_logp.detach().exp()

The weight is ``.detach()``-ed, so it carries no gradient — it acts purely as
a per-example reweighting of the loss, not as an additional learnable path.  A
non-detached weight would couple the gradient through the probability term.
"""

from __future__ import annotations

import torch

__all__ = ["wpo_weights"]


def wpo_weights(
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-example WPO weight (arXiv:2406.11827).

    Computes the policy's average per-token probability on the completion and
    returns its (detached) exponential::

        avg_logp = (per_token_logps * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return avg_logp.detach().exp()

    The result is **detached**: it contributes no gradient and serves only as
    a per-example multiplicative reweighting of the downstream loss, keeping
    the per-example loss.

    Args:
        per_token_logps: Per-token log-probabilities of the completion under
            the current policy. Shape ``(..., T)``; the leading dims may be
            empty (per-example under ``vmap``) or a batch axis.
        completion_mask: Tensor of shape ``(..., T)``; non-zero where a token
            belongs to the completion span. Cast to the logp dtype before
            multiplying.

    Returns:
        Detached per-example weight tensor of shape ``(...)`` (one weight per
        sequence). All-zero mask rows are protected by ``clamp(min=1)`` so
        there is no division by zero.
    """
    mask = completion_mask.to(per_token_logps.dtype)
    token_count = mask.sum(dim=-1).clamp(min=1)
    avg_logp = (per_token_logps * mask).sum(dim=-1) / token_count
    return avg_logp.detach().exp()
