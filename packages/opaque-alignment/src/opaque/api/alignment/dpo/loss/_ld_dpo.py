# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""LD-DPO length-desensitised sequence logp split.

Implements the length-desensitisation decomposition from:

    Liu, W., Bai, Y., Han, C., Weng, R., Xu, J., Cao, X., Wang, J., & Cai, X.
    (2024). Length Desensitization in Direct Preference Optimization.
    arXiv:2409.06411.

LD-DPO observes that vanilla DPO is biased by completion length: longer
completions accumulate more (signed) log-prob mass, which entangles a verbose
bias into the preference signal.  LD-DPO splits each completion's logp into a
"public"/shared prefix (weighted ``1.0``) and a length-desensitised tail
(weighted by ``alpha`` ∈ ``[0, 1]``).  Setting ``alpha < 1`` damps the tail's
contribution, desensitising the objective to completion length::

    pos    = completion_mask.cumsum(-1)
    weight = where((pos >= 1) & (pos <= shared_prefix_len), 1.0, alpha)
    logp   = (per_token_logps * completion_mask * weight).sum(-1)

At ``alpha = 1`` this reduces to the plain masked-sum sequence logp; at
``alpha = 0`` only the shared-prefix tokens contribute.

Positions are completion-relative: prompt and padding positions remain zero,
while completion tokens are numbered ``1, 2, ...``. ``shared_prefix_len`` may
be a Python ``int`` or a per-example tensor; it is broadcast against the
position axis, so a per-example tensor of shape ``(..., 1)`` selects a
different prefix length per example.
"""

from __future__ import annotations

import torch

__all__ = ["ld_dpo_split"]


def ld_dpo_split(
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    shared_prefix_len: torch.Tensor | int,
    alpha: float,
) -> torch.Tensor:
    """Length-desensitised sequence logp split (LD-DPO, arXiv:2409.06411).

    Weights completion tokens at the shared-prefix positions by ``1.0`` and
    tokens beyond the prefix by ``alpha``, then masked-sums over the sequence
    axis::

        pos    = completion_mask.cumsum(-1)
        weight = where((pos >= 1) & (pos <= shared_prefix_len), 1.0, alpha)
        return (per_token_logps * completion_mask * weight).sum(-1)

    Args:
        per_token_logps: Per-token log-probabilities of the completion. Shape
            ``(..., T)``; leading dims may be empty (per-example under
            ``vmap``) or a batch axis.
        completion_mask: Tensor of shape ``(..., T)``; non-zero where a token
            belongs to the completion span. Cast to the logp dtype.
        shared_prefix_len: Length of the shared/public prefix. A Python
            ``int`` (same prefix for every example) or a per-example tensor
            that broadcasts against the position axis ``(T,)`` (e.g. shape
            ``(..., 1)``). The first ``shared_prefix_len`` completion tokens are
            weighted ``1.0``; later completion tokens are weighted ``alpha``.
        alpha: Weight on the length-desensitised tail, typically ``∈ [0, 1]``.
            ``alpha = 1`` recovers the plain masked-sum; ``alpha = 0`` keeps
            only the prefix tokens.

    Returns:
        Float tensor of shape ``(...)`` (one length-desensitised logp per
        sequence).
    """
    mask = completion_mask.to(per_token_logps.dtype)
    completion_pos = (completion_mask != 0).cumsum(dim=-1)
    prefix_len = shared_prefix_len
    if isinstance(prefix_len, torch.Tensor):
        while prefix_len.ndim < completion_pos.ndim:
            prefix_len = prefix_len.unsqueeze(-1)
    is_prefix = (completion_pos >= 1) & (completion_pos <= prefix_len)
    weight = torch.where(
        is_prefix,
        torch.ones((), dtype=per_token_logps.dtype, device=per_token_logps.device),
        torch.full(
            (), alpha, dtype=per_token_logps.dtype, device=per_token_logps.device
        ),
    )
    masked = per_token_logps * mask * weight
    return masked.sum(dim=-1)
