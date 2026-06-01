# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Batched per-sequence log-probability helper.

:func:`get_batch_logps` (the TRL ``get_batch_logps`` analogue) applies the
causal-LM shift, builds an ignore-index (``-100``) loss mask, gathers per-token
log-probs while keeping gather indices valid, and reduces over the sequence
axis (sum, or mask-weighted mean when ``average_log_prob=True``).

It is a pure, vmap-safe tensor function (plan §3.4): the ignore-index handling
uses :func:`torch.where` rather than Python control flow, and the mean divisor
is clamped to ``>= 1`` so an all-ignored sequence does not divide by zero.
Negative-axis indexing makes it work both per-example under ``vmap``
(``logits`` ``(T, V)``, ``labels`` ``(T,)``) and on a batched input.
"""

from __future__ import annotations

import torch

from ._gather import selective_log_softmax

__all__ = ["get_batch_logps"]

_IGNORE_INDEX = -100


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Per-sequence log-probability with ``-100`` ignore-index masking.

    Args:
        logits: Float tensor of shape ``(..., T, V)``.
        labels: Integer tensor of shape ``(..., T)``. Positions equal to
            ``-100`` are ignored and contribute ``0`` to the sum.
        average_log_prob: If ``True``, divide the per-sequence sum by the
            number of non-ignored tokens (clamped to ``>= 1``); otherwise
            return the raw sum.

    Returns:
        Float tensor of shape ``(...)`` (one scalar per sequence).
    """
    # Causal-LM shift.
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]

    loss_mask = shifted_labels != _IGNORE_INDEX
    # Replace ignored labels with a valid index (0) so gather never sees -100.
    safe_labels = torch.where(
        loss_mask, shifted_labels, torch.zeros_like(shifted_labels)
    )

    per_token_logp = selective_log_softmax(shifted_logits, safe_labels)
    masked = per_token_logp * loss_mask.to(per_token_logp.dtype)
    summed = masked.sum(dim=-1)

    if average_log_prob:
        token_counts = loss_mask.sum(dim=-1).clamp(min=1).to(summed.dtype)
        return summed / token_counts
    return summed
