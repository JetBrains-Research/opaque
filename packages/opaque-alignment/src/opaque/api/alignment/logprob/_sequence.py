# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-sequence completion log-probability.

:func:`sequence_logp` is the standard DPO / causal-LM per-sequence logp: it
applies the next-token shift (predictions ``logits[..., :-1, :]`` align with
targets ``input_ids[..., 1:]``), gathers per-token log-probs via
:func:`selective_log_softmax`, masks to the completion span, and sums over the
sequence axis.

It is a pure tensor function: negative-axis indexing is
used throughout so it works both when called per-example under ``vmap``
(``logits`` ``(T, V)``, ``input_ids`` ``(T,)``) and when called on a batched
input (``logits`` ``(B, T, V)``, ``input_ids`` ``(B, T)``).

The ``ld_alpha`` (LD-DPO, arXiv:2409.10524) length-desensitised logp split is
a documented stub here; the real decomposition lands in Phase γ
(``loss/dpo/_ld_dpo.py``).
"""

from __future__ import annotations

import torch

from ._gather import selective_log_softmax

__all__ = ["sequence_logp"]


def sequence_logp(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    ld_alpha: float | None = None,
    shared_prefix_len: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum of completion-token log-probabilities per sequence.

    Applies the causal-LM shift, computes per-token log-probs, multiplies by
    the (shifted) completion mask, and sums over the sequence (last remaining)
    axis.

    Args:
        logits: Float tensor of shape ``(..., T, V)``. The ``(...)`` leading
            dims may be empty (per-example under ``vmap``) or a batch axis.
        input_ids: Integer tensor of shape ``(..., T)`` (token ids).
        completion_mask: Tensor of shape ``(..., T)``; non-zero where a token
            belongs to the completion span and should contribute to the logp.
            It is cast to the logits dtype before multiplying.
        ld_alpha: LD-DPO length-desensitisation coefficient. Must be ``None``
            in this phase; any other value raises :class:`NotImplementedError`.
        shared_prefix_len: LD-DPO shared-prefix length. Accepted and ignored
            when ``ld_alpha is None``.

    Returns:
        Float tensor of shape ``(...)`` (one summed logp per sequence).

    Raises:
        NotImplementedError: If ``ld_alpha is not None``.
    """
    if ld_alpha is not None:
        raise NotImplementedError(
            "ld_alpha decomposition lands in Phase γ (_ld_dpo); use ld_alpha=None"
        )
    del shared_prefix_len  # accepted, ignored when ld_alpha is None

    # Causal-LM shift: predict token t+1 from the logits at position t.
    shifted_logits = logits[..., :-1, :]
    target_ids = input_ids[..., 1:]
    target_mask = completion_mask[..., 1:]

    per_token_logp = selective_log_softmax(shifted_logits, target_ids)
    masked = per_token_logp * target_mask.to(per_token_logp.dtype)
    return masked.sum(dim=-1)
