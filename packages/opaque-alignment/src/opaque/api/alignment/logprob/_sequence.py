# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-sequence completion log-probability.

:func:`sequence_logp` is the standard DPO / causal-LM per-sequence logp: it
applies the next-token shift (predictions ``logits[..., :-1, :]`` align with
targets ``input_ids[..., 1:]``), gathers per-token log-probs via
:func:`selective_log_softmax`, masks to the completion span, and sums over the
sequence axis.

Negative-axis indexing is used throughout so it works both when called
per-example (``logits`` ``(T, V)``, ``input_ids`` ``(T,)``) and when called on a
batched input (``logits`` ``(B, T, V)``, ``input_ids`` ``(B, T)``).

The ``ld_alpha`` (LD-DPO, arXiv:2409.10524) length-desensitised logp split is
not implemented here; pass ``ld_alpha=None``.
"""

from __future__ import annotations

import torch

from opaque.api.alignment._fused_lce import lce_available, linear_nll_sum

from ._gather import selective_log_softmax

__all__ = ["sequence_logp", "fused_sequence_logp", "length_normalize"]


def length_normalize(
    logp: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Divide a summed completion log-prob by its completion-token count.

    Turns a :func:`sequence_logp` / :func:`fused_sequence_logp` output (the
    *sum* of completion log-probs) into the per-token mean ``(1/|y|)·log π(y)`` —
    the length-normalized reward used by SimPO and ORPO. The completion length is
    counted with the same next-token shift as :func:`sequence_logp`
    (``completion_mask[..., 1:]``), clamped to ``>= 1``.

    Args:
        logp: Summed completion log-prob, shape ``(...,)`` (per example under
            ``vmap``, or batched).
        completion_mask: ``(..., T)``; non-zero on completion-span tokens (the
            same mask passed to ``sequence_logp``).

    Returns:
        Per-example length-normalized log-prob, same shape as *logp*.
    """
    count = completion_mask[..., 1:].sum(-1).clamp(min=1)
    return logp / count.to(logp.dtype)


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
        ld_alpha: LD-DPO length-desensitisation coefficient. Must be ``None``;
            any other value raises :class:`NotImplementedError`.
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


def fused_sequence_logp(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Memory-efficient :func:`sequence_logp` over hidden states (fused, per-example).

    Mathematically ``sequence_logp(hidden_states @ lm_head_weight.T, input_ids,
    completion_mask)``, but on CUDA + half precision with
    ``opaque-alignment[patches]`` installed it computes the completion log-prob
    via the patches fused linear-CE kernel — no ``(T, V)`` logits are
    materialised and the LSE is recomputed in the backward. Otherwise it falls
    back to the eager form.

    Since ``sequence_logp = Σ_completion log p = −Σ_completion CE``, the kernel is
    the preference-logp kernel: encode the completion span as the kept labels
    (everything else → ``-100``), take the kernel's CE sum, and negate.

    **Per-example.** Pass one sequence ``(T, H)`` and drive with
    ``vmap(grad(...))`` (how ``train_dpo`` already computes logp inside
    ``clipped_grad``); the kernel's merged vmap rules then make the chosen /
    rejected microbatch one forward + one backward kernel. Unlike
    :func:`sequence_logp`, it is *not* meant to be called directly on a batch
    axis. The LD-DPO ``ld_alpha`` split is not supported on the fused path; use
    the eager :func:`sequence_logp` for that.

    Args:
        hidden_states: Last-layer hidden states ``(T, H)`` for one sequence
            (batched to ``(B, T, H)`` only by an outer ``vmap``).
        lm_head_weight: LM-head weight ``(V, H)``; logits are
            ``hidden_states @ lm_head_weight.T``.
        input_ids: Token ids ``(T,)``.
        completion_mask: ``(T,)``; non-zero on completion-span tokens.

    Returns:
        Scalar summed completion log-prob (matches :func:`sequence_logp`).
    """
    if lce_available(hidden_states):
        # Completion tokens keep their id; everything else → ignore_index, so the
        # kernel's CE sum is exactly −Σ_completion logp. The kernel applies the
        # next-token shift internally, matching sequence_logp's shifted mask.
        masked_labels = torch.where(
            completion_mask.to(torch.bool),
            input_ids,
            torch.full_like(input_ids, -100),
        )
        return -linear_nll_sum(hidden_states, lm_head_weight, masked_labels)
    logits = hidden_states @ lm_head_weight.transpose(-2, -1)
    return sequence_logp(logits, input_ids, completion_mask)
