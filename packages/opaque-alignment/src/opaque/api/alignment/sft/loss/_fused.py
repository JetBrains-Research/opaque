# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Chunked fused-linear SFT loss — the opt-in memory-efficient path.

The eager SFT losses (:func:`nll_loss`, :func:`dft_loss`) take the
already-materialised ``(B, T, V)`` logits. For large vocabularies that tensor
dominates activation memory. :func:`fused_linear_sft_loss` instead takes the
*hidden states* and the ``lm_head`` weight and projects them in chunks of
``chunk_size`` examples, materialising at most ``(chunk_size, T, V)`` logits at
a time::

    peak logits memory  =  O(chunk_size · T · V)   (this function)
    instead of          =  O(B · T · V)            (eager, all-at-once)

It is **opt-in**: the eager path stays the default because it returns the logits
(so ``entropy_from_logits`` / ``mean_token_accuracy`` are free), whereas the
fused path returns only the per-example loss. The loss math is selected by
passing the eager loss function directly — ``loss_fn=nll_loss`` (default) or
``loss_fn=dft_loss`` — so there is no string registry and the fused / eager
paths stay symmetric (the same function object selects the variant in both).

Design (mirrors :mod:`opaque.api.alignment.dpo.kernel._fused_linear_preference`):
self-contained pure PyTorch, no custom ``autograd.Function``. Chunking is a pure
partition of the *non-interacting* batch axis, so ``torch.cat`` of the per-chunk
losses equals the all-at-once form — and so does its gradient. The result is
therefore independent of ``chunk_size`` up to float reduction order, and the
function composes under ``torch.func.grad`` / ``vmap`` (static, Python-int chunk
bounds keep it traceable). The frozen-``lm_head`` case needs no special
handling: under ``torch.func.grad`` only the parameters in the differentiated
set receive gradients.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from opaque.api.alignment.sft.loss._nll import nll_loss

__all__ = ["fused_linear_sft_loss"]

_SftLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def fused_linear_sft_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_fn: _SftLossFn = nll_loss,
    chunk_size: int = 1,
) -> torch.Tensor:
    """Chunked fused-linear SFT loss (memory-efficient, opt-in).

    Computes, for every example ``i``::

        loss_i = loss_fn(hidden_states_i @ lm_head_weight.T, labels_i)

    materialising at most ``chunk_size`` examples' ``(T, V)`` logits at once.
    Math-identical to the eager
    ``loss_fn(hidden_states @ lm_head_weight.T, labels)`` up to float reduction
    order (chunking partitions the non-interacting batch axis).

    Args:
        hidden_states: Last-layer hidden states, ``(B, T, H)``.
        lm_head_weight: LM-head projection weight, ``(V, H)``; logits are
            ``hidden_states @ lm_head_weight.T``.
        labels: Token-id targets, ``(B, T)``; ``-100`` positions are ignored by
            the loss (the causal shift is applied inside ``loss_fn``).
        loss_fn: An eager per-example SFT loss taking ``(logits, labels)`` and
            returning a per-example loss — :func:`nll_loss` (default) or
            :func:`dft_loss`.
        chunk_size: Number of examples whose logits are materialised at once;
            controls peak memory. Must be ``>= 1``. Defaults to ``1``.

    Returns:
        Per-example loss tensor of shape ``(B,)``.

    Raises:
        ValueError: If ``chunk_size < 1``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    weight_t = lm_head_weight.transpose(-2, -1)
    batch_size = hidden_states.shape[0]

    chunk_losses: list[torch.Tensor] = []
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        # Materialise only this chunk's logits: (chunk, T, V).
        logits = hidden_states[start:stop] @ weight_t
        chunk_losses.append(loss_fn(logits, labels[start:stop]))

    return torch.cat(chunk_losses, dim=0)
