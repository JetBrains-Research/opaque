# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Public dispatcher for the chunked fused-linear KTO kernel (plan §7.10).

:func:`opaque_fused_linear_kto_loss` is the public entry point for the unpaired
(KTO) chunked-preference kernel. It is an autocast-aware thin wrapper over
:func:`opaque.api.alignment.kernel._fused_linear_unpaired.fused_linear_unpaired_preference`:
it casts floating-point inputs to the active autocast dtype (a no-op on CPU /
when autocast is inactive) and then delegates to the chunked core.

The kernel computes, per example, the completion sequence log-prob from the
hidden states + LM head, forms the per-example log-ratio against the reference
log-prob, and evaluates the KTO loss (arXiv:2402.01306 Eq. 8) with the scalar,
caller-computed, detached batch-mean ``kl`` term broadcast across the batch. It
chunks over the batch to keep peak logits memory at ``O(chunk_size · T · V)``
instead of ``O(B · T · V)``; see ``_fused_linear_unpaired`` for the full design
and the Tier-2 / KL-completion deferral notes.
"""

from __future__ import annotations

import torch

from ._fused_linear_unpaired import (
    _follow_autocast,
    fused_linear_unpaired_preference,
)

__all__ = ["opaque_fused_linear_kto_loss"]


def opaque_fused_linear_kto_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    completion_labels: torch.Tensor,
    label: torch.Tensor,
    ref_logp: torch.Tensor,
    *,
    beta: float = 0.1,
    kl: torch.Tensor,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    chunk_size: int = 1,
) -> torch.Tensor:
    """Chunked fused-linear KTO loss (plan §7.10), autocast-aware.

    Wraps :func:`fused_linear_unpaired_preference`. The completion forward is
    chunked over the batch so that at most ``chunk_size`` examples' ``(T, V)``
    logits are materialised at once, giving peak logits memory
    ``O(chunk_size · T · V)`` instead of ``O(B · T · V)``.

    Note: this kernel chunks only the **completion** forward. The KL_completion
    forward — which materialises its own ``(B, T, V)`` logits to compute the
    ``kl`` term — is NOT chunked here. ``kl`` is a scalar detached batch-mean
    (``z_0``) that the caller computes once over the microbatch, **outside** the
    per-example region, and passes in (the Tier-2 contract; plan §3.3, §8.1).
    Chunking the KL forward is the deferred ``opaque_selective_log_softmax`` item
    (plan §7.10); under the default KTO precompute path the KL logps are
    precomputed and that forward does not run at train time.

    Args:
        hidden_states: Completion hidden states, ``(B, T, H)``.
        lm_head_weight: LM head projection, ``(V, H)``. Logits are
            ``hidden @ lm_head_weight.T``.
        target_ids: Token ids for the completion, ``(B, T)``.
        completion_labels: Labels ``(B, T)`` with ``-100`` on prompt / ignored
            tokens. Non-``-100`` positions form the completion mask.
        label: Per-example boolean tensor ``(B,)``; ``True`` marks a desirable
            example.
        ref_logp: Reference completion sequence log-probs, ``(B,)``.
        beta: KTO temperature. Defaults to ``0.1``.
        kl: **Scalar, detached** batch-mean KL term ``z_0``, broadcast across the
            batch. Caller-computed outside the per-example region; see plan
            §3.3 / §8.1. Same constant for every chunk.
        desirable_weight: Weight on the desirable term. Defaults to ``1.0``.
        undesirable_weight: Weight on the undesirable term. Defaults to ``1.0``.
        chunk_size: Number of examples materialised at once; must be ``>= 1``.
            Defaults to ``1``.

    Returns:
        Per-example KTO loss tensor of shape ``(B,)``.
    """
    # Autocast-aware entry: cast floating CUDA/CPU tensors to the active
    # autocast dtype (no-op on CPU / when autocast is inactive). Integer tensors
    # (target_ids, completion_labels) and the boolean label pass through.
    hidden_states, lm_head_weight, ref_logp, kl = _follow_autocast(
        hidden_states, lm_head_weight, ref_logp, kl
    )
    return fused_linear_unpaired_preference(
        hidden_states,
        lm_head_weight,
        target_ids,
        completion_labels,
        label,
        ref_logp,
        beta=beta,
        kl=kl,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        chunk_size=chunk_size,
    )
