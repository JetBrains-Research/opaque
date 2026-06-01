# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Chunked fused-linear *paired*-preference core (plan §7.10).

The purpose of this kernel is **peak memory**, not raw FLOPs. A naive paired
preference loss (DPO/ORPO/CPO/SimPO and the DPO variant family) materialises
the full ``(2B, T, V)`` logits tensor in one ``hidden @ lm_head.T`` before
reducing it to per-sequence log-probabilities. For a realistic
``B·T·V`` that tensor dominates activation memory.

Instead, :func:`fused_linear_preference` chunks over the *pairs* axis and
materialises only ``chunk_size`` pairs' logits at a time:

    peak logits memory  =  O(chunk_size · T · V)   (this kernel)
    instead of          =  O(B · T · V)            (materialise-all reference)

Per chunk it does:

1. project the chosen / rejected hidden states through ``lm_head_weight`` to
   get ``(chunk, T, V)`` logits for each side;
2. reduce to per-sequence completion log-probabilities via
   :func:`opaque.api.alignment.logprob._sequence.sequence_logp` (causal shift +
   completion mask + sum);
3. subtract the (precomputed) reference log-probabilities to form per-pair
   log-ratios;
4. evaluate ``per_pair_loss_fn(chosen_logratio, rejected_logratio)`` for the
   chunk.

The per-chunk loss tensors are concatenated back to ``(B,)``.

**Design constraints (plan §5, §7.10):**

- *Self-contained, pure PyTorch.* No Triton, no custom ``autograd.Function``.
  The whole thing is a composition of differentiable tensor ops, so autograd
  flows through every chunk and the chunk boundary is transparent to the
  gradient (chunking is a partition of the batch — ``torch.cat`` of
  per-chunk results has the same backward as the all-at-once form).
- *``torch.func``-composable.* No in-place mutation that breaks autograd, no
  data-dependent Python control flow on tensor *values*, no ``.item()`` on
  traced tensors. Chunk boundaries are derived from the *static* batch size,
  so the function traces cleanly under ``torch.func.grad`` / ``vmap``.
- *Autocast-aware entry.* The public dispatchers call :func:`follow_autocast`
  before delegating here; on CPU (autocast inactive) that is a no-op.

The GPU peak-memory win (``< B·T·V``) is validated by a Cadence preset; the
CPU test-suite validates numeric parity, ``chunk_size`` invariance, and
``torch.func.grad`` composability (plan §7.10 gate).

**Layout.** ``hidden_states`` carries the chosen and rejected pairs in two
separate batched arguments (``chosen_hidden`` / ``rejected_hidden``), each
``(B, T, H)``. The public :func:`opaque_fused_linear_dpo_loss` accepts the
concatenated ``(2B, T, H)`` form documented in the plan and splits it before
calling this core (see ``_dpo_dispatch``).
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch

from opaque.api.alignment.logprob._sequence import sequence_logp

__all__ = ["PerPairLossFn", "fused_linear_preference"]


class PerPairLossFn(Protocol):
    """Per-pair preference loss called on a chunk of log-ratios.

    Implementations receive the chosen / rejected log-ratios for the pairs in
    a chunk (each ``(chunk,)``) and return a ``(chunk,)`` per-pair loss. This
    is exactly the signature of the DPO variant functions in
    :mod:`opaque.api.alignment.loss.dpo` once their keyword arguments are bound
    (e.g. via :func:`functools.partial`).
    """

    def __call__(
        self,
        chosen_logratio: torch.Tensor,
        rejected_logratio: torch.Tensor,
    ) -> torch.Tensor: ...


def _chunk_bounds(batch_size: int, chunk_size: int) -> list[tuple[int, int]]:
    """Static ``(start, stop)`` chunk bounds over the pairs axis.

    Computed from the *static* ``batch_size`` (a Python int), so no
    tensor-valued control flow is introduced — the resulting Python loop
    traces cleanly under ``torch.func``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [
        (start, min(start + chunk_size, batch_size))
        for start in range(0, batch_size, chunk_size)
    ]


def fused_linear_preference(
    chosen_hidden: torch.Tensor,
    rejected_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    chosen_target_ids: torch.Tensor,
    rejected_target_ids: torch.Tensor,
    chosen_completion_mask: torch.Tensor,
    rejected_completion_mask: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    per_pair_loss_fn: PerPairLossFn | Callable[..., torch.Tensor],
    *,
    chunk_size: int = 1,
) -> torch.Tensor:
    """Chunked paired-preference loss core (plan §7.10).

    Computes, for every pair ``i``::

        chosen_logp_i   = sequence_logp(chosen_hidden_i @ Wᵀ, ...)
        rejected_logp_i = sequence_logp(rejected_hidden_i @ Wᵀ, ...)
        chosen_logratio_i   = chosen_logp_i   - ref_chosen_logp_i
        rejected_logratio_i = rejected_logp_i - ref_rejected_logp_i
        loss_i = per_pair_loss_fn(chosen_logratio_i, rejected_logratio_i)

    materialising at most ``chunk_size`` pairs' ``(T, V)`` logits per side at a
    time, giving peak logits memory ``O(chunk_size · T · V)`` instead of
    ``O(B · T · V)``.

    The result is **independent of ``chunk_size``** up to floating-point
    reduction order: chunking partitions the batch and the per-pair losses do
    not interact across pairs, so ``torch.cat`` of the per-chunk results equals
    the all-at-once form (and its gradient does too).

    Args:
        chosen_hidden: Chosen-response hidden states, ``(B, T, H)``.
        rejected_hidden: Rejected-response hidden states, ``(B, T, H)``.
        lm_head_weight: LM head projection, ``(V, H)``. Logits are
            ``hidden @ lm_head_weight.T``.
        chosen_target_ids: Token ids for the chosen side, ``(B, T)``.
        rejected_target_ids: Token ids for the rejected side, ``(B, T)``.
        chosen_completion_mask: Completion mask for the chosen side, ``(B, T)``
            (non-zero on completion tokens).
        rejected_completion_mask: Completion mask for the rejected side,
            ``(B, T)``.
        ref_chosen_logp: Reference chosen sequence log-probs, ``(B,)``.
        ref_rejected_logp: Reference rejected sequence log-probs, ``(B,)``.
        per_pair_loss_fn: Callable mapping ``(chosen_logratio, rejected_logratio)``
            (each ``(chunk,)``) to a ``(chunk,)`` per-pair loss. The DPO variant
            functions match this once their keyword args are bound.
        chunk_size: Number of pairs whose logits are materialised at once.
            Controls peak memory; must be ``>= 1``. Defaults to ``1``.

    Returns:
        Per-pair loss tensor of shape ``(B,)``.
    """
    batch_size = chosen_hidden.shape[0]
    weight_t = lm_head_weight.transpose(-2, -1)

    chunk_losses: list[torch.Tensor] = []
    for start, stop in _chunk_bounds(batch_size, chunk_size):
        # Materialise only this chunk's logits: (chunk, T, V) per side.
        chosen_logits = chosen_hidden[start:stop] @ weight_t
        rejected_logits = rejected_hidden[start:stop] @ weight_t

        chosen_logp = sequence_logp(
            chosen_logits,
            chosen_target_ids[start:stop],
            chosen_completion_mask[start:stop],
        )
        rejected_logp = sequence_logp(
            rejected_logits,
            rejected_target_ids[start:stop],
            rejected_completion_mask[start:stop],
        )

        chosen_logratio = chosen_logp - ref_chosen_logp[start:stop]
        rejected_logratio = rejected_logp - ref_rejected_logp[start:stop]

        chunk_losses.append(per_pair_loss_fn(chosen_logratio, rejected_logratio))

    return torch.cat(chunk_losses, dim=0)
