# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Chunked fused-linear *unpaired*-preference core (KTO) — plan §7.10.

This is the unpaired sibling of
:mod:`opaque.api.alignment.kernel._fused_linear_preference`. Where the paired
core chunks over ``(chosen, rejected)`` pairs for DPO/ORPO/CPO/SimPO, this core
chunks over the **per-example completion** batch for KTO (arXiv:2402.01306),
which carries a single completion plus a boolean ``label`` per example.

The purpose is **peak memory**, not raw FLOPs. A naive unpaired preference loss
materialises the full ``(B, T, V)`` completion logits in one
``hidden @ lm_head.T`` before reducing it to per-sequence log-probabilities. For
a realistic ``B·T·V`` that tensor dominates activation memory.

Instead, :func:`fused_linear_unpaired_preference` chunks over the *batch* axis
and materialises only ``chunk_size`` examples' logits at a time:

    peak logits memory  =  O(chunk_size · T · V)   (this kernel)
    instead of          =  O(B · T · V)            (materialise-all reference)

Per chunk it does:

1. project the completion hidden states through ``lm_head_weight`` to get
   ``(chunk, T, V)`` logits;
2. reduce to per-sequence completion log-probabilities via
   :func:`opaque.api.alignment.logprob._sequence.sequence_logp` (causal shift +
   completion mask + sum) — the completion mask is derived from
   ``completion_labels`` (non-``-100`` positions are completion tokens);
3. subtract the (precomputed) reference log-probability to form a per-example
   log-ratio, and split it into the chosen / rejected log-ratios that
   :func:`opaque.api.alignment.kto.loss._kto.kto_loss` expects, using the
   per-example boolean ``label`` (``chosen_lr = logratio · label``,
   ``rejected_lr = logratio · ~label``);
4. evaluate :func:`kto_loss` for the chunk, broadcasting the scalar detached
   batch-mean ``kl`` aggregate into every example.

The per-chunk loss tensors are concatenated back to ``(B,)``.

**The KL_completion forward is NOT chunked here.** KTO's ``kl`` term is the
detached batch-mean KL ``z_0`` (plan §3.3 Tier 2; arXiv:2402.01306 Eq. 8). It is
computed by the *caller* — once, over the active microbatch, **outside** the
per-example region — and passed in as a scalar. Its own forward materialises a
separate ``(B, T, V)`` logits tensor, which this kernel does **not** chunk: that
is the Tier-2 deferred ``opaque_selective_log_softmax`` item (plan §7.10). Under
the default KTO precompute path the KL logps are precomputed, so that forward
does not happen at train time at all and the deferral is free.

**Design constraints (plan §5, §7.10):**

- *Self-contained, pure PyTorch.* No Triton, no custom ``autograd.Function``.
  The whole thing is a composition of differentiable tensor ops, so autograd
  flows through every chunk and the chunk boundary is transparent to the
  gradient (chunking is a partition of the batch — ``torch.cat`` of per-chunk
  results has the same backward as the all-at-once form).
- *``torch.func``-composable.* No in-place mutation that breaks autograd, no
  data-dependent Python control flow on tensor *values*, no ``.item()`` on
  traced tensors. Chunk boundaries are derived from the *static* batch size, so
  the function traces cleanly under ``torch.func.grad`` / ``vmap``.
- *Autocast-aware entry.* The public dispatcher (``_kto_dispatch``) calls
  :func:`_follow_autocast` before delegating here; on CPU (autocast inactive)
  that is a no-op.

:func:`_follow_autocast` is a small **private** copy of the autocast shim that
the public dispatchers normally pull from
``opaque.api.alignment.kernel._utils``. It is inlined here so this work-unit does
not depend on a sibling-owned file; see plan §7.10 for the duplication rationale.
The GPU peak-memory win (``< B·T·V``) is validated by a Cadence preset; the CPU
test-suite validates numeric parity, ``chunk_size`` invariance, and
``torch.func.grad`` composability (plan §7.10 gate).
"""

from __future__ import annotations

import torch

from opaque.api.alignment.logprob._sequence import sequence_logp
from opaque.api.alignment.kto.loss._kto import kto_loss

__all__ = ["fused_linear_unpaired_preference"]


def _follow_autocast(*tensors: object) -> tuple[object, ...]:
    """Cast floating-point tensors to the active autocast dtype, if any.

    Private, self-contained copy of the autocast shim normally provided by
    :mod:`opaque.api.alignment.kernel._utils` (which a sibling work-unit owns).
    Inlining it here keeps this work-unit free of a cross-unit import; see plan
    §7.10 for the intentional-duplication rationale.

    The active device type is read from the first floating-point CUDA/CPU tensor;
    the cast is a no-op when autocast is inactive (the common CPU-test path) or
    when a tensor already has the target dtype. Non-tensor arguments, integer
    tensors, and ``None`` are passed through unchanged.

    Args:
        *tensors: Mixed tensors / non-tensors in call order.

    Returns:
        Tuple of the same length and order, with floating-point tensors cast to
        the active autocast dtype where applicable.
    """
    device_type: str | None = None
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            device_type = t.device.type
            break

    if device_type is None or not torch.is_autocast_enabled(device_type):
        return tensors

    target = torch.get_autocast_dtype(device_type)
    out: list[object] = []
    for t in tensors:
        if (
            isinstance(t, torch.Tensor)
            and t.is_floating_point()
            and t.device.type == device_type
            and t.dtype != target
        ):
            out.append(t.to(target))
        else:
            out.append(t)
    return tuple(out)


def _chunk_bounds(batch_size: int, chunk_size: int) -> list[tuple[int, int]]:
    """Static ``(start, stop)`` chunk bounds over the batch axis.

    Computed from the *static* ``batch_size`` (a Python int), so no
    tensor-valued control flow is introduced — the resulting Python loop traces
    cleanly under ``torch.func``.

    Args:
        batch_size: Number of examples in the batch.
        chunk_size: Number of examples per chunk; must be ``>= 1``.

    Returns:
        List of ``(start, stop)`` half-open index ranges covering ``[0, B)``.

    Raises:
        ValueError: If ``chunk_size < 1``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [
        (start, min(start + chunk_size, batch_size))
        for start in range(0, batch_size, chunk_size)
    ]


def fused_linear_unpaired_preference(
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
    """Chunked unpaired-preference (KTO) loss core (plan §7.10).

    Computes, for every example ``i``::

        logp_i      = sequence_logp(hidden_i @ Wᵀ, target_ids_i, mask_i)
        logratio_i  = logp_i - ref_logp_i
        chosen_lr_i   = logratio_i ·  label_i
        rejected_lr_i = logratio_i · ~label_i
        loss_i = kto_loss(chosen_lr_i, rejected_lr_i, label_i, beta=…, kl=…)

    materialising at most ``chunk_size`` examples' ``(T, V)`` logits at a time,
    giving peak logits memory ``O(chunk_size · T · V)`` instead of
    ``O(B · T · V)``.

    The completion mask is derived from ``completion_labels``: positions equal to
    ``-100`` (prompt / ignored tokens) are excluded; the rest are completion
    tokens. The gather index for the per-token log-prob is ``target_ids`` (which
    is well-defined at every position, unlike the ``-100``-bearing labels).

    The result is **independent of ``chunk_size``** up to floating-point
    reduction order: chunking partitions the batch and the per-example losses do
    not interact across examples, so ``torch.cat`` of the per-chunk results
    equals the all-at-once form (and its gradient does too). The scalar ``kl``
    aggregate is the **same constant** for every chunk — it is not recomputed per
    chunk — preserving both chunk invariance and the Tier-2 detach contract.

    Args:
        hidden_states: Completion hidden states, ``(B, T, H)``.
        lm_head_weight: LM head projection, ``(V, H)``. Logits are
            ``hidden @ lm_head_weight.T``.
        target_ids: Token ids for the completion, ``(B, T)``. Used as the gather
            index for per-token log-probs.
        completion_labels: Labels ``(B, T)`` with ``-100`` on prompt / ignored
            tokens. Non-``-100`` positions form the completion mask.
        label: Per-example boolean tensor ``(B,)``; ``True`` marks a desirable
            (chosen) example, ``False`` an undesirable (rejected) one.
        ref_logp: Reference completion sequence log-probs, ``(B,)``.
        beta: KTO temperature (reference-deviation strength). Defaults to
            ``0.1``.
        kl: **Scalar, detached** batch-mean KL term ``z_0`` (Tier 2; plan §3.3,
            §8.1). Computed once by the caller over the active microbatch —
            **outside** any per-example region — and broadcast into every
            example unchanged. The *same* scalar enters every chunk, so chunking
            never perturbs the aggregate. It MUST be ``.detach()``-ed before
            being passed; :func:`kto_loss` does not back-propagate through it
            regardless.
        desirable_weight: Weight on the desirable (``label=True``) term.
            Defaults to ``1.0``.
        undesirable_weight: Weight on the undesirable (``label=False``) term.
            Defaults to ``1.0``.
        chunk_size: Number of examples whose logits are materialised at once.
            Controls peak memory; must be ``>= 1``. Defaults to ``1``.

    Returns:
        Per-example KTO loss tensor of shape ``(B,)``.
    """
    batch_size = hidden_states.shape[0]
    weight_t = lm_head_weight.transpose(-2, -1)
    label_bool = label.bool()

    chunk_losses: list[torch.Tensor] = []
    for start, stop in _chunk_bounds(batch_size, chunk_size):
        # Materialise only this chunk's logits: (chunk, T, V).
        logits = hidden_states[start:stop] @ weight_t

        # Completion mask = positions whose label is not the ignore index.
        chunk_labels = completion_labels[start:stop]
        completion_mask = (chunk_labels != -100).to(logits.dtype)

        logp = sequence_logp(
            logits,
            target_ids[start:stop],
            completion_mask,
        )

        logratio = logp - ref_logp[start:stop]
        chunk_label = label_bool[start:stop]
        # Split the single per-example log-ratio into the chosen / rejected
        # slots kto_loss consumes. The unused side is masked to zero; kto_loss
        # selects the active side per example via the same boolean label.
        label_f = chunk_label.to(logratio.dtype)
        chosen_logratio = logratio * label_f
        rejected_logratio = logratio * (1.0 - label_f)

        chunk_losses.append(
            kto_loss(
                chosen_logratio,
                rejected_logratio,
                chunk_label,
                beta=beta,
                kl=kl,
                desirable_weight=desirable_weight,
                undesirable_weight=undesirable_weight,
            )
        )

    return torch.cat(chunk_losses, dim=0)
