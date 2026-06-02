# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Public DPO dispatcher for the chunked fused-linear preference kernel
.

:func:`fused_linear_dpo_loss` is the user-facing entry point. It takes an
eager per-pair DPO loss callable (``per_pair_loss_fn``, e.g.
:func:`opaque.api.alignment.dpo.loss.sigmoid_loss`), binds its keyword arguments
(``beta`` and, where applicable, ``label_smoothing``), and delegates the actual
work to the reusable chunked core
:func:`opaque.api.alignment.dpo.kernel._fused_linear_preference.fused_linear_preference`.

**Why this exists (memory).** A naive DPO loss materialises the full
``(2B, T, V)`` logits tensor (``hidden @ lm_head.T``) before reducing it to
per-sequence log-probabilities. For realistic ``B·T·V`` that tensor dominates
activation memory. The chunked core instead materialises only ``chunk_size``
pairs' logits at a time, giving peak logits memory ``O(chunk_size · T · V)``
instead of ``O(B · T · V)``. Chunking is a pure partition of the pairs axis, so
it does **not** change the numeric result (up to floating-point reduction
order) nor the gradient — see the parity / invariance tests.

**Validation split.** The GPU peak-memory win (``< (B,T,V)``) is
validated by a Cadence preset; the CPU test-suite validates numeric parity
against an all-at-once eager reference, ``chunk_size`` invariance, and
``torch.func.grad`` composability.

**Layout.** ``hidden_states`` / ``target_ids`` / ``completion_mask`` carry the
chosen and rejected sequences **concatenated along the batch axis**: rows
``[0:B]`` are the *chosen* responses and rows ``[B:2B]`` are the *rejected*
responses, so the leading dimension is ``2B`` for ``B`` preference pairs. This
matches the layout documented in plan §7.10 (``(B, T, H) chosen, (B, T, H)
rejected concatenated along batch dim``). The reference log-probabilities are
supplied already split as ``ref_chosen_logp`` / ``ref_rejected_logp``, each
``(B,)``.

**Composability / autocast.** The function is pure-PyTorch (no custom
``autograd.Function``, no Triton), so it composes under ``torch.func.grad`` /
``vmap``. :func:`follow_autocast` is called on entry so that, inside a
``torch.autocast(...)`` region, the matmul runs in the active autocast dtype
end-to-end; on CPU (autocast inactive) it is a no-op.
"""

from __future__ import annotations

import functools
import inspect

import torch

from opaque.api.alignment.dpo.kernel._fused_linear_preference import (
    PerPairLossFn,
    fused_linear_preference,
)
from opaque.api.alignment.dpo.kernel._utils import follow_autocast
from opaque.api.alignment.dpo.loss._sigmoid import sigmoid_loss

__all__ = ["fused_linear_dpo_loss"]


def _bind_per_pair_loss_fn(
    loss_fn: PerPairLossFn,
    *,
    beta: float,
    label_smoothing: float,
) -> PerPairLossFn:
    """Bind a DPO variant's scalar keyword arguments.

    ``loss_fn`` is an eager per-pair DPO loss (e.g. :func:`sigmoid_loss`); this
    returns a callable ``(chosen_logratio, rejected_logratio) -> Tensor`` (the
    :class:`~opaque.api.alignment.dpo.kernel._fused_linear_preference.PerPairLossFn`
    contract). ``beta`` is bound for every variant; ``label_smoothing`` is bound
    only for variants that accept it (e.g. :func:`sigmoid_loss`), so variants
    without it are not passed an unexpected keyword.
    """
    params = inspect.signature(loss_fn).parameters
    bound_kwargs: dict[str, float] = {"beta": beta}
    if "label_smoothing" in params:
        bound_kwargs["label_smoothing"] = label_smoothing
    return functools.partial(loss_fn, **bound_kwargs)


def _lce_fast_path_available(hidden: torch.Tensor) -> bool:
    """True when the fused linear-CE fast path can run (CUDA + half + ``[patches]``).

    Mirrors the per-call gate the opaque-patches CE component uses: the Triton
    kernel needs CUDA and half-precision activations, and ``opaque-patches`` is
    an optional dependency (``opaque-alignment[patches]``). Otherwise the caller
    falls back to the self-contained pure-PyTorch chunked path.
    """
    if not hidden.is_cuda or hidden.dtype not in (torch.float16, torch.bfloat16):
        return False
    try:
        import opaque.api.patches.kernels.linear_cross_entropy  # noqa: F401
    except Exception:
        return False
    return True


def _sequence_logp_via_lce(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence completion log-prob via the fused linear cross-entropy kernel.

    ``sequence_logp = −Σ_completion CE``, so the patches fused linear-CE kernel
    *is* the preference-logp kernel: encode the completion span as
    ``ignore_index`` on the labels, evaluate the unreduced fused linear-CE per
    sequence (``vmap`` over the batch axis), and negate. The ``(N, T, V)`` logits
    are never materialised — the kernel recomputes the LSE in its backward.
    """
    from opaque.api.patches.kernels.linear_cross_entropy import (
        Opaque_LinearCrossEntropyLoss,
    )

    # Completion tokens keep their id; everything else → ignore_index (dropped),
    # exactly reproducing ``sequence_logp``'s completion mask.
    masked_labels = torch.where(
        completion_mask.to(torch.bool),
        target_ids,
        torch.full_like(target_ids, -100),
    )

    def _neg_logp(seq_hidden: torch.Tensor, seq_labels: torch.Tensor) -> torch.Tensor:
        # Unreduced Σ CE over one sequence's completion tokens = −logp.
        return Opaque_LinearCrossEntropyLoss.apply(
            seq_hidden, lm_head_weight, seq_labels, -100, 0, 0.0
        )

    return -torch.vmap(_neg_logp)(hidden, masked_labels)


def fused_linear_dpo_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    *,
    beta: float = 0.1,
    per_pair_loss_fn: PerPairLossFn = sigmoid_loss,
    chunk_size: int = 1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Fused-linear DPO preference loss (auto-selecting).

    Computes the per-pair DPO loss without materialising the full
    ``(2B, T, V)`` logits tensor. On CUDA + half precision with
    ``opaque-alignment[patches]`` installed it dispatches to the fused
    linear-CE Triton kernel (recompute-in-backward — the real training-memory
    win); otherwise it runs the self-contained pure-PyTorch chunked path below.
    Both paths are numerically equivalent and return the same ``(B,)`` loss. For each chunk of ``chunk_size`` pairs the
    kernel projects the chosen / rejected hidden states through
    ``lm_head_weight``, reduces to per-sequence completion log-probabilities,
    subtracts the reference log-probabilities to form log-ratios, and evaluates
    the selected DPO variant. Per-chunk results are concatenated to ``(B,)``.

    Peak logits memory is ``O(chunk_size · T · V)`` rather than the
    all-at-once ``O(B · T · V)``. The result and its gradient are independent of
    ``chunk_size`` (up to floating-point reduction order), because chunking is a
    pure partition of the (non-interacting) pairs axis.

    Args:
        hidden_states: Concatenated hidden states ``(2B, T, H)``. Rows
            ``[0:B]`` are the chosen responses; rows ``[B:2B]`` are the rejected
            responses (see module docstring for the layout rationale).
        lm_head_weight: LM-head projection weight ``(V, H)``. Logits are
            ``hidden @ lm_head_weight.T``.
        target_ids: Concatenated token ids ``(2B, T)`` in the same chosen /
            rejected row order as ``hidden_states``.
        completion_mask: Concatenated completion mask ``(2B, T)`` (non-zero on
            completion tokens), same row order.
        ref_chosen_logp: Reference chosen sequence log-probabilities ``(B,)``.
        ref_rejected_logp: Reference rejected sequence log-probabilities
            ``(B,)``.
        beta: DPO temperature (reference-deviation strength). Defaults to
            ``0.1``.
        per_pair_loss_fn: An eager per-pair DPO loss taking
            ``(chosen_logratio, rejected_logratio, *, beta[, label_smoothing])``
            and returning a per-pair ``(B,)`` tensor — e.g. :func:`sigmoid_loss`
            (default) or :func:`hinge_loss` from
            :mod:`opaque.api.alignment.dpo.loss`. ``beta`` (and
            ``label_smoothing`` where accepted) is bound by the dispatcher.
        chunk_size: Number of pairs whose logits are materialised at once.
            Controls peak memory; must be ``>= 1``. Defaults to ``1``.
        label_smoothing: Label-smoothing coefficient passed through to variants
            that accept it (e.g. ``sigmoid``); ignored by variants that do not.
            Defaults to ``0.0``.

    Returns:
        Per-pair DPO loss tensor of shape ``(B,)``.

    Raises:
        ValueError: If the leading (batch) dimension of ``hidden_states`` /
            ``target_ids`` / ``completion_mask`` is not even (it must be ``2B``).
    """
    # Autocast-aware entry: cast the floating-point activations / weight to the
    # active autocast dtype if inside an autocast region; no-op on CPU.
    hidden_states, lm_head_weight, ref_chosen_logp, ref_rejected_logp = follow_autocast(
        hidden_states,
        lm_head_weight,
        ref_chosen_logp,
        ref_rejected_logp,
    )

    two_b = hidden_states.shape[0]
    if two_b % 2 != 0:
        raise ValueError(
            "hidden_states leading dim must be 2B (chosen rows [0:B] then "
            f"rejected rows [B:2B]); got an odd batch dim {two_b}."
        )
    batch = two_b // 2

    # Split the concatenated (2B, ...) tensors into chosen / rejected halves.
    chosen_hidden = hidden_states[:batch]
    rejected_hidden = hidden_states[batch:]
    chosen_target_ids = target_ids[:batch]
    rejected_target_ids = target_ids[batch:]
    chosen_completion_mask = completion_mask[:batch]
    rejected_completion_mask = completion_mask[batch:]

    bound_loss_fn = _bind_per_pair_loss_fn(
        per_pair_loss_fn,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    # Fast path (auto-selected): the patches fused linear-CE kernel computes the
    # per-sequence logp without materialising ``(2B, T, V)`` logits and
    # recomputes the LSE in its backward (the real training-memory win). Used
    # only on CUDA + half precision with ``[patches]`` installed; otherwise the
    # self-contained pure-PyTorch chunked path below runs (so CPU CI is green).
    if _lce_fast_path_available(hidden_states):
        seq_logp = _sequence_logp_via_lce(
            hidden_states, lm_head_weight, target_ids, completion_mask
        )  # (2B,)
        chosen_logratio = seq_logp[:batch] - ref_chosen_logp
        rejected_logratio = seq_logp[batch:] - ref_rejected_logp
        return bound_loss_fn(chosen_logratio, rejected_logratio)

    return fused_linear_preference(
        chosen_hidden,
        rejected_hidden,
        lm_head_weight,
        chosen_target_ids,
        rejected_target_ids,
        chosen_completion_mask,
        rejected_completion_mask,
        ref_chosen_logp,
        ref_rejected_logp,
        bound_loss_fn,
        chunk_size=chunk_size,
    )
