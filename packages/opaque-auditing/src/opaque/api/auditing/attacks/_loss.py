"""Loss-based membership scoring for privacy auditing.

Computes per-example membership scores using the same ``torch.func.vmap``
and ``batch_argnums`` pattern as :func:`opaque.clipped_grad`. Higher scores
indicate higher likelihood of being a training member.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from opaque.api.auditing.attacks._helpers import (
    _check_unshuffled,
    _extract_batch_tensors,
    _merge_args,
    _validate_batch_argnums,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["loss_scores"]


def loss_scores(
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataloader: Any,
    reference_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Compute membership scores as negative per-example loss.

    Higher score = lower loss = more likely a training member.
    Uses ``torch.func.vmap`` for per-example loss computation, following
    the same ``batch_argnums`` convention as :func:`~opaque.clipped_grad`.

    The ``dataloader`` must yield batches compatible with ``loss_fn``.
    Each batch should be a tensor (single ``batch_argnums``) or a tuple
    of tensors (multiple ``batch_argnums``). Use a custom ``collate_fn``
    on the DataLoader to handle dict-style batches (e.g., HuggingFace).

    When ``reference_scores`` are provided, the returned scores are
    ``current - reference``, which equals ``loss(w0) - loss(wt)``
    (the loss reduction from initial to current model). This matches
    Algorithm 3 of Steinke et al. (2023).

    Args:
        loss_fn: Per-example loss function, same as used with
            :func:`~opaque.clipped_grad`. Must be vmap-compatible.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
        batch_argnums: Indices of ``loss_fn`` positional arguments that come
            from dataset batches. Must be sorted, unique, non-negative.
        dataloader: An iterable of batches (typically a ``DataLoader``).
            Must yield tensors or tuples of tensors matching
            ``batch_argnums``.
        reference_scores: Baseline scores from an untrained model, shape
            ``(n,)``. When provided, returned scores are the loss reduction
            ``scores - reference_scores``. Typically obtained by calling
            ``loss_scores`` on the untrained model before training.

    Returns:
        Array of membership scores, shape ``(n,)``. Scores are negated
        losses (higher = more likely member), optionally adjusted by
        reference scores.

    Raises:
        ValueError: If ``dataloader`` shuffles (RandomSampler-family
            sampler) — scores are paired positionally with the coin-flip
            labels and must preserve canary order.

    Example (HuggingFace pattern)::

        from torch.utils.data import DataLoader, Subset

        def canary_collate(examples):
            batch = data_collator(examples)
            return (batch["input_ids"].to(device),)

        canary_loader = DataLoader(
            Subset(dataset, cf.canary_indices),
            batch_size=32, collate_fn=canary_collate,
        )
        ref = auditing.loss_scores(
            loss_fn, initial_params,
            batch_argnums=(1,), dataloader=canary_loader,
        )
        scores = auditing.loss_scores(
            loss_fn, trained_params,
            batch_argnums=(1,), dataloader=canary_loader,
            reference_scores=ref,
        )

    Example (PyTorch ``(x, y)`` pattern)::

        loader = DataLoader(dataset, batch_size=256)
        scores = auditing.loss_scores(
            loss_fn, params,
            batch_argnums=(1, 2), dataloader=loader,
        )
    """
    import torch

    _validate_batch_argnums(batch_argnums, len(args))

    # Build in_dims for vmap: None for non-batch args, 0 for batch args
    n_args = len(args) + len(batch_argnums)
    in_dims = tuple(0 if i in batch_argnums else None for i in range(n_args))
    per_example_fn = torch.func.vmap(loss_fn, in_dims=in_dims)

    _check_unshuffled(dataloader)
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_tensors = _extract_batch_tensors(batch, batch_argnums)

            full_args = _merge_args(args, batch_tensors, batch_argnums)

            losses = per_example_fn(*full_args)
            all_scores.append(-losses.detach().cpu().numpy())

    scores = np.concatenate(all_scores)

    if reference_scores is not None:
        reference_scores = np.asarray(reference_scores)
        if reference_scores.shape != scores.shape:
            raise ValueError(
                f"reference_scores shape {reference_scores.shape} does not match "
                f"scores shape {scores.shape}"
            )
        scores = scores - reference_scores

    return scores
