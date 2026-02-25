"""Membership scoring utilities for privacy auditing.

Computes per-example membership scores using the same ``torch.func.vmap``
and ``batch_argnums`` pattern as :func:`opaque.clipped_grad`. Higher scores
indicate higher likelihood of being a training member.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["score"]


def score(
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataset: Any,
    indices: np.ndarray | None = None,
    collate_fn: Callable | None = None,
    batch_unpack: Callable | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute membership scores as negative per-example loss.

    Higher score = lower loss = more likely a training member.
    Uses ``torch.func.vmap`` for per-example loss computation, following
    the same ``batch_argnums`` convention as :func:`~opaque.clipped_grad`.

    Args:
        loss_fn: Per-example loss function, same as used with
            :func:`~opaque.clipped_grad`. Must be vmap-compatible.
        *args: Non-batched arguments to ``loss_fn`` (e.g., model parameters).
            These correspond to the argument positions *not* listed in
            ``batch_argnums``.
        batch_argnums: Indices of ``loss_fn`` positional arguments that come
            from dataset batches. For example, ``(1,)`` means argument 1 is
            batched (HuggingFace pattern), ``(1, 2)`` means arguments 1 and 2
            are batched (PyTorch ``(x, y)`` pattern).
        dataset: Dataset to score. Must support ``len()`` and indexing.
        indices: If provided, only score these dataset indices. Typically
            ``experiment.canary_indices``.
        collate_fn: Collate function for the DataLoader (e.g.,
            ``DataCollatorForLanguageModeling``). If ``None``, uses
            PyTorch default collation.
        batch_unpack: Callable mapping a DataLoader batch to a tuple of
            tensors, one per ``batch_argnums`` position. For example::

                batch_unpack=lambda b: (b["input_ids"].to(device),)

            If ``None``, batches are unpacked as tuples/lists: ``batch[0]``,
            ``batch[1]``, etc.
        batch_size: Batch size for scoring. Default: 256.

    Returns:
        Array of membership scores, shape ``(n,)``. Scores are negated
        losses (higher = more likely member).

    Example (HuggingFace pattern)::

        scores = score(
            per_example_loss_fn,
            trainable_params,           # arg 0: not batched
            batch_argnums=(1,),         # arg 1: tokens from dataset
            dataset=train_dataset,
            indices=experiment.canary_indices,
            collate_fn=data_collator,
            batch_unpack=lambda b: (b["input_ids"].to(device),),
            batch_size=32,
        )

    Example (PyTorch ``(x, y)`` pattern)::

        scores = score(
            loss_fn,
            params,                     # arg 0: not batched
            batch_argnums=(1, 2),       # args 1, 2: x, y from dataset
            dataset=dataset,
            batch_size=256,
        )
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    if indices is not None:
        subset = Subset(dataset, np.asarray(indices).tolist())
    else:
        subset = dataset

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader = DataLoader(subset, **loader_kwargs)

    # Build in_dims for vmap: None for non-batch args, 0 for batch args
    n_args = len(args) + len(batch_argnums)
    in_dims = tuple(0 if i in batch_argnums else None for i in range(n_args))
    per_example_fn = torch.func.vmap(loss_fn, in_dims=in_dims)

    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            # Extract batched tensors from the DataLoader batch
            if batch_unpack is not None:
                batch_tensors = batch_unpack(batch)
            elif isinstance(batch, dict):
                # Dict batch (HF-style): use values in order of batch_argnums
                keys = list(batch.keys())
                batch_tensors = tuple(batch[keys[i]] for i in range(len(batch_argnums)))
            elif isinstance(batch, (list, tuple)):
                batch_tensors = tuple(batch[i] for i in range(len(batch_argnums)))
            else:
                batch_tensors = (batch,)

            # Merge non-batched args and batch tensors into full arg list
            full_args = _merge_args(args, batch_tensors, batch_argnums)

            losses = per_example_fn(*full_args)
            all_scores.append(-losses.detach().cpu().numpy())

    return np.concatenate(all_scores)


def _merge_args(
    args: tuple[Any, ...],
    batch_tensors: tuple[Any, ...],
    batch_argnums: tuple[int, ...],
) -> list[Any]:
    """Merge non-batched args and batch tensors into a single arg list.

    Non-batch args go into positions not in ``batch_argnums``.
    Batch tensors go into the positions specified by ``batch_argnums``.
    """
    n_total = len(args) + len(batch_argnums)
    result: list[Any] = [None] * n_total

    # Place batch tensors at their specified positions
    batch_argnums_sorted = sorted(batch_argnums)
    for pos, tensor in zip(batch_argnums_sorted, batch_tensors):
        result[pos] = tensor

    # Fill remaining positions with non-batch args
    arg_iter = iter(args)
    for i in range(n_total):
        if i not in batch_argnums:
            result[i] = next(arg_iter)

    return result
