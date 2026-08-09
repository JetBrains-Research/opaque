"""Shared helpers for membership inference scoring functions."""

from __future__ import annotations

from typing import Any


def _validate_batch_argnums(batch_argnums: tuple[int, ...], n_non_batch: int) -> None:
    """Validate batch_argnums constraints."""
    if not batch_argnums:
        raise ValueError(f"batch_argnums must be non-empty, got {batch_argnums}")
    if any(a < 0 for a in batch_argnums):
        raise ValueError(f"batch_argnums must be non-negative, got {batch_argnums}")
    if len(set(batch_argnums)) != len(batch_argnums):
        raise ValueError(f"batch_argnums must be unique, got {batch_argnums}")
    if tuple(sorted(batch_argnums)) != batch_argnums:
        raise ValueError(f"batch_argnums must be sorted, got {batch_argnums}")
    n_total = n_non_batch + len(batch_argnums)
    if max(batch_argnums) >= n_total:
        raise ValueError(
            f"batch_argnums index {max(batch_argnums)} out of range for "
            f"{n_total} total arguments ({n_non_batch} non-batched + "
            f"{len(batch_argnums)} batched), got {batch_argnums}"
        )


def _merge_args(
    args: tuple[Any, ...],
    batch_tensors: tuple[Any, ...],
    batch_argnums: tuple[int, ...],
) -> list[Any]:
    """Merge non-batched args and batch tensors into a single arg list."""
    n_total = len(args) + len(batch_argnums)
    result: list[Any] = [None] * n_total

    for pos, tensor in zip(batch_argnums, batch_tensors, strict=False):
        result[pos] = tensor

    arg_iter = iter(args)
    for i in range(n_total):
        if i not in batch_argnums:
            result[i] = next(arg_iter)

    return result


def _extract_batch_tensors(
    batch: Any,
    batch_argnums: tuple[int, ...],
) -> tuple[Any, ...]:
    """Extract tensors from a dataloader batch.

    Returns a tuple of tensors matching the length of ``batch_argnums``.
    """
    if isinstance(batch, (list, tuple)):
        return tuple(batch[i] for i in range(len(batch_argnums)))
    return (batch,)


def _check_unshuffled(dataloader: Any) -> None:
    """Raise when a torch ``DataLoader`` would shuffle the scoring order.

    Membership scores are paired positionally with the coin-flip labels, so
    a shuffled loader silently attaches scores to the wrong labels and the
    audit reports no leakage that was never measured.  Detects the torch
    shuffling samplers on both the ``sampler`` and ``batch_sampler.sampler``
    seats; arbitrary iterables and custom samplers rely on the explicit
    ``order`` token on :func:`~opaque.auditing.one_run` (see
    :func:`scoring_order`).
    """
    import torch.utils.data as tud

    shuffling = (tud.RandomSampler, tud.SubsetRandomSampler)
    sampler = getattr(dataloader, "sampler", None)
    inner = getattr(getattr(dataloader, "batch_sampler", None), "sampler", None)
    if isinstance(sampler, shuffling) or isinstance(inner, shuffling):
        raise ValueError(
            "dataloader is shuffled (RandomSampler/SubsetRandomSampler); "
            "membership scores must preserve canary order — construct the "
            "DataLoader with shuffle=False over the canary Subset"
        )


def scoring_order(dataloader: Any) -> Any:
    """Return the dataset indices ``dataloader`` will score, in order.

    Produces the ``order`` token for :func:`~opaque.auditing.one_run` from
    the loader itself, so the score-to-label pairing is verified against
    what was actually iterated rather than against user bookkeeping.
    Supports the canonical auditing setup: a strictly sequential
    ``DataLoader`` over ``torch.utils.data.Subset`` (e.g.
    ``Subset(dataset, coin_flip.canary_indices)``).

    Raises:
        ValueError: If the loader shuffles, uses a non-sequential sampler
            (iteration order underivable), or does not wrap a ``Subset``
            (no index provenance).
    """
    import numpy as np
    import torch.utils.data as tud

    _check_unshuffled(dataloader)
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and not isinstance(sampler, tud.SequentialSampler):
        raise ValueError(
            "scoring_order requires a strictly sequential DataLoader (got "
            f"sampler {type(sampler).__name__}); the iteration order of a "
            "custom sampler cannot be derived."
        )
    indices = getattr(getattr(dataloader, "dataset", None), "indices", None)
    if indices is None:
        raise ValueError(
            "scoring_order requires a DataLoader over "
            "torch.utils.data.Subset (e.g. Subset(dataset, "
            "coin_flip.canary_indices)) so the scored dataset indices are "
            "recoverable."
        )
    return np.asarray(indices)
