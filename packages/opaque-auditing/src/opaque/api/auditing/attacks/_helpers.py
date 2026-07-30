"""Shared helpers for membership inference scoring functions."""

from __future__ import annotations

from typing import Any


def _validate_batch_argnums(batch_argnums: tuple[int, ...], n_non_batch: int) -> None:
    """Validate batch_argnums constraints."""
    if not batch_argnums:
        raise ValueError("batch_argnums must be non-empty")
    if any(a < 0 for a in batch_argnums):
        raise ValueError(f"batch_argnums must be non-negative, got {batch_argnums}")
    if len(set(batch_argnums)) != len(batch_argnums):
        raise ValueError(f"batch_argnums must be unique, got {batch_argnums}")
    n_total = n_non_batch + len(batch_argnums)
    if max(batch_argnums) >= n_total:
        raise ValueError(
            f"batch_argnums index {max(batch_argnums)} out of range for "
            f"{n_total} total arguments ({n_non_batch} non-batched + "
            f"{len(batch_argnums)} batched)"
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
