# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Collate utilities for Poisson sampling."""

from __future__ import annotations

import copy
import functools
from collections.abc import Mapping
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


def _empty_like(template: T) -> T:
    """Create an empty-batch replica of a collate output.

    Recursively walks the structure and replaces every tensor with a
    zero-batch-dim copy (``tensor[:0]``), preserving dtype, device,
    and all non-batch dimensions.  Non-tensor leaves are passed through
    unchanged.  Supports dicts, ``Mapping`` subclasses (e.g. HuggingFace
    ``BatchEncoding``), lists, and tuples.

    Args:
        template: A representative collate output (first non-empty batch).

    Returns:
        Structure identical to *template* but with every tensor having
        ``shape[0] == 0``.
    """
    if isinstance(template, torch.Tensor):
        return template[:0]
    if isinstance(template, Mapping):
        return {k: _empty_like(v) for k, v in template.items()}
    if isinstance(template, tuple):
        return tuple(_empty_like(v) for v in template)
    if isinstance(template, list):
        return [_empty_like(v) for v in template]
    return template


def poisson_collate(collate_fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap a collate function to handle empty batches from Poisson sampling.

    ``PoissonSampler`` can yield empty index lists, producing batches with
    zero examples.  Standard collators (e.g., HuggingFace
    ``DataCollatorForLanguageModeling``) crash on empty input because they
    unconditionally index ``examples[0]``.

    This wrapper learns the output structure from the first non-empty batch
    and returns a properly-shaped empty batch (batch dim = 0) when the
    example list is empty::

        collate = poisson_collate(my_collate_fn)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)

        for batch in loader:
            input_ids = batch["input_ids"]   # shape (0, seq_len) when empty
            if len(input_ids) == 0:
                continue
            ...

    .. note::

        If the very first batch is empty (before any structure has been
        learned), the call falls through to *collate_fn* which will
        typically crash — this is the correct signal that the sample rate
        is misconfigured.

    Args:
        collate_fn: Original collate function.

    Returns:
        Wrapped function that returns empty-batch-dim outputs for empty
        example lists.
    """
    template: list[T | None] = [None]  # mutable cell for closure

    @functools.wraps(collate_fn)
    def wrapper(examples):
        if not examples:
            if template[0] is None:
                return collate_fn(examples)
            return _empty_like(template[0])

        result = collate_fn(examples)

        if template[0] is None:
            template[0] = copy.deepcopy(result)

        return result

    return wrapper
