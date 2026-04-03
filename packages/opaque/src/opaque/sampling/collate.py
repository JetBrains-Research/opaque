# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Collate utilities for Poisson sampling."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar

T = TypeVar("T")


def poisson_collate(collate_fn: Callable[..., T]) -> Callable[..., T | None]:
    """Wrap a collate function to handle empty batches from Poisson sampling.

    ``PoissonSampler`` can yield empty index lists, producing batches with
    zero examples.  Standard collators (e.g., HuggingFace
    ``DataCollatorForLanguageModeling``) crash on empty input because they
    unconditionally index ``examples[0]``.

    This wrapper returns ``None`` when the example list is empty.  The
    training loop should check for ``None`` and skip the step::

        collate = poisson_collate(my_collate_fn)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)

        for batch in loader:
            if batch is None:
                continue  # empty Poisson batch
            ...

    .. tip::

        If using HuggingFace ``DataCollatorForLanguageModeling``, the
        automatic compat patch (applied at ``import opaque``) handles
        empty inputs transparently by returning properly-shaped empty
        tensors — no ``None`` check needed.

    Args:
        collate_fn: Original collate function.

    Returns:
        Wrapped function that returns ``None`` for empty example lists.
    """

    @functools.wraps(collate_fn)
    def wrapper(examples):
        if not examples:
            return None
        return collate_fn(examples)

    return wrapper
