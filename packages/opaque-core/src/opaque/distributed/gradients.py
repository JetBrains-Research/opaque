"""Gradient / pytree reduction helpers for distributed DP training.

Provides ``reduce_pytree(_)`` for generic per-leaf all-reduce and
``sum_gradients(_)`` as a thin alias scoped to clipped gradients.
"""

from __future__ import annotations

from typing import Any

import torch

from opaque.core.pytree import tree_map

from .collectives import all_reduce_, is_distributed


def reduce_pytree_(pytree: Any, op: str = "sum") -> None:
    """All-reduce every tensor leaf in ``pytree`` in place (no-op if not distributed)."""
    if not is_distributed():
        return

    def _reduce(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            all_reduce_(leaf, op=op)
        return leaf

    tree_map(_reduce, pytree)


def reduce_pytree(pytree: Any, op: str = "sum") -> Any:
    """Return a pytree with each tensor leaf reduced; input unchanged."""

    def _clone(leaf: Any) -> Any:
        return leaf.clone() if isinstance(leaf, torch.Tensor) else leaf

    reduced = tree_map(_clone, pytree)
    reduce_pytree_(reduced, op=op)
    return reduced


def sum_gradients_(gradients: Any) -> None:
    """DP-specific alias for ``reduce_pytree_(op="sum")``."""
    reduce_pytree_(gradients, op="sum")


def sum_gradients(gradients: Any) -> Any:
    """DP-specific alias for ``reduce_pytree(op="sum")``."""
    return reduce_pytree(gradients, op="sum")


__all__ = [
    "reduce_pytree",
    "reduce_pytree_",
    "sum_gradients",
    "sum_gradients_",
]
