"""Thin wrappers around ``torch.distributed`` collectives.

Provides ``is_distributed``, ``get_rank``, ``get_world_size``, ``all_reduce``
and its in-place variant, and ``barrier``. All are safe to call outside a
process group; non-distributed contexts fall through to sensible defaults
(rank 0, world_size 1, no-op barrier).
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Return True if ``torch.distributed`` is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current rank (0 if not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return the world size (1 if not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


_OP_MAP: dict[str, object] = {}


def _resolve_op(op: str):
    """Map a string to a ``torch.distributed.ReduceOp``. Populated lazily."""
    if not _OP_MAP:
        _OP_MAP.update(
            {
                "sum": dist.ReduceOp.SUM,
                "mean": dist.ReduceOp.AVG,
                "max": dist.ReduceOp.MAX,
                "min": dist.ReduceOp.MIN,
                "product": dist.ReduceOp.PRODUCT,
            }
        )
    if op not in _OP_MAP:
        raise ValueError(
            f"Invalid reduction operation: {op}. Must be one of: {list(_OP_MAP.keys())}"
        )
    return _OP_MAP[op]


def all_reduce_(tensor: torch.Tensor, op: str = "sum") -> None:
    """All-reduce ``tensor`` across ranks in place.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialized.
        ValueError: If ``op`` is not a recognized reduction.
    """
    reduce_op = _resolve_op(op)
    if not is_distributed():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call torch.distributed.init_process_group() first."
        )
    dist.all_reduce(tensor, op=reduce_op)


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """Return a reduced clone of ``tensor``; input is unchanged."""
    reduced = tensor.clone()
    all_reduce_(reduced, op=op)
    return reduced


def barrier() -> None:
    """Block until every rank reaches this call (no-op if not distributed)."""
    if is_distributed():
        dist.barrier()


__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "all_reduce_",
    "barrier",
]
