"""Thin wrappers around optional provider collectives.

Provides ``is_distributed``, ``get_rank``, ``get_world_size``, ``all_reduce``,
and ``barrier``. All are safe to call outside a
process group; non-distributed contexts fall through to sensible defaults
(rank 0, world_size 1, no-op barrier).
"""

from __future__ import annotations

from typing import Any

from opaque.api.engine import runtime


def is_distributed() -> bool:
    """Return whether more than one process participates in collectives."""
    return get_world_size() > 1


def get_rank() -> int:
    """Return the current rank (0 if not distributed)."""
    return runtime.distributed_rank()


def get_world_size() -> int:
    """Return the world size (1 if not distributed)."""
    return runtime.distributed_world_size()


def all_reduce(tensor: Any, op: str = "sum") -> Any:
    """Return a reduced value; input is unchanged."""
    return runtime.distributed_all_reduce(tensor, op=runtime.ReduceOp(op))


def barrier() -> None:
    """Block until every rank reaches this call (no-op if not distributed)."""
    runtime.distributed_barrier()


def is_main_process() -> bool:
    """Return True on the rank-0 process (always True if not distributed)."""
    return get_rank() == 0


def num_processes() -> int:
    """Return the world size (1 if not distributed). Alias of ``get_world_size``."""
    return get_world_size()


def process_index() -> int:
    """Return the current rank (0 if not distributed). Alias of ``get_rank``."""
    return get_rank()


def wait_for_everyone() -> None:
    """Block until every rank reaches this call (no-op if not distributed).

    Module-level alias of :func:`barrier`, matching the
    ``accelerator.wait_for_everyone()`` idiom that callers port from.
    """
    barrier()


def gather_for_metrics(tensor: Any) -> Any:
    """All-gather ``tensor`` across ranks and concatenate along dim 0.

    In a non-distributed context returns ``tensor`` unchanged. Intended for
    metric aggregation (e.g. detached KL means, reference-logprob shards),
    where duplicate samples from Poisson sampling are not a correctness
    concern. This is **not** a gradient primitive — do not use it inside the
    clipped/noised per-example gradient path.

    All ranks must pass arrays with the same dtype, rank, and trailing shape;
    the leading dimension may vary by rank.
    """
    return runtime.distributed_all_gather(tensor, axis=0)


__all__ = [
    "all_reduce",
    "barrier",
    "gather_for_metrics",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "num_processes",
    "process_index",
    "wait_for_everyone",
]
