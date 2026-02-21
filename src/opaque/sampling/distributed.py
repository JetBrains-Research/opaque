"""Distributed helper utilities for sampling components."""

from __future__ import annotations

from opaque.distributed import get_rank, get_world_size, is_distributed
from opaque.random import RngKey, fold_in

__all__ = [
    "local_shard_bounds",
    "rank_key",
]


def local_shard_bounds(dataset_size: int) -> tuple[int, int]:
    """Return [start, end) shard bounds for current rank.

    If distributed is not initialized, returns ``(0, dataset_size)``.
    """
    if dataset_size < 0:
        raise ValueError(f"dataset_size must be >= 0, got {dataset_size}")

    if not is_distributed():
        return 0, dataset_size

    rank = get_rank()
    world_size = get_world_size()
    shard = dataset_size // world_size
    start = rank * shard
    end = dataset_size if rank == world_size - 1 else (start + shard)
    return start, end


def rank_key(key: RngKey) -> RngKey:
    """Derive a rank-specific key in distributed mode."""
    if not is_distributed():
        return key
    rank = get_rank()
    return key if rank == 0 else fold_in(key, rank)
