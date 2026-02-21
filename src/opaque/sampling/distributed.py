"""Distributed helper utilities for sampling.

Pure functions that take explicit ``rank`` and ``world_size`` parameters.
Use ``local_shard_bounds()`` to compute shard boundaries, then pass a
``Subset`` to the sampler.
"""

from __future__ import annotations

__all__ = [
    "local_shard_bounds",
]


def local_shard_bounds(
    dataset_size: int, *, rank: int = 0, world_size: int = 1
) -> tuple[int, int]:
    """Return ``[start, end)`` shard bounds for the given rank.

    Args:
        dataset_size: Total number of examples in the dataset.
        rank: Rank of the current worker (0-indexed). Defaults to 0.
        world_size: Total number of workers. Defaults to 1 (single device).

    Returns:
        Tuple of (start_index, end_index). The last rank receives any
        remainder examples when ``dataset_size`` is not evenly divisible.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> local_shard_bounds(1000, rank=0, world_size=4)
        (0, 250)
        >>> local_shard_bounds(1000, rank=3, world_size=4)
        (750, 1000)
    """
    if dataset_size < 0:
        raise ValueError(f"dataset_size must be >= 0, got {dataset_size}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(
            f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}"
        )

    if world_size == 1:
        return 0, dataset_size

    shard = dataset_size // world_size
    start = rank * shard
    end = dataset_size if rank == world_size - 1 else (start + shard)
    return start, end
