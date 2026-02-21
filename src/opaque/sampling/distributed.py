"""Distributed helper utilities for sampling.

Pure functions that take explicit ``rank`` and ``world_size`` parameters.
Use ``local_shard()`` to slice a dataset for a given rank, then pass the
resulting ``Subset`` to the sampler.
"""

from __future__ import annotations

from collections.abc import Sized

from torch.utils.data import Subset

__all__ = [
    "local_shard",
]


def local_shard(
    dataset: Sized, *, rank: int = 0, world_size: int = 1
) -> Subset:
    """Return the shard of ``dataset`` that belongs to ``rank``.

    This is the recommended way to partition a dataset for distributed
    DP training.  Each rank gets a contiguous, non-overlapping slice;
    the last rank receives any remainder examples.

    Args:
        dataset: Any object with ``__len__`` (e.g. a PyTorch ``Dataset``).
        rank: Rank of the current worker (0-indexed). Defaults to 0.
        world_size: Total number of workers. Defaults to 1 (single device).

    Returns:
        A ``torch.utils.data.Subset`` view into ``dataset``.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> from torch.utils.data import TensorDataset
        >>> ds = TensorDataset(torch.randn(1000, 10))
        >>> shard = local_shard(ds, rank=0, world_size=4)
        >>> len(shard)
        250
    """
    start, end = _local_shard_bounds(len(dataset), rank=rank, world_size=world_size)
    return Subset(dataset, range(start, end))


def _local_shard_bounds(
    dataset_size: int, *, rank: int = 0, world_size: int = 1
) -> tuple[int, int]:
    """Return ``[start, end)`` index bounds for the given rank.

    Low-level helper used by :func:`local_shard`.
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
