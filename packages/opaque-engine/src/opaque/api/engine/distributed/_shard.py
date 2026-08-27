"""Dataset sharding helpers for distributed DP training.

Pure functions that take explicit ``rank`` and ``world_size`` parameters. Use
:func:`local_shard` to slice a dataset for a given rank, then pass the
resulting ``Subset`` to the sampler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


class DatasetShard:
    """A read-only view of a contiguous range of another sequence.

    What :func:`local_shard` returns. It holds the source sequence and the
    indices this rank owns rather than copying elements, so sharding a dataset
    costs nothing and the underlying objects are loaded lazily by whatever
    ``dataset`` does on indexing.

    It supports exactly the sequence protocol a sampler and a data loader need:
    ``len(shard)`` gives this rank's element count, ``shard[i]`` gives one
    element, and ``shard[a:b]`` gives a ``list`` of them. It is not a copy and
    not writable; iterate or index it, and build any richer structure yourself.

    Attributes:
        dataset: The sequence being viewed.
        indices: The ``range`` of source positions this shard covers.
    """

    def __init__(self, dataset: Sequence[Any], indices: range) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int | slice) -> Any:
        selected = self.indices[index]
        if isinstance(selected, range):
            return [self.dataset[source_index] for source_index in selected]
        return self.dataset[selected]


def local_shard(
    dataset: Sequence[Any], *, rank: int = 0, world_size: int = 1
) -> DatasetShard:
    """Return the shard of ``dataset`` that belongs to ``rank``.

    Each rank gets a contiguous, non-overlapping slice; the last rank receives
    any remainder examples, so shards differ in length by up to
    ``world_size - 1`` elements. Sharding by rank changes what each rank
    samples from — derive the sampler's key per rank to match, and remember
    that per-rank dataset sizes feed a sampling rate.

    Args:
        dataset: Any indexable sequence or dataset. Only ``__len__`` and
            ``__getitem__`` are used.
        rank: Rank of the current worker (0-indexed). Defaults to 0.
        world_size: Total number of workers. Defaults to 1.

    Returns:
        A :class:`DatasetShard` view over this rank's range. No elements are
        copied.

    Raises:
        ValueError: If ``world_size`` is below 1 or ``rank`` is outside
            ``[0, world_size)``.
    """
    start, end = _local_shard_bounds(len(dataset), rank=rank, world_size=world_size)
    return DatasetShard(dataset, range(start, end))


def _local_shard_bounds(
    dataset_size: int, *, rank: int = 0, world_size: int = 1
) -> tuple[int, int]:
    """Return ``[start, end)`` index bounds for ``rank``."""
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


__all__ = ["DatasetShard", "local_shard"]
