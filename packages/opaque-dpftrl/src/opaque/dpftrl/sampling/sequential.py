"""Sequential batch sampler for fixed-order training.

This sampler iterates through a dataset sequentially in fixed-size
batches, dropping the last incomplete batch.  It is used by the BLT
(Buffered Linear Toeplitz) mechanism, which requires a deterministic
batch order with fixed ``min_sep`` between participations.

Unlike Poisson or BnB sampling there is no randomness — the dataset
should be pre-shuffled once before constructing the sampler.
"""

from collections.abc import Iterator

from torch.utils.data import Sampler


class SequentialBatchSampler(Sampler):
    """Deterministic fixed-size sequential batching.

    Yields contiguous, non-overlapping chunks of ``batch_size`` indices
    ``[0, ..., B-1], [B, ..., 2B-1], ...``.  The last chunk is dropped
    when smaller than ``batch_size``.

    This sampler has no RNG key — it is fully deterministic.
    Call ``dataset.shuffle(seed=...)`` once beforehand to randomise
    which examples land in which batch.

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        batch_size: Exact number of examples per batch (must be ≥ 1).

    Example:
        >>> sampler = SequentialBatchSampler(dataset, batch_size=256)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(self, data_source: object, batch_size: int):
        super().__init__()

        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self._num_samples = len(data_source)
        self._batch_size = batch_size
        self._num_batches = self._num_samples // self._batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Yield fixed-size batches of contiguous indices.

        The last incomplete batch (if any) is dropped.
        """
        for i in range(self._num_batches):
            start = i * self._batch_size
            yield list(range(start, start + self._batch_size))

    def __len__(self) -> int:
        """Number of complete batches."""
        return self._num_batches

    @property
    def expected_batch_size(self) -> float:
        """Batch size (exact, not statistical)."""
        return float(self._batch_size)
