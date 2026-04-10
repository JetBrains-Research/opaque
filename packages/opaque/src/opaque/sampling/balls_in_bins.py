"""Balls-in-Bins sampler for DP-SGD training.

In the Balls-in-Bins (BnB) sampling scheme, each epoch the dataset is
randomly shuffled and partitioned into ``num_bins`` equally-sized bins.
Each bin is processed exactly once, so every example participates exactly
once per epoch with deterministic batch sizes.

This provides privacy amplification because the adversary does not know
which bin contains a given example (rate = 1/num_bins).

For distributed training, shard the dataset **before** creating the sampler
using ``local_shard()`` and derive a per-rank key with ``fold_in(key, rank)``.

References:
    - Chua et al. (2025), "Scalable Shuffle Differential Privacy"
    - Choquette-Choo et al. (2024), "Privacy Amplification for Matrix Mechanisms"
"""

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque.random import RngKey


class BallsInBinsSampler(Sampler):
    """Balls-in-Bins sampler: deterministic batch sizes, random assignment.

    Each epoch, the dataset is reshuffled and split into ``num_bins``
    contiguous bins of size ``floor(len(dataset) / num_bins)``.
    Remainder examples are dropped (like PyTorch's ``drop_last=True``).

    For distributed training, shard the dataset externally and pass a
    per-rank key via ``fold_in(key, rank)``.

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        num_bins: Number of bins per epoch (k ≥ 2). Typically
            ``dataset_size / desired_batch_size``.
        num_epochs: Number of epochs to yield. If None, yields indefinitely.
        key: RNG key for reproducibility.

    Example:
        >>> from opaque.random import key
        >>> dataset = MyDataset(...)
        >>> sampler = BallsInBinsSampler(
        ...     dataset, num_bins=100, num_epochs=10, key=key(42)
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        data_source,
        num_bins: int,
        num_epochs: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__()

        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if num_epochs is not None and num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1 or None, got {num_epochs}")

        self.data_source = data_source
        self.num_bins = num_bins
        self.num_epochs = num_epochs

        self._num_samples = len(data_source)
        self._bin_size = self._num_samples // num_bins

        if self._bin_size == 0:
            raise ValueError(
                f"dataset too small ({self._num_samples}) for {num_bins} bins"
            )

        self.generator = np.random.default_rng(key.seed)

    def _epoch_batches(self) -> list[list[int]]:
        """Shuffle and partition into bins for one epoch."""
        indices = self.generator.permutation(self._num_samples)
        # Take only num_bins * bin_size elements (drop remainder)
        usable = self.num_bins * self._bin_size
        indices = indices[:usable]
        # Split into num_bins contiguous chunks
        bins = indices.reshape(self.num_bins, self._bin_size)
        return [row.tolist() for row in bins]

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches: all bins from each epoch in order.

        Yields:
            Lists of indices, one per bin. Each epoch produces
            ``num_bins`` batches.
        """
        if self.num_epochs is None:
            while True:
                yield from self._epoch_batches()
        else:
            for _ in range(self.num_epochs):
                yield from self._epoch_batches()

    def __len__(self) -> int:
        """Total number of batches across all epochs.

        Raises:
            TypeError: If num_epochs is None (infinite iteration).
        """
        if self.num_epochs is None:
            raise TypeError("len() of unsized object (num_epochs=None)")
        return self.num_bins * self.num_epochs

    @property
    def batch_size(self) -> int:
        """Deterministic batch size: floor(len(dataset) / num_bins)."""
        return self._bin_size

    @property
    def sample_rate(self) -> float:
        """Effective sampling rate: 1 / num_bins."""
        return 1.0 / self.num_bins
