"""Balls-in-Bins sampler for DP-SGD training.

In the Balls-in-Bins (BnB) sampling scheme each example independently
and uniformly picks one of ``num_bins`` bins (Definition 3.1 of
Choquette-Choo et al. 2024).  Bin sizes are therefore **random**,
following Binomial(N, 1/num_bins) marginally.

The bin assignment is generated once in ``__init__`` and reused every
epoch (round-robin).  This is required for the dominating-pair privacy
accounting (Lemma 3.2) to be valid: each example must stay in its bin
across all epochs.

For distributed training, shard the dataset **before** creating the
sampler using ``local_shard()`` and derive a per-rank key with
``fold_in(key, rank)``.

References:
    - Chua et al. (2025), "Scalable Shuffle Differential Privacy"
    - Choquette-Choo et al. (2024), "Privacy Amplification for Matrix Mechanisms"
"""

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque.random.types import RngKey


class BallsInBinsSampler(Sampler):
    """Balls-in-Bins sampler: random independent bin assignment, fixed across epochs.

    Each example independently chooses one of ``num_bins`` bins uniformly
    at random (true BnB, Definition 3.1 of Choquette-Choo et al. 2024).
    Bin sizes are random — some bins may be larger or smaller than the
    expected size ``len(dataset) / num_bins``, and bins can even be empty.

    The assignment is generated once and reused every epoch — this is
    required for the BnB privacy accounting (Lemma 3.2) to be valid.

    For distributed training, shard the dataset externally and pass a
    per-rank key via ``fold_in(key, rank)``.

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        num_bins: Number of bins per epoch (b ≥ 2).  Typically
            ``dataset_size / desired_batch_size``.
        n_steps: Total number of batches to yield.  Must be a positive
            multiple of ``num_bins`` (per-bin participation count is
            ``n_steps // num_bins``).  ``None`` yields indefinitely.
        key: RNG key for reproducibility.

    Example:
        >>> from opaque.random import key
        >>> sampler = BallsInBinsSampler(
        ...     dataset, num_bins=100, n_steps=1000, key=key(42)
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        data_source: object,
        num_bins: int,
        n_steps: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__()

        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if n_steps is not None:
            if n_steps < 1:
                raise ValueError(f"n_steps must be >= 1 or None, got {n_steps}")
            if n_steps % num_bins != 0:
                raise ValueError(
                    f"n_steps ({n_steps}) must be a positive multiple of "
                    f"num_bins ({num_bins}); BnB analysis assumes integer epochs."
                )

        self.data_source = data_source
        self.num_bins = num_bins
        self.n_steps = n_steps

        self._num_samples = len(data_source)

        generator = np.random.default_rng(key.seed)
        # True BnB: each example independently picks a bin.
        assignments = generator.integers(0, num_bins, size=self._num_samples)
        self._bins: list[list[int]] = [[] for _ in range(num_bins)]
        for idx, b in enumerate(assignments):
            self._bins[b].append(idx)

    @property
    def num_epochs(self) -> int | None:
        """Per-bin participation count: ``n_steps // num_bins``."""
        return None if self.n_steps is None else self.n_steps // self.num_bins

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches: round-robin over bins, repeated for ``num_epochs``.

        The same bin assignment is yielded every epoch.  Empty bins are
        skipped.

        Yields:
            Lists of indices, one per non-empty bin per epoch.
        """
        if self.n_steps is None:
            while True:
                for batch in self._bins:
                    if batch:
                        yield batch
        else:
            epochs = self.n_steps // self.num_bins
            for _ in range(epochs):
                for batch in self._bins:
                    if batch:
                        yield batch

    def __len__(self) -> int:
        """Total number of non-empty batches across all epochs.

        Raises:
            TypeError: If n_steps is None (infinite iteration).
        """
        if self.n_steps is None:
            raise TypeError("len() of unsized object (n_steps=None)")
        non_empty = sum(1 for b in self._bins if b)
        epochs = self.n_steps // self.num_bins
        return non_empty * epochs

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size: len(dataset) / num_bins."""
        return self._num_samples / self.num_bins

    @property
    def sample_rate(self) -> float:
        """Effective sampling rate: 1 / num_bins."""
        return 1.0 / self.num_bins
