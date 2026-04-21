"""Poisson samplers for differential privacy.

These samplers implement Poisson subsampling, where each example in the dataset
is independently included in a batch with probability ``sample_rate``. This provides
privacy amplification, reducing the privacy cost compared to fixed-batch sampling.

For distributed training, shard the dataset **before** creating the sampler using
``local_shard()`` and derive a per-rank key with ``fold_in(key, rank)``.
"""

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque.random import RngKey


class PoissonSampler(Sampler):
    """Poisson sampler for privacy amplification.

    Each example in the dataset is independently included with probability
    ``sample_rate``. This creates variable-sized batches, which provides privacy
    amplification: the effective privacy cost is reduced by approximately
    √(1/sample_rate) compared to uniform sampling.

    For distributed training, shard the dataset externally and pass a per-rank
    key via ``fold_in(key, rank)``:

    .. code-block:: python

        from opaque.distributed.shard import local_shard

        shard = local_shard(dataset, rank=rank, world_size=world_size)
        sampler = PoissonSampler(shard, sample_rate=0.01, key=fold_in(key(42), rank))

    Args:
        data_source: Dataset to sample from (any object with ``__len__``)
        sample_rate: Probability of including each example (0 < p ≤ 1)
        num_iterations: Number of batches to yield. If None, yields batches indefinitely.
        key: RNG key for reproducibility. Use ``key()`` or ``fold_in()``.

    Example:
        >>> from opaque.random import key
        >>> dataset = MyDataset(...)
        >>> # Yield 10 batches
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, num_iterations=10, key=key(42))
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>> # Yield batches indefinitely
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, num_iterations=None, key=key(42))
        >>> loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        - Batch sizes are variable (Poisson property).
        - Expected batch size: ``len(data_source) * sample_rate``.
        - Use with DataLoader's ``batch_sampler`` parameter (not ``sampler``).
    """

    def __init__(
        self,
        data_source: object,
        sample_rate: float,
        num_iterations: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__()

        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if num_iterations is not None and num_iterations < 1:
            raise ValueError(
                f"num_iterations must be >= 1 or None, got {num_iterations}"
            )

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.num_iterations = num_iterations

        self._num_samples = len(data_source)

        # Convert RngKey to numpy generator
        self.generator = np.random.default_rng(key.seed)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches as lists of indices.

        Each iteration samples the entire dataset using Poisson subsampling.
        Examples are included independently with probability ``sample_rate``.

        Returns:
            Iterator yielding lists of indices (variable size)
        """
        if self.num_iterations is None:
            # Infinite iteration
            while True:
                included = self.generator.random(self._num_samples) < self.sample_rate
                indices = np.where(included)[0]
                yield indices.tolist()
        else:
            # Fixed number of iterations
            for _ in range(self.num_iterations):
                included = self.generator.random(self._num_samples) < self.sample_rate
                indices = np.where(included)[0]
                yield indices.tolist()

    def __len__(self) -> int:
        """Return number of batches.

        Raises:
            TypeError: If num_iterations is None (infinite iteration)
        """
        if self.num_iterations is None:
            raise TypeError("len() of unsized object (num_iterations=None)")
        return self.num_iterations

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size = num_samples * sample_rate."""
        return self._num_samples * self.sample_rate

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size for Poisson sampling."""
        return self._num_samples * self.sample_rate * (1 - self.sample_rate)
