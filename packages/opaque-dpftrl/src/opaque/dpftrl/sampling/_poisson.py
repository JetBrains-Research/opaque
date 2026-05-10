"""Cyclic Poisson sampler for DP-FTRL training (BandMF + identity baseline).

This sampler generalises plain Poisson subsampling to the cyclic
participation pattern used by BandMF amplification: examples are
partitioned into ``bands`` groups; at iteration ``i`` the sampler yields
a Poisson(``sample_rate``) batch from group ``i % bands``.  Plain
Poisson sampling (the identity-encoder baseline) is the ``bands == 1``
special case.

For distributed training, shard the dataset before constructing the
sampler with ``opaque.distributed.local_shard`` and derive a per-rank
key with ``opaque.random.fold_in(key, rank)``.

References:
    - BandMF amplification: https://arxiv.org/abs/2306.08153
    - Cyclic Poisson sampling for matrix mechanisms: https://arxiv.org/abs/2211.06530
"""

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque.random.types import RngKey
from opaque.dpftrl.sampling._partitions import (
    PartitionType,
    _equal_split_partition,
    _independent_partition,
)


class CyclicPoissonSampler(Sampler):
    """Cyclic Poisson sampler for DP-FTRL.

    Examples are partitioned into ``bands`` groups.  At iteration ``i``,
    each example in group ``i % bands`` is independently included with
    probability ``sample_rate``.  ``bands=1`` collapses to plain
    Poisson subsampling (the identity-encoder baseline).

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        sample_rate: Per-step Poisson sampling probability ``∈ (0, 1]``.
        bands: Number of cyclic groups (band width).  Defaults to ``1``
            (plain Poisson).
        n_steps: Total number of batches to yield.  Defaults to ``1``.
        truncated_batch_size: Optional cap on per-step batch size.
        partition_type: Strategy for partitioning the dataset into groups
            (only used when ``bands > 1``).
        key: RNG key for reproducibility.

    Example::

        from opaque.random import key
        sampler = CyclicPoissonSampler(
            dataset, sample_rate=0.01, bands=4, n_steps=1000, key=key(42),
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        Batch sizes are variable (Poisson).  Expected batch size per step
        is ``|group| * sample_rate`` where ``|group| = |D| / bands``.
        Use with ``DataLoader``'s ``batch_sampler`` parameter.
    """

    def __init__(
        self,
        data_source: object,
        sample_rate: float,
        bands: int = 1,
        n_steps: int = 1,
        truncated_batch_size: int | None = None,
        partition_type: PartitionType = PartitionType.EQUAL_SPLIT,
        *,
        key: RngKey,
    ):
        super().__init__()

        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if bands < 1:
            raise ValueError(f"bands must be >= 1, got {bands}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if truncated_batch_size is not None and truncated_batch_size < 1:
            raise ValueError(
                f"truncated_batch_size must be >= 1, got {truncated_batch_size}"
            )

        self.num_examples = len(data_source)
        self.generator = np.random.default_rng(key.seed)

        if partition_type == PartitionType.INDEPENDENT:
            partition_fn = _independent_partition
        elif partition_type == PartitionType.EQUAL_SPLIT:
            partition_fn = _equal_split_partition
        else:
            raise ValueError(f"Unsupported partition_type: {partition_type}")

        dtype = np.min_scalar_type(-self.num_examples)
        self.partition = partition_fn(self.num_examples, bands, self.generator, dtype)

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.bands = bands
        self.n_steps = n_steps
        self.truncated_batch_size = truncated_batch_size
        self.partition_type = partition_type

    def __iter__(self) -> Iterator[list[int]]:
        """Yield Poisson batches.

        For each step, samples from group ``step % bands`` with inclusion
        probability ``sample_rate`` per example.  When
        ``truncated_batch_size`` is set, caps the result.
        """
        for step in range(self.n_steps):
            group_idx = step % self.bands
            group = self.partition[group_idx]

            sample_size = self.generator.binomial(n=len(group), p=self.sample_rate)
            if self.truncated_batch_size is not None:
                sample_size = min(sample_size, self.truncated_batch_size)

            if sample_size > 0:
                batch = self.generator.choice(
                    group, size=sample_size, replace=False, shuffle=False
                )
            else:
                batch = np.array([], dtype=group.dtype)

            yield batch.tolist()

    def __len__(self) -> int:
        return self.n_steps

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size: ``|group| * sample_rate``."""
        avg_group_size = self.num_examples / self.bands
        return avg_group_size * self.sample_rate

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size (Poisson property)."""
        avg_group_size = self.num_examples / self.bands
        return avg_group_size * self.sample_rate * (1 - self.sample_rate)
