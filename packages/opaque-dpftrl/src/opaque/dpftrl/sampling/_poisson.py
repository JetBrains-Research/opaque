"""Cyclic Poisson sampler for DP-FTRL training (identity and band-MF regimes).

Examples are partitioned into ``bands`` groups; at iteration ``i`` the
sampler draws a Poisson(``sample_rate``) batch from group ``i % bands``.
With ``bands == 1`` this is plain Poisson subsampling over the full dataset
(identity-encoder baseline).  With ``bands > 1`` it matches the cyclic
participation pattern used in BandMF amplification.

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
    """Cyclic Poisson subsampling for DP-FTRL (one active group per step).

    At iteration ``i``, each example in group ``i % bands`` is included
    independently with probability ``sample_rate``.  ``bands=1`` is the
    degenerate case: a single group, so each step is plain Poisson over the
    full dataset (identity baseline).

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        sample_rate: Per-step inclusion probability ``∈ (0, 1]``.
        bands: Number of groups in the participation cycle.  ``1`` = one group
            (plain Poisson on the full dataset each step).
        n_steps: Total number of batches to yield.  Defaults to ``1``.
        partition_type: Partition strategy when ``bands > 1``.
        key: RNG key for reproducibility.

    Example::

        from opaque.random import key
        sampler = CyclicPoissonSampler(
            dataset, sample_rate=0.01, bands=4, n_steps=1000, key=key(42),
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        Batch sizes are variable (Poisson).  There is **no** batch-size cap
        on this sampler: ``ftrl_acc.poisson`` accounting matches uncapped
        Poisson draws only.  For capped batches, use DP-SGD's
        :class:`opaque.dpsgd.sampling.PoissonSubsampler` with
        ``dpsgd_acc.poisson(..., truncated_batch_size=, dataset_size=)``.

        Expected batch size per step is ``|group| * sample_rate`` where
        ``|group| = |D| / bands``.  Use with ``DataLoader``'s ``batch_sampler``.
    """

    def __init__(
        self,
        data_source: object,
        sample_rate: float,
        bands: int = 1,
        n_steps: int = 1,
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
        self.partition_type = partition_type

    def __iter__(self) -> Iterator[list[int]]:
        """Yield Poisson batches.

        For each step, samples from group ``step % bands`` with inclusion
        probability ``sample_rate`` per example.
        """
        for step in range(self.n_steps):
            group_idx = step % self.bands
            group = self.partition[group_idx]

            sample_size = self.generator.binomial(n=len(group), p=self.sample_rate)

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
