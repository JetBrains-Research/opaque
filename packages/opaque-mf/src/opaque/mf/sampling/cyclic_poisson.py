"""Cyclic Poisson sampler for BandMF.

Cyclic Poisson sampling creates batches by cycling through partitioned groups
of the dataset. This enables privacy amplification via correlated noise mechanisms
like BandMF.

For distributed training, shard the dataset **before** creating the sampler using
``local_shard()`` and derive a per-rank key with ``fold_in(key, rank)``.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - Cyclic sampling: https://arxiv.org/abs/2211.06530
"""

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque.core.random import RngKey
from opaque.core.sampling._utils import (
    PartitionType,
    _equal_split_partition,
    _independent_partition,
)


class CyclicPoissonSampler(Sampler):
    """Cyclic Poisson sampler for BandMF privacy amplification.

    This sampler implements cyclic Poisson sampling, which generalizes several
    common sampling strategies:

    - **Standard Poisson** (cycle_length=1): Each example independently included
      with probability ``sampling_prob`` (standard DP-SGD).
    - **Fixed-order** (sampling_prob=1.0, cycle_length=n//b): Deterministic
      multi-epoch order, each example in a group seen once per cycle.
    - **BandMF** (cycle_length > 1, 0 < sampling_prob < 1): Cyclic sampling for
      correlated noise mechanisms (10-50% utility gain over independent noise).

    Examples partition the dataset into ``cycle_length`` groups. At iteration i,
    the sampler yields a batch from group ``i % cycle_length``. This creates
    predictable structure for matrix factorization noise mechanisms.

    For distributed training, shard the dataset externally and pass a per-rank
    key via ``fold_in(key, rank)``:

    .. code-block:: python

        from opaque.core.sampling.distributed import local_shard

        shard = local_shard(dataset, rank=rank, world_size=world_size)
        sampler = CyclicPoissonSampler(shard, sampling_prob=0.5, key=fold_in(key(42), rank))

    Args:
        data_source: Dataset to sample from (any object with ``__len__``)
        sampling_prob: Probability of including each eligible example (0 < p ≤ 1)
        cycle_length: Number of groups. Defaults to 1 (standard Poisson).
        iterations: Total number of batches to yield. Defaults to 1.
        truncated_batch_size: If set, cap each batch size at this value.
        partition_type: How to partition into groups.
        key: RNG key for reproducibility.

    Example:
        >>> from opaque.core.random import key
        >>> dataset = MyDataset(size=1000)
        >>> sampler = CyclicPoissonSampler(
        ...     dataset,
        ...     sampling_prob=0.5,
        ...     cycle_length=3,
        ...     iterations=20,
        ...     key=key(42),
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        - Batch sizes are variable (Poisson property).
        - Expected batch size per iteration: ``|group| * sampling_prob``.
        - Use with DataLoader's ``batch_sampler`` parameter (not ``sampler``).
    """

    def __init__(
        self,
        data_source,
        sampling_prob: float,
        cycle_length: int = 1,
        iterations: int | None = None,
        truncated_batch_size: int | None = None,
        partition_type: PartitionType = PartitionType.EQUAL_SPLIT,
        *,
        key: RngKey,
    ):
        """Initialize cyclic Poisson sampler."""
        super().__init__()

        # ============ Validate inputs ============
        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if not 0 < sampling_prob <= 1:
            raise ValueError(f"sampling_prob must be in (0, 1], got {sampling_prob}")
        if cycle_length < 1:
            raise ValueError(f"cycle_length must be >= 1, got {cycle_length}")
        if truncated_batch_size is not None and truncated_batch_size < 1:
            raise ValueError(
                f"truncated_batch_size must be >= 1, got {truncated_batch_size}"
            )

        # ============ Compute data partition ============
        self.num_examples = len(data_source)

        # ============ RNG setup ============
        # Convert RngKey to numpy generator
        self.generator = np.random.default_rng(key.seed)
        partition_rng = self.generator

        # Determine partition function
        if partition_type == PartitionType.INDEPENDENT:
            partition_fn = _independent_partition
        elif partition_type == PartitionType.EQUAL_SPLIT:
            partition_fn = _equal_split_partition
        else:
            raise ValueError(f"Unsupported partition_type: {partition_type}")

        # Partition the dataset into groups
        dtype = np.min_scalar_type(-self.num_examples)
        self.partition = partition_fn(
            self.num_examples, cycle_length, partition_rng, dtype
        )

        # ============ Store parameters ============
        self.data_source = data_source
        self.sampling_prob = sampling_prob
        self.cycle_length = cycle_length
        self.truncated_batch_size = truncated_batch_size
        self.partition_type = partition_type

        # Compute iterations
        if iterations is None:
            self.iterations = 1
        else:
            if iterations < 1:
                raise ValueError(f"iterations must be >= 1, got {iterations}")
            self.iterations = iterations

    def __iter__(self) -> Iterator[list[int]]:
        """Yield cyclic batches as lists of indices.

        For each iteration, samples from the group ``step % cycle_length``.
        Each example in the group is included independently with probability
        ``sampling_prob``, then optionally truncated.

        Yields:
            Variable-size lists of indices
        """
        for step in range(self.iterations):
            # Get the group for this iteration
            group_idx = step % self.cycle_length
            group = self.partition[group_idx]

            # Sample binomially: each example included with probability sampling_prob
            sample_size = self.generator.binomial(n=len(group), p=self.sampling_prob)

            # Cap at truncated_batch_size if specified
            if self.truncated_batch_size is not None:
                sample_size = min(sample_size, self.truncated_batch_size)

            # Sample without replacement
            if sample_size > 0:
                batch = self.generator.choice(
                    group, size=sample_size, replace=False, shuffle=False
                )
            else:
                batch = np.array([], dtype=group.dtype)

            # Yield as list of Python ints
            yield batch.tolist()

    def __len__(self) -> int:
        """Return total number of batches (iterations)."""
        return self.iterations

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size across all iterations.

        Returns the average expected size: (total possible samples / num_iterations) * sampling_prob.
        """
        # Average group size * sampling_prob
        avg_group_size = self.num_examples / self.cycle_length
        return avg_group_size * self.sampling_prob

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size (Poisson property).

        Returns the average variance across groups.
        """
        # Average: group_size * p * (1 - p)
        avg_group_size = self.num_examples / self.cycle_length
        return avg_group_size * self.sampling_prob * (1 - self.sampling_prob)
