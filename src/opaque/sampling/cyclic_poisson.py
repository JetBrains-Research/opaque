"""Cyclic Poisson sampler for BandMF with DDP support.

Cyclic Poisson sampling creates batches by cycling through partitioned groups
of the dataset. This enables privacy amplification via correlated noise mechanisms
like BandMF.

Supports distributed training with automatic environment detection:
- INDEPENDENT: Single device (each rank cycles independently through full dataset)
- SHARDED: Distributed training (each rank cycles through its shard)

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - Cyclic sampling: https://arxiv.org/abs/2211.06530
"""

import warnings
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque.distributed import get_rank, get_world_size, is_distributed
from opaque.random import RngKey, fold_in
from opaque.sampling._utils import (
    PartitionType,
    _equal_split_partition,
    _independent_partition,
)


class CyclicPoissonSampler(Sampler):
    """Cyclic Poisson sampler for BandMF privacy amplification with DDP support.

    This sampler implements cyclic Poisson sampling, which generalizes several
    common sampling strategies:

    - **Standard Poisson** (cycle_length=1): Each example independently included
      with probability ``sampling_prob`` (standard DP-SGD).
    - **Fixed-order** (sampling_prob=1.0, cycle_length=n//b): Deterministic
      multi-epoch order, each example in a group seen once per cycle.
    - **BandMF** (cycle_length > 1, 0 < sampling_prob < 1): Cyclic sampling for
      correlated noise mechanisms (10-50% utility gain over independent noise).

    Examples partition the dataset into ``cycle_length`` groups. At iteration i,
    the sampler yields a batch from group (i % cycle_length). This creates
    predictable structure for matrix factorization noise mechanisms.

    Supports distributed training with automatic environment detection:
    - **INDEPENDENT**: Single device (default for world_size=1)
    - **SHARDED**: Distributed training (default for world_size > 1)
      Each rank cycles through its assigned shard independently.

    Args:
        data_source: Dataset to sample from (any object with __len__)
        sampling_prob: Probability of including each eligible example (0 < p <= 1)
        cycle_length: Number of groups. Defaults to 1 (standard Poisson)
            - cycle_length=1: Each example independently sampled (standard Poisson)
            - cycle_length=N: Examples split into N groups, cyclic round-robin
        iterations: Total number of batches to yield. Defaults to None (use 1 epoch)
        truncated_batch_size: If set, cap each batch size at this value
        partition_type: How to partition into groups
            - EQUAL_SPLIT: Shuffle and split into equal-sized groups (default)
            - INDEPENDENT: Multinomial group assignment (random sizes)
        generator: Optional random seed or generator:
            - None: Unseeded (non-reproducible)
            - int: Random seed. Auto-shifts by rank in distributed mode.
            - np.random.Generator: Use provided generator directly

    Example (single device):
        >>> dataset = MyDataset(size=1000)
        >>> sampler = CyclicPoissonSampler(
        ...     dataset,
        ...     sampling_prob=0.5,
        ...     cycle_length=3,
        ...     iterations=20,
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>> # Cycles through groups: 0,1,2,0,1,2,... for 20 iterations

    Example (distributed - automatic):
        >>> from opaque.random import key
        >>> # Run with: torchrun --nproc_per_node=4 train.py
        >>> dataset = MyDataset(size=1000)
        >>> sampler = CyclicPoissonSampler(
        ...     dataset,
        ...     sampling_prob=0.5,
        ...     cycle_length=3,
        ...     iterations=20,
        ...     key=key(42),  # Auto-shifts by rank via fold_in
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>> # Device 0: partition [0:250], fold_in(key(42), 0)
        >>> # Device 1: partition [250:500], fold_in(key(42), 1)
        >>> # Each yields 20 batches independently

    Note:
        - Batch sizes are variable (Poisson property)
        - Expected batch size per iteration: |group| * sampling_prob
        - Use with DataLoader's batch_sampler parameter (not sampler)
        - Auto mode selection: Detects distributed and uses SHARDED by default
        - Auto rank shifting: Automatically applies fold_in(key, rank) in distributed mode
        - SHARDED mode: Zero communication overhead (no AllReduce)
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
        """Initialize cyclic Poisson sampler with DDP awareness."""
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

        # ============ Distributed detection ============
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_distributed = is_distributed()

        # Auto-select mode based on distributed status
        if self.is_distributed:
            self.mode = "SHARDED"
            warnings.warn(
                f"Detected distributed environment (world_size={self.world_size}). "
                f"Using SHARDED mode: each rank cycles through its shard independently.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self.mode = "INDEPENDENT"

        # ============ Compute data partition for this rank ============
        self.num_total_examples = len(data_source)

        if self.mode == "SHARDED":
            # Each rank gets a shard of the dataset
            self.shard_size = self.num_total_examples // self.world_size
            self.start_idx = self.rank * self.shard_size

            # Last rank gets the remainder
            if self.rank == self.world_size - 1:
                self.end_idx = self.num_total_examples
            else:
                self.end_idx = self.start_idx + self.shard_size

            self.num_examples_local = self.end_idx - self.start_idx
        else:
            # Single device: use full dataset
            self.shard_size = self.num_total_examples
            self.start_idx = 0
            self.end_idx = self.num_total_examples
            self.num_examples_local = self.num_total_examples

        # ============ RNG setup (with rank-based key derivation) ============
        # Fold in rank for diversity in distributed mode
        if self.is_distributed and self.rank > 0:
            # Fold rank into key for per-rank diversity
            rank_key = fold_in(key, self.rank)
        else:
            # Use key as-is for single device or rank 0
            rank_key = key

        # Convert RngKey to numpy generator and store for partition and iteration
        self.generator = np.random.default_rng(rank_key.seed)
        partition_rng = self.generator

        # Determine partition function
        if partition_type == PartitionType.INDEPENDENT:
            partition_fn = _independent_partition
        elif partition_type == PartitionType.EQUAL_SPLIT:
            partition_fn = _equal_split_partition
        else:
            raise ValueError(f"Unsupported partition_type: {partition_type}")

        # Partition the local dataset (or full dataset if single device)
        dtype = np.min_scalar_type(-self.num_examples_local)
        local_partition = partition_fn(
            self.num_examples_local, cycle_length, partition_rng, dtype
        )

        # Convert local indices to global indices (for distributed case)
        self.partition = [indices + self.start_idx for indices in local_partition]

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

        For each iteration, samples from the group (step % cycle_length).
        Each example in the group is included independently with probability
        sampling_prob, then optionally truncated.

        In SHARDED mode, all indices are from this rank's assigned shard.
        In INDEPENDENT mode, indices are from the full dataset.

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
        avg_group_size = self.num_examples_local / self.cycle_length
        return avg_group_size * self.sampling_prob

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size (Poisson property).

        Returns the average variance across groups.
        """
        # Average: group_size * p * (1 - p)
        avg_group_size = self.num_examples_local / self.cycle_length
        return avg_group_size * self.sampling_prob * (1 - self.sampling_prob)
