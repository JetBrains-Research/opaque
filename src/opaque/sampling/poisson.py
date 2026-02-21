"""Poisson samplers for differential privacy.

These samplers implement Poisson subsampling, where each example in the dataset
is independently included in a batch with probability `sample_rate`. This provides
privacy amplification, reducing the privacy cost compared to fixed-batch sampling.

Supports distributed training with automatic environment detection:
- INDEPENDENT: Single-device default
- SHARDED: Workers sample from disjoint shards when distributed is initialized
"""

import warnings
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque.distributed import get_rank, get_world_size, is_distributed
from opaque.random import RngKey, fold_in


class PoissonSampler(Sampler):
    """Poisson sampler for privacy amplification.

    Each example in the dataset is independently included with probability
    `sample_rate`. This creates variable-sized batches, which provides privacy
    amplification: the effective privacy cost is reduced by approximately
    √(1/sample_rate) compared to uniform sampling.

    Supports distributed training with automatic environment detection:
    - **INDEPENDENT**: Single-device training (default when not distributed)
    - **SHARDED**: Workers sample from disjoint shards (default when distributed)

    Args:
        data_source: Dataset to sample from (any object with __len__)
        sample_rate: Probability of including each example (0 < p <= 1)
        num_epochs: Number of epochs to iterate over
        key: RNG key for reproducibility. Use ``key()`` or ``training_key()`` helpers.
            In distributed SHARDED mode, automatically applies ``fold_in(key, rank)`` for
            per-rank diversity while maintaining reproducibility.

    Example (single device):
        >>> from opaque.random import key
        >>> dataset = MyDataset(...)
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, num_epochs=10, key=key(42))
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>>
        >>> for batch in loader:
        ...     # batch has variable size!
        ...     # Expected size: len(dataset) * sample_rate
        ...     pass

    Example (distributed - automatic rank shifting):
        >>> from opaque.random import key, fold_in
        >>> # Rank shifting happens automatically in SHARDED mode:
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, num_epochs=10, key=key(42))
        >>> # Rank 0: uses key(42), Rank 1: uses fold_in(key(42), 1), etc.
        >>> loader = DataLoader(dataset, batch_sampler=sampler)

    Example (training loop):
        >>> from opaque.random import training_key
        >>> for epoch in range(epochs):
        ...     k = training_key(base_seed=42, step=epoch)
        ...     sampler = PoissonSampler(dataset, sample_rate=0.01, key=k)
        ...     loader = DataLoader(dataset, batch_sampler=sampler)
        ...     for batch in loader:
        ...         # train ...

    Note:
        - Batch sizes are variable (Poisson property)
        - Expected batch size: len(dataset) * sample_rate
        - Variance: len(dataset) * sample_rate * (1 - sample_rate)
        - Use with DataLoader's batch_sampler parameter (not sampler)
        - Detects distributed env and uses sharded sampling by default
        - In SHARDED mode, applies fold_in(key, rank) automatically
        - Sharded sampling: Each worker samples from its shard only (zero communication overhead)
    """

    def __init__(
        self,
        data_source,
        sample_rate: float,
        num_epochs: int = 1,
        *,
        key: RngKey,
    ):
        super().__init__()

        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

        # Get rank and world size from distributed module
        rank = get_rank()
        world_size = get_world_size()
        dist_initialized = is_distributed()

        # Smart default: SHARDED for distributed, INDEPENDENT for single device
        use_sharded = dist_initialized
        if dist_initialized:
            warnings.warn(
                f"Detected distributed environment (world_size={world_size}). "
                f"Automatically using SHARDED mode for correct DP accounting.",
                UserWarning,
                stacklevel=2,
            )

        # Validate distributed parameters
        if use_sharded:
            if not dist_initialized:
                raise ValueError(
                    "SHARDED mode requires distributed training to be initialized. "
                    "Initialize with torch.distributed.init_process_group() "
                    "or use torch.distributed.launch / torchrun."
                )
            if not 0 <= rank < world_size:
                raise ValueError(
                    f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}"
                )
            if world_size < 1:
                raise ValueError(f"world_size must be >= 1, got {world_size}")

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.num_epochs = num_epochs
        self._use_sharded = use_sharded
        self.rank = rank
        self.world_size = world_size

        self._num_samples = len(data_source)

        # RNG setup: fold in rank for diversity in SHARDED distributed mode
        if self._use_sharded and dist_initialized and self.rank > 0:
            # Fold rank into key for per-rank diversity
            rank_key = fold_in(key, self.rank)
        else:
            # Use key as-is for single device or INDEPENDENT mode
            rank_key = key

        # Convert RngKey to numpy generator
        self.generator = np.random.default_rng(rank_key.seed)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches as lists of indices.

        Each call samples the entire dataset once per epoch using Poisson
        subsampling. Examples are included independently with probability
        sample_rate.

        In SHARDED mode, each worker samples only from its assigned shard.
        In INDEPENDENT mode, each worker samples independently (may differ).

        Returns:
            Iterator yielding lists of indices (variable size)
        """
        for _ in range(self.num_epochs):
            if self._use_sharded:
                # Compute this worker's shard boundaries
                shard_size = self._num_samples // self.world_size
                start_idx = self.rank * shard_size

                # Last worker gets remainder
                if self.rank == self.world_size - 1:
                    end_idx = self._num_samples
                else:
                    end_idx = start_idx + shard_size

                # Sample only from this worker's shard
                shard_length = end_idx - start_idx
                shard_mask = self.generator.random(shard_length) < self.sample_rate
                shard_indices = np.where(shard_mask)[0] + start_idx

                yield shard_indices.tolist()

            else:  # INDEPENDENT
                # Original behavior: each worker samples independently
                included = self.generator.random(self._num_samples) < self.sample_rate
                indices = np.where(included)[0]

                yield indices.tolist()

    def __len__(self) -> int:
        """Return number of batches (one per epoch)."""
        return self.num_epochs

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size = num_samples * sample_rate."""
        return self._num_samples * self.sample_rate

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size for Poisson sampling."""
        return self._num_samples * self.sample_rate * (1 - self.sample_rate)
