"""Poisson samplers for differential privacy.

These samplers implement Poisson subsampling, where each example in the dataset
is independently included in a batch with probability `sample_rate`. This provides
privacy amplification, reducing the privacy cost compared to fixed-batch sampling.

Supports distributed training with automatic environment detection:
- INDEPENDENT: Each worker samples independently (single device default)
- SHARDED: Workers sample from disjoint shards (distributed default, ensures "single Poisson")
"""

import warnings
from collections.abc import Iterator
from typing import Literal

import numpy as np
from torch.utils.data import Sampler

from opaque.distributed import get_rank, get_world_size, is_distributed


class PoissonSampler(Sampler):
    """Poisson sampler for privacy amplification.

    Each example in the dataset is independently included with probability
    `sample_rate`. This creates variable-sized batches, which provides privacy
    amplification: the effective privacy cost is reduced by approximately
    √(1/sample_rate) compared to uniform sampling.

    Supports distributed training with automatic environment detection:
    - **Auto-detection**: Automatically detects distributed environment from RANK/WORLD_SIZE env vars
    - **INDEPENDENT**: Single device training (default for world_size=1)
    - **SHARDED**: Workers sample from disjoint shards (default for distributed, ensures "single Poisson")

    Args:
        data_source: Dataset to sample from (any object with __len__)
        sample_rate: Probability of including each example (0 < p <= 1)
        num_epochs: Number of epochs to iterate over
        generator: Optional random generator or seed for reproducibility. Can be:
            - ``None``: Uses unseeded generator (non-reproducible)
            - ``int``: Seeds the generator. In distributed mode, automatically shifts by rank
              to ensure different samples per device (e.g., seed=42 becomes 42, 43, 44, ... per rank)
            - ``np.random.Generator``: Uses provided generator directly (user responsible for seeding)
        distributed: Distributed handling mode:
            - "auto": auto-select based on dist init
            - True: force SHARDED mode (requires torch.distributed initialized)
            - False: force INDEPENDENT mode (even if distributed is initialized)

    Example (single device):
        >>> dataset = MyDataset(...)
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, num_epochs=10)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>>
        >>> for batch in loader:
        ...     # batch has variable size!
        ...     # Expected size: len(dataset) * sample_rate
        ...     pass

    Example (distributed - automatic):
        >>> # When running with torchrun, automatically uses SHARDED mode
        >>> # Seed is automatically shifted by rank for diversity:
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, num_epochs=10, generator=42)
        >>> # Device 0: seed=42, Device 1: seed=43, Device 2: seed=44, ...
        >>> loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        - Batch sizes are variable (Poisson property)
        - Expected batch size: len(dataset) * sample_rate
        - Variance: len(dataset) * sample_rate * (1 - sample_rate)
                - Use with DataLoader's batch_sampler parameter (not sampler)
                - **Auto mode selection**: Detects distributed env and uses sharded sampling by default
                - **Auto seed shifting**: When int seed is provided in distributed mode, shifts by rank
                - Sharded sampling: Each worker samples from its shard only (zero communication overhead)
                - Independent sampling in distributed training: model as parallel Poisson sampling
                    (use acc.parallel_poisson(acc.poisson(acc.gaussian(nm), rate), num_workers=world_size))
    """

    def __init__(
        self,
        data_source,
        sample_rate: float,
        num_epochs: int = 1,
        generator: np.random.Generator | int | None = None,
        distributed: Literal["auto"] | bool = "auto",
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

        if distributed not in ("auto", True, False):
            raise ValueError(
                f"distributed must be one of {{'auto', True, False}}, got {distributed!r}"
            )

        if distributed is True:
            if not dist_initialized:
                raise RuntimeError(
                    "distributed=True requested but torch.distributed is not initialized."
                )
            use_sharded = True
        elif distributed is False:
            use_sharded = False
        else:
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

        # Warn about parallel Poisson accounting in independent sampling with distributed
        if self._should_warn_parallel_poisson(use_sharded, dist_initialized, world_size):
            warnings.warn(
                "Using independent sampling in distributed training (world_size > 1) "
                "uses parallel Poisson sampling. Account with acc.parallel_poisson("
                "acc.poisson(acc.gaussian(nm), rate), num_workers=world_size). "
                "Consider sharded sampling for standard DP-SGD accounting.",
                UserWarning,
                stacklevel=2,
            )

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.num_epochs = num_epochs
        self._use_sharded = use_sharded
        self.rank = rank
        self.world_size = world_size

        self._num_samples = len(data_source)

        # RNG setup: auto-shift seed by rank for sampling diversity in distributed mode
        if isinstance(generator, int):
            # Seed is an integer: shift by rank for diversity across devices
            if rank is not None and rank > 0:
                shifted_seed = generator + rank
            else:
                shifted_seed = generator
            self.generator = np.random.default_rng(shifted_seed)
        elif generator is not None:
            # Generator object provided: use as-is
            self.generator = generator
        else:
            # No seed specified: use unseeded default generator
            self.generator = np.random.default_rng()

    def _should_warn_parallel_poisson(
        self, use_sharded: bool, dist_initialized: bool, world_size: int | None
    ) -> bool:
        """Check if we should warn about parallel Poisson accounting."""
        return (
            (not use_sharded)
            and dist_initialized
            and world_size is not None
            and world_size > 1
        )

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
