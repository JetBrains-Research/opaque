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

import numpy as np
from torch.utils.data import Sampler

from opaque.distributed import get_rank, get_world_size, is_distributed
from opaque.sampling.types import SamplingMode


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
        mode: Sampling mode for distributed training. If None (default), automatically
            selects SHARDED for distributed (world_size > 1) or INDEPENDENT for single device

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
        - **Auto mode selection**: Detects distributed env and uses SHARDED by default
        - **Auto seed shifting**: When int seed is provided in distributed mode, shifts by rank
        - SHARDED mode: Each worker samples from its shard only (zero communication overhead)
        - INDEPENDENT mode: Use mixture Gaussian accounting (future work)
    """

    def __init__(
        self,
        data_source,
        sample_rate: float,
        num_epochs: int = 1,
        generator: np.random.Generator | int | None = None,
        mode: SamplingMode | None = None,
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
        if mode is None:
            if dist_initialized:
                mode = SamplingMode.SHARDED
                warnings.warn(
                    f"Detected distributed environment (world_size={world_size}). "
                    f"Automatically using SHARDED mode for correct DP accounting.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                mode = SamplingMode.INDEPENDENT

        # Validate distributed parameters
        if mode == SamplingMode.SHARDED:
            if not dist_initialized:
                raise ValueError(
                    f"SHARDED mode requires distributed training to be initialized. "
                    f"Initialize with torch.distributed.init_process_group() "
                    f"or use torch.distributed.launch / torchrun."
                )
            if not 0 <= rank < world_size:
                raise ValueError(
                    f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}"
                )
            if world_size < 1:
                raise ValueError(f"world_size must be >= 1, got {world_size}")

        # Warn about mixture Gaussian accounting in INDEPENDENT mode with distributed
        if self._should_warn_mixture_gaussian(mode, rank, world_size):
            warnings.warn(
                "Using INDEPENDENT mode in distributed training (world_size > 1) "
                "leads to mixture Gaussian accounting, which is not yet supported. "
                "Consider using SHARDED mode for standard DP-SGD accounting.",
                UserWarning,
                stacklevel=2,
            )

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.num_epochs = num_epochs
        self.mode = mode
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

    def _should_warn_mixture_gaussian(
        self, mode: SamplingMode, rank: int | None, world_size: int | None
    ) -> bool:
        """Check if we should warn about mixture Gaussian accounting."""
        return (
            mode == SamplingMode.INDEPENDENT
            and rank is not None
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
            if self.mode == SamplingMode.SHARDED:
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
