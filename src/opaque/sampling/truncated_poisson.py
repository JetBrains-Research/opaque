"""Truncated Poisson sampler for differential privacy.

This module provides TruncatedPoissonSampler, which combines Poisson subsampling
for privacy amplification with a maximum batch size constraint for stability.

Supports distributed training with automatic environment detection:
- INDEPENDENT: Each worker samples independently (single device default)
- SHARDED: Workers sample from disjoint shards (distributed default, ensures "single Poisson")
"""

from collections.abc import Iterator

import numpy as np

from opaque.sampling.poisson import PoissonSampler
from opaque.sampling.types import SamplingMode


class TruncatedPoissonSampler(PoissonSampler):
    """Truncated Poisson sampler with maximum batch size.

    Like PoissonSampler, but caps batch size at `max_batch_size`. This provides:
    1. Privacy amplification from Poisson subsampling
    2. Bounded batch size for stability and memory constraints
    3. Tighter privacy accounting via truncated Poisson analysis

    Supports distributed training with automatic environment detection:
    - **Auto-detection**: Automatically detects distributed environment from RANK/WORLD_SIZE env vars
    - **INDEPENDENT**: Single device training (default for world_size=1)
    - **SHARDED**: Workers sample from disjoint shards (default for distributed, ensures "single Poisson")

    Args:
        data_source: Dataset to sample from (any object with __len__)
        sample_rate: Probability of including each example (0 < p <= 1)
        max_batch_size: Maximum batch size (caps Poisson samples)
        num_epochs: Number of epochs to iterate over
        generator: Optional numpy random generator for reproducibility
        mode: Sampling mode for distributed training. If None (default), automatically
            selects SHARDED for distributed (world_size > 1) or INDEPENDENT for single device

    Example:
        >>> dataset = MyDataset(...)
        >>> sampler = TruncatedPoissonSampler(
        ...     dataset,
        ...     sample_rate=0.01,
        ...     max_batch_size=128,
        ...     num_epochs=10,
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>>
        >>> for batch in loader:
        ...     # Batch size is variable but capped at 128
        ...     assert len(batch) <= 128
        ...     pass

    Note:
        - Provides tighter privacy bounds than standard Poisson
        - Use opaque.accounting.compose_truncated_poisson_gaussian() for accounting
        - If sample_rate * len(dataset) << max_batch_size, rarely truncates
        - **Auto mode selection**: Detects distributed env and uses SHARDED by default
        - In SHARDED mode, each worker truncates its shard sample independently
    """

    def __init__(
        self,
        data_source,
        sample_rate: float,
        max_batch_size: int,
        num_epochs: int = 1,
        generator: np.random.Generator | None = None,
        mode: SamplingMode | None = None,
    ):
        super().__init__(data_source, sample_rate, num_epochs, generator, mode)

        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")

        self.max_batch_size = max_batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches capped at max_batch_size.

        Calls parent's Poisson sampling, then truncates if needed by randomly
        selecting max_batch_size examples from the Poisson sample.

        Returns:
            Iterator yielding lists of indices (variable size, capped)
        """
        # Use parent's Poisson sampling
        for indices in super().__iter__():
            # Truncate if needed
            if len(indices) > self.max_batch_size:
                # Randomly select max_batch_size examples (uniform from Poisson sample)
                indices_array = np.array(indices)
                indices = self.generator.choice(
                    indices_array,
                    size=self.max_batch_size,
                    replace=False,
                ).tolist()

            yield indices
