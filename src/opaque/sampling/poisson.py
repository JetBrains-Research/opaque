"""Poisson samplers for differential privacy.

These samplers implement Poisson subsampling, where each example in the dataset
is independently included in a batch with probability `sample_rate`. This provides
privacy amplification, reducing the privacy cost compared to fixed-batch sampling.
"""

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class PoissonSampler(Sampler):
    """Poisson sampler for privacy amplification.

    Each example in the dataset is independently included with probability
    `sample_rate`. This creates variable-sized batches, which provides privacy
    amplification: the effective privacy cost is reduced by approximately
    √(1/sample_rate) compared to uniform sampling.

    Args:
        data_source: Dataset to sample from (any object with __len__)
        sample_rate: Probability of including each example (0 < p <= 1)
        num_epochs: Number of epochs to iterate over
        generator: Optional numpy random generator for reproducibility

    Example:
        >>> dataset = MyDataset(...)
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, num_epochs=10)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>>
        >>> for batch in loader:
        ...     # batch has variable size!
        ...     # Expected size: len(dataset) * sample_rate
        ...     pass

    Note:
        - Batch sizes are variable (Poisson property)
        - Expected batch size: len(dataset) * sample_rate
        - Variance: len(dataset) * sample_rate * (1 - sample_rate)
        - Use with DataLoader's batch_sampler parameter (not sampler)
    """

    def __init__(
        self,
        data_source,
        sample_rate: float,
        num_epochs: int = 1,
        generator: np.random.Generator | None = None,
    ):
        super().__init__(data_source)

        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.num_epochs = num_epochs
        self.generator = generator if generator is not None else np.random.default_rng()

        self._num_samples = len(data_source)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches as lists of indices.

        Each call samples the entire dataset once per epoch using Poisson
        subsampling. Examples are included independently with probability
        sample_rate.

        Returns:
            Iterator yielding lists of indices (variable size)
        """
        for _ in range(self.num_epochs):
            # Each example included independently with probability p
            included = self.generator.random(self._num_samples) < self.sample_rate
            indices = np.where(included)[0]

            # Yield as list (DataLoader expects list)
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


class TruncatedPoissonSampler(PoissonSampler):
    """Truncated Poisson sampler with maximum batch size.

    Like PoissonSampler, but caps batch size at `max_batch_size`. This provides:
    1. Privacy amplification from Poisson subsampling
    2. Bounded batch size for stability and memory constraints
    3. Tighter privacy accounting via truncated Poisson analysis

    Args:
        data_source: Dataset to sample from (any object with __len__)
        sample_rate: Probability of including each example (0 < p <= 1)
        max_batch_size: Maximum batch size (caps Poisson samples)
        num_epochs: Number of epochs to iterate over
        generator: Optional numpy random generator for reproducibility

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
    """

    def __init__(
        self,
        data_source,
        sample_rate: float,
        max_batch_size: int,
        num_epochs: int = 1,
        generator: np.random.Generator | None = None,
    ):
        super().__init__(data_source, sample_rate, num_epochs, generator)

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
