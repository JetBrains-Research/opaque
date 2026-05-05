"""Truncated Poisson sampler for differential privacy.

This module provides TruncatedPoissonSampler, which combines Poisson subsampling
for privacy amplification with a maximum batch size constraint for stability.

For distributed training, shard the dataset externally (same as PoissonSampler).
"""

from collections.abc import Iterator

import numpy as np

from opaque.random.types import RngKey
from opaque.dpsgd.sampling._poisson import PoissonSampler


class TruncatedPoissonSampler(PoissonSampler):
    """Truncated Poisson sampler with maximum batch size.

    Like PoissonSampler, but caps batch size at ``max_batch_size``. This provides:
    1. Privacy amplification from Poisson subsampling
    2. Bounded batch size for stability and memory constraints
    3. Tighter privacy accounting via truncated Poisson analysis

    Args:
        data_source: Dataset to sample from (any object with ``__len__``)
        sample_rate: Probability of including each example (0 < p ≤ 1)
        max_batch_size: Maximum batch size (caps Poisson samples)
        num_iterations: Number of batches to yield. If None, yields batches indefinitely.
        key: RNG key for reproducibility. Use ``key()`` or ``fold_in()``.

    Example:
        >>> from opaque.random import key
        >>> dataset = MyDataset(...)
        >>> sampler = TruncatedPoissonSampler(
        ...     dataset,
        ...     sample_rate=0.01,
        ...     max_batch_size=128,
        ...     num_iterations=10,
        ...     key=key(42),
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        - Provides tighter privacy bounds than standard Poisson.
        - Use ``opaque.accounting.compose_truncated_poisson_gaussian()`` for accounting.
        - If ``sample_rate * len(dataset) << max_batch_size``, rarely truncates.
    """

    def __init__(
        self,
        data_source: object,
        sample_rate: float,
        max_batch_size: int,
        num_iterations: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__(
            data_source,
            sample_rate,
            num_iterations,
            key=key,
        )

        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")

        self.max_batch_size = max_batch_size

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["max_batch_size"] = int(self.max_batch_size)
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if "max_batch_size" in state:
            self.max_batch_size = int(state["max_batch_size"])

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches capped at ``max_batch_size``.

        Uses one per-iteration generator for both Poisson sampling and
        truncation, keeping randomness deterministic across save/resume.
        """
        while self.num_iterations is None or self._iter_count < self.num_iterations:
            gen = self._generator_for_iter(self._iter_count)
            included = gen.random(self._num_samples) < self.sample_rate
            indices = np.where(included)[0]
            if indices.size > self.max_batch_size:
                indices = gen.choice(indices, size=self.max_batch_size, replace=False)
            self._iter_count += 1
            yield indices.tolist()
