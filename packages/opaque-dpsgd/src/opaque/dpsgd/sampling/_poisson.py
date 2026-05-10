"""Poisson subsampling for DP-SGD training.

Poisson subsampling: each example is independently included with
probability ``sample_rate``.  Optional ``truncated_batch_size`` caps the
realised batch for stable sizes and memory; that is **weaker** for privacy than
plain Poisson at the same ``sample_rate``—pair with
:func:`opaque.dpsgd.accounting.poisson` passing both
``truncated_batch_size`` and ``dataset_size``.

For distributed training, shard the dataset **before** creating the
sampler using ``local_shard()`` and derive a per-rank key with
``fold_in(key, rank)``.
"""

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from opaque._poisson_impl import plain_poisson_step_indices
from opaque.random.types import RngKey


class PoissonSubsampler(Sampler):
    """Poisson subsampler for privacy amplification (DP-SGD).

    Each example is independently included with probability
    ``sample_rate``.  When ``truncated_batch_size`` is set, batches are
    capped at that size (uniform without replacement from the Poisson
    sample).

    For distributed training, shard the dataset externally and pass a
    per-rank key via ``fold_in(key, rank)``::

        from opaque.distributed import local_shard

        shard = local_shard(dataset, rank=rank, world_size=world_size)
        sampler = PoissonSubsampler(
            shard, sample_rate=0.01, key=fold_in(key(42), rank)
        )

    Args:
        data_source: Dataset to sample from (any object with ``__len__``).
        sample_rate: Probability of including each example ``∈ (0, 1]``.
        n_steps: Number of batches to yield. ``None`` yields indefinitely.
        truncated_batch_size: Optional cap on per-step batch size (truncated
            Poisson; use matching accounting—privacy is weaker than uncapped
            Poisson at the same ``sample_rate``).
        key: RNG key for reproducibility.

    Example::

        from opaque.random import key
        sampler = PoissonSubsampler(
            dataset, sample_rate=0.01, n_steps=1000, key=key(42),
        )
        loader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        - Batch sizes are variable (Poisson property).
        - Expected batch size: ``len(data_source) * sample_rate`` (uncapped).
        - Use with ``DataLoader``'s ``batch_sampler`` parameter.
    """

    def __init__(
        self,
        data_source: object,
        sample_rate: float,
        n_steps: int | None = None,
        truncated_batch_size: int | None = None,
        *,
        key: RngKey,
    ):
        super().__init__()

        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if n_steps is not None and n_steps < 1:
            raise ValueError(f"n_steps must be >= 1 or None, got {n_steps}")
        if truncated_batch_size is not None and truncated_batch_size < 1:
            raise ValueError(
                f"truncated_batch_size must be >= 1, got {truncated_batch_size}"
            )

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.truncated_batch_size = truncated_batch_size

        self._num_samples = len(data_source)

        self.generator = np.random.default_rng(key.seed)

    def _sample_step(self) -> list[int]:
        return plain_poisson_step_indices(
            self.generator,
            self._num_samples,
            self.sample_rate,
            self.truncated_batch_size,
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches as lists of indices.

        Each iteration samples the entire dataset using Poisson subsampling,
        optionally truncated.
        """
        if self.n_steps is None:
            while True:
                yield self._sample_step()
        else:
            for _ in range(self.n_steps):
                yield self._sample_step()

    def __len__(self) -> int:
        """Return number of batches.

        Raises:
            TypeError: If n_steps is None (infinite iteration).
        """
        if self.n_steps is None:
            raise TypeError("len() of unsized object (n_steps=None)")
        return self.n_steps

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size = num_samples * sample_rate (before truncation)."""
        return self._num_samples * self.sample_rate

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size for Poisson sampling (before truncation)."""
        return self._num_samples * self.sample_rate * (1 - self.sample_rate)
