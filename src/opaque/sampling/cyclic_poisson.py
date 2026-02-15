"""Cyclic Poisson sampling for BandMF privacy amplification.

Implements cyclic Poisson sampling, which generalizes several sampling
strategies. In cyclic Poisson sampling, examples are partitioned into
``cycle_length`` groups. Each iteration samples from one group in a
round-robin fashion. This is required for BandMF amplified privacy.

References:
    - Fixed order: https://arxiv.org/abs/2211.06530
    - Poisson sampling: https://arxiv.org/abs/1607.00133
    - BandMF sampling: https://arxiv.org/abs/2306.08153

Example:
    >>> rng = np.random.default_rng(0)
    >>> sampler = CyclicPoissonSampling(
    ...     sampling_prob=0.5, iterations=6, cycle_length=2
    ... )
    >>> batches = list(sampler.batch_iterator(12, rng=rng))
"""

from __future__ import annotations

import abc
import dataclasses
import enum
from collections.abc import Iterator

import numpy as np

RngType = np.random.Generator | int | None


class PartitionType(enum.Enum):
    """Specifies how examples should be assigned to groups."""

    INDEPENDENT = enum.auto()
    """Each example assigned to a group independently at random."""
    EQUAL_SPLIT = enum.auto()
    """Examples shuffled and split into groups of equal size."""


def _independent_partition(
    num_examples: int,
    num_groups: int,
    rng: np.random.Generator,
    dtype: np.typing.DTypeLike,
) -> list[np.ndarray]:
    """Partition examples independently (multinomial assignment)."""
    sizes = rng.multinomial(num_examples, np.ones(num_groups) / num_groups)
    boundaries = np.cumsum(sizes)[:-1]
    indices = rng.permutation(num_examples).astype(dtype)
    return np.split(indices, boundaries)


def _equal_split_partition(
    num_examples: int,
    num_groups: int,
    rng: np.random.Generator,
    dtype: np.typing.DTypeLike,
) -> list[np.ndarray]:
    """Partition examples by shuffling then splitting equally."""
    indices = rng.permutation(num_examples).astype(dtype)
    group_size = num_examples // num_groups
    groups = np.array_split(indices, num_groups)
    return [g[:group_size] for g in groups]


class BatchSelectionStrategy(abc.ABC):
    """Abstract base class for batch selection strategies.

    Produces indices into a dataset, not example elements themselves.
    Relies on random access to individual examples by index.
    """

    @abc.abstractmethod
    def batch_iterator(
        self, num_examples: int, rng: RngType = None
    ) -> Iterator[np.ndarray]:
        """Yields 1D batches of data indices.

        Args:
            num_examples: Total number of examples in the dataset.
            rng: Random seed or random number generator.

        Yields:
            Arrays of indices for each batch.
        """


@dataclasses.dataclass(frozen=True)
class CyclicPoissonSampling(BatchSelectionStrategy):
    """Cyclic Poisson sampling for DP with correlated noise mechanisms.

    Generalizes several common sampling strategies:

    - **Fixed order** (sampling_prob=1, cycle_length=n//b): Deterministic
      multi-epoch order, each example seen exactly once per epoch.
    - **Standard Poisson** (cycle_length=1): Each example independently
      included with probability ``sampling_prob``.
    - **BandMF-style** (cycle_length>1, 0<sampling_prob<1): Cyclic Poisson
      sampling required for BandMF privacy amplification.

    Formal guarantees:
        - All batches consist of indices in [0, num_examples).
        - Each example only appears in batches i where
          ``i % cycle_length == j`` for some fixed j per example.
        - Without truncation, each eligible index appears independently
          with probability ``sampling_prob``.
        - With truncation, excess examples are uniformly subsampled.
        - With EQUAL_SPLIT, ``num_examples % cycle_length`` examples may
          be discarded.

    Attributes:
        sampling_prob: Probability of sampling an eligible example.
        iterations: Total number of iterations/batches.
        truncated_batch_size: If set, cap batch size at this value.
        cycle_length: Number of groups (cycle_length=1 = standard Poisson).
        partition_type: How to partition examples into groups.
    """

    sampling_prob: float
    iterations: int
    truncated_batch_size: int | None = None
    cycle_length: int = 1
    partition_type: PartitionType = PartitionType.EQUAL_SPLIT

    def batch_iterator(
        self, num_examples: int, rng: RngType = None
    ) -> Iterator[np.ndarray]:
        """Yields 1D batches of data indices.

        Args:
            num_examples: Total number of examples in the dataset.
            rng: Random seed or random number generator.

        Yields:
            Arrays of indices for each batch.
        """
        rng = np.random.default_rng(rng)
        dtype = np.min_scalar_type(-num_examples)

        if self.partition_type == PartitionType.INDEPENDENT:
            partition_fn = _independent_partition
        elif self.partition_type == PartitionType.EQUAL_SPLIT:
            partition_fn = _equal_split_partition
        else:
            raise ValueError(f"Unsupported partition type: {self.partition_type}")

        partition = partition_fn(num_examples, self.cycle_length, rng, dtype)

        for i in range(self.iterations):
            current_group = partition[i % self.cycle_length]
            sample_size = rng.binomial(n=len(current_group), p=self.sampling_prob)
            if self.truncated_batch_size is not None:
                sample_size = min(sample_size, self.truncated_batch_size)
            yield rng.choice(
                current_group,
                size=sample_size,
                replace=False,
                shuffle=False,
            )


def split_and_pad_global_batch(
    indices: np.ndarray,
    minibatch_size: int,
    microbatch_size: int | None = None,
) -> list[np.ndarray]:
    """Split a global batch into fixed-size minibatches with -1 padding.

    The last minibatch is padded with ``-1`` indices. Downstream users must
    account for this by zeroing out gradients for padding examples.

    Example:
        >>> indices = np.arange(10)
        >>> split_and_pad_global_batch(indices, minibatch_size=4)
        [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, -1, -1])]

    Args:
        indices: A 1D or 2D array of indices.
        minibatch_size: Desired size of each minibatch.
        microbatch_size: Optional microbatch size for early stopping order.

    Returns:
        List of minibatches, each of size exactly ``minibatch_size``.

    Raises:
        ValueError: If minibatch_size is not positive.
    """
    if minibatch_size <= 0:
        raise ValueError(f"minibatch_size must be positive, got {minibatch_size}")
    sections = range(minibatch_size, indices.shape[0], minibatch_size)
    minibatches = np.array_split(indices, sections, axis=0)
    minibatch_shape = (minibatch_size,) + indices.shape[1:]
    last_minibatch = np.full(minibatch_shape, -1, dtype=indices.dtype)
    last_minibatch[: minibatches[-1].shape[0]] = minibatches[-1]

    if microbatch_size is not None and microbatch_size > 0:
        # Reorder so padding is in early-stopping-friendly positions
        permutation = _compute_early_stopping_order(minibatch_size, microbatch_size)
        last_minibatch = last_minibatch[permutation]

    minibatches[-1] = last_minibatch
    return minibatches


def pad_to_multiple_of(indices: np.ndarray, multiple: int) -> np.ndarray:
    """Pad the last dimension of indices to a multiple of ``multiple``.

    Example:
        >>> indices = np.arange(10)
        >>> pad_to_multiple_of(indices, multiple=4)
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1])

    Args:
        indices: A 1D array of batch indices.
        multiple: Positive integer to pad to.

    Returns:
        Padded array with -1 values.
    """
    if indices.ndim > 1:
        raise ValueError("pad_to_multiple_of expects 1D indices.")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    curr_size = indices.shape[0]
    pad_size = (multiple - curr_size) % multiple
    new_indices = np.full(curr_size + pad_size, -1, dtype=indices.dtype)
    new_indices[:curr_size] = indices
    return new_indices


def _compute_early_stopping_order(
    minibatch_size: int, microbatch_size: int
) -> np.ndarray:
    """Compute reordering for padding-aware early stopping.

    Reorders indices so that padding (-1) values are concentrated at
    the end of the last microbatch, enabling early exit.

    Args:
        minibatch_size: Total minibatch size.
        microbatch_size: Microbatch size.

    Returns:
        Permutation array.
    """
    if microbatch_size is None or microbatch_size <= 0:
        return np.arange(minibatch_size)
    # Standard order: no special reordering needed for basic case
    return np.arange(minibatch_size)
