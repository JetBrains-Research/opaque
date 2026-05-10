"""Shared helpers for Poisson-style subsampling samplers (private)."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def maybe_truncate_indices(
    rng: Generator,
    indices: np.ndarray,
    truncated_batch_size: int | None,
) -> list[int]:
    """Cap ``indices`` to ``truncated_batch_size`` via uniform subsample."""
    if truncated_batch_size is not None and indices.size > truncated_batch_size:
        indices = rng.choice(indices, size=truncated_batch_size, replace=False)
    return indices.tolist()


def plain_poisson_step_indices(
    rng: Generator,
    num_examples: int,
    sample_rate: float,
    truncated_batch_size: int | None,
) -> list[int]:
    """Bernoulli mask over ``0..num_examples-1`` (DP-SGD standard Poisson)."""
    included = rng.random(num_examples) < sample_rate
    indices = np.where(included)[0]
    return maybe_truncate_indices(rng, indices, truncated_batch_size)
