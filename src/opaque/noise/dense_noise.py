"""Dense matrix correlated noise mechanism.

Convenience wrapper that optimizes a dense strategy matrix and returns
ready-to-use ``(init_fn, noise_fn)`` for DP-FTRL training.

References:
    - Denisov et al., 2022: https://arxiv.org/abs/2202.08312
    - Choquette-Choo et al., 2022: https://arxiv.org/abs/2211.06530
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.matrix_factorization.dense import optimize
from opaque.noise.matrix_factorization.noise import MFNoiseState, matrix_factorization_noise


def dense_noise(
    n_steps: int,
    *,
    stddev: float,
    seed: int | None = None,
    epochs: int = 1,
    bands: int | None = None,
    equal_norm: bool = False,
) -> tuple[
    Callable[[Any], MFNoiseState],
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
]:
    """Create a dense matrix correlated noise mechanism.

    Optimizes a dense strategy matrix C for ``n_steps`` iterations,
    inverts it, then wraps the result in the matrix factorization noise API.

    Note: Dense optimization materializes the full n x n matrix. For large
    ``n_steps``, prefer ``band_mf_noise`` or ``blt_noise`` which use
    streaming representations.

    Args:
        n_steps: Number of training iterations.
        stddev: Standard deviation for the base noise.
        seed: Optional random seed for reproducibility.
        epochs: Number of epochs (for fixed-epoch participation).
        bands: Number of bands in the strategy (for banded optimization).
        equal_norm: If True, optimize with equal column norm constraint.

    Returns:
        A tuple ``(init_fn, noise_fn)`` where:

        - ``state = init_fn(grad_template)``
        - ``noisy_grads, new_state = noise_fn(clipped_grads, state)``

    Example:
        >>> init_fn, noise_fn = dense_noise(100, stddev=1.0, seed=42)
        >>> state = init_fn(grad_template)
        >>> for step in range(100):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    strategy_matrix = optimize(
        n_steps,
        epochs=epochs,
        bands=bands,
        equal_norm=equal_norm,
    )
    noising_matrix = torch.linalg.solve(
        strategy_matrix, torch.eye(n_steps, dtype=strategy_matrix.dtype)
    )
    return matrix_factorization_noise(noising_matrix, stddev=stddev, seed=seed)


__all__ = ["dense_noise"]
