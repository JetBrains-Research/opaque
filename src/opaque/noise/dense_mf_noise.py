"""Dense matrix correlated noise mechanism.

Convenience wrapper that optimizes a dense strategy matrix and returns
ready-to-use ``(noise_fn, state)`` for DP-FTRL training.

References:
    - Denisov et al., 2022: https://arxiv.org/abs/2202.08312
    - Choquette-Choo et al., 2022: https://arxiv.org/abs/2211.06530
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.matrix_factorization.dense import optimize
from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    _matrix_factorization_noise,
)
from opaque.random import RngKey


def dense_mf_noise(
    grad_template: Any,
    n_steps: int,
    *,
    stddev: float,
    key: RngKey,
    epochs: int = 1,
    bands: int | None = None,
    equal_norm: bool = False,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a dense matrix correlated noise mechanism.

    Optimizes a dense strategy matrix C for ``n_steps`` iterations,
    inverts it, then wraps the result in the matrix factorization noise API.

    Note: Dense optimization materializes the full n x n matrix. For large
    ``n_steps``, prefer ``band_mf_noise`` or ``blt_mf_noise`` which use
    streaming representations.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank.

    Args:
        grad_template: A pytree with the same structure and shapes as the
            gradients that will be passed to ``noise_fn``.
        n_steps: Number of training iterations.
        stddev: Standard deviation for the base noise.
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.
        epochs: Number of epochs (for fixed-epoch participation).
        bands: Number of bands in the strategy (for banded optimization).
        equal_norm: If True, optimize with equal column norm constraint.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> from opaque.random import key
        >>> noise_fn, state = dense_mf_noise(grad_template, 100, stddev=1.0, key=key(42))
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
    return _matrix_factorization_noise(
        grad_template,
        noising_matrix,
        stddev=stddev,
        key=key,
    )


__all__ = ["dense_mf_noise"]
