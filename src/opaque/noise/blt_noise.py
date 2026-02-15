"""BLT (Buffered Linear Toeplitz) correlated noise mechanism.

Convenience wrapper that optimizes BLT parameters and returns
ready-to-use ``(init_fn, noise_fn)`` for DP-FTRL training.

References:
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.noise.matrix_factorization.buffered_toeplitz import optimize
from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    matrix_factorization_noise,
)


def blt_noise(
    n_steps: int,
    *,
    stddev: float,
    seed: int | None = None,
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = "max",
    max_buffers: int = 10,
) -> tuple[
    Callable[[Any], MFNoiseState],
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
]:
    """Create a BLT correlated noise mechanism.

    Optimizes BLT parameters for ``n_steps`` iterations with the given
    participation pattern, then wraps the result in the matrix
    factorization noise API.

    Args:
        n_steps: Number of training iterations.
        stddev: Standard deviation for the base noise.
        seed: Optional random seed for reproducibility.
        min_sep: Minimum separation between participations (default 1).
        max_participations: Maximum participations per user (default 1).
        error: Error metric to optimize: ``'max'`` or ``'mean'``.
        max_buffers: Maximum number of BLT buffers to try (default 10).

    Returns:
        A tuple ``(init_fn, noise_fn)`` where:

        - ``state = init_fn(grad_template)``
        - ``noisy_grads, new_state = noise_fn(clipped_grads, state)``

    Example:
        >>> init_fn, noise_fn = blt_noise(1000, stddev=1.0, seed=42)
        >>> state = init_fn(grad_template)
        >>> for step in range(1000):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    blt = optimize(
        n=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        error=error,
        max_buffers=max_buffers,
    )
    noising = blt.inverse_as_streaming_matrix()
    return matrix_factorization_noise(noising, stddev=stddev, seed=seed)


__all__ = ["blt_noise"]
