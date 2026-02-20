"""BLT (Buffered Linear Toeplitz) correlated noise mechanism.

Convenience wrapper that optimizes BLT parameters and returns
ready-to-use ``(noise_fn, state)`` for DP-FTRL training.

References:
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.gaussian_noise import _create_rng_state
from opaque.random import RngKey
from opaque.noise.matrix_factorization.buffered_toeplitz import (
    inverse_as_streaming_matrix,
    optimize,
)
from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    _matrix_factorization_noise,
)


def blt_mf_noise(
    grad_template: Any,
    n_steps: int,
    *,
    stddev: float,
    key: RngKey,
    synchronized: str | bool = "auto",
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = "max",
    max_buffers: int = 10,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a BLT correlated noise mechanism.

    Optimizes BLT parameters for ``n_steps`` iterations with the given
    participation pattern, then wraps the result in the matrix
    factorization noise API.

    Args:
        grad_template: A pytree with the same structure and shapes as the
            gradients that will be passed to ``noise_fn``.
        n_steps: Number of training iterations.
        stddev: Standard deviation for the base noise.
        key: Explicit RNG key for deterministic, functional randomness.
        synchronized: Synchronization mode for distributed training:
            - ``"auto"`` (default): Auto-detect and sync if distributed
            - ``True``: Force synchronized noise (same seed across devices)
            - ``False``: Independent noise per device (seed + rank offset)
        min_sep: Minimum separation between participations (default 1).
        max_participations: Maximum participations per user (default 1).
        error: Error metric to optimize: ``'max'`` or ``'mean'``.
        max_buffers: Maximum number of BLT buffers to try (default 10).

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> from opaque.random import key
        >>> noise_fn, state = blt_mf_noise(grad_template, 1000, stddev=1.0, key=key(42))
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
    noising = inverse_as_streaming_matrix(blt)
    gen, resolved_seed, is_sync = _create_rng_state(key, synchronized)
    return _matrix_factorization_noise(
        grad_template,
        noising,
        stddev=stddev,
        gen=gen,
        seed=resolved_seed,
        synchronized=is_sync,
    )


__all__ = ["blt_mf_noise"]
