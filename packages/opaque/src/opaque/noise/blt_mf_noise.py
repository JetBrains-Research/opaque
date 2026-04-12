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

from opaque.noise.band_mf_noise import _momentum_workload_coef
from opaque.noise.matrix_factorization.buffered_toeplitz import (
    inverse_as_streaming_matrix,
    optimize,
)
from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    _matrix_factorization_noise,
)
from opaque.random import RngKey


def blt_mf_noise(
    grad_template: Any,
    n_steps: int,
    *,
    stddev: float,
    key: RngKey,
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = "max",
    max_buffers: int = 10,
    momentum: float,
    lr_schedule: torch.Tensor | None = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a BLT correlated noise mechanism.

    Optimizes BLT parameters for ``n_steps`` iterations with the given
    participation pattern, then wraps the result in the matrix
    factorization noise API.

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
        min_sep: Minimum separation between participations (default 1).
        max_participations: Maximum participations per user (default 1).
        error: Error metric to optimize: ``'max'`` or ``'mean'``.
        max_buffers: Maximum number of BLT buffers to try (default 10).
        momentum: Polyak momentum coefficient (must be >= 0).
            Determines the optimizer workload ``[1, β, β², ...]``.
            Use β=1.0 for prefix-sum (true FTRL), β<1 for momentum-SGD.
            β=0.0 is allowed for testing (identity workload, equivalent to
            independent noise) but emits a warning.
        lr_schedule: Optional per-step learning rate schedule, shape [n_steps].
            When provided, the workload becomes ``[η₀, η₁·β, η₂·β², ...]``
            so the noise is optimized for the actual LR trajectory.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> from opaque.random import key
        >>> noise_fn, state = blt_mf_noise(
        ...     grad_template, 1000, stddev=1.0, key=key(42), momentum=0.9,
        ... )
        >>> for step in range(1000):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    workload_coef = _momentum_workload_coef(momentum, n_steps, lr_schedule=lr_schedule)

    blt = optimize(
        n=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        error=error,
        max_buffers=max_buffers,
        workload_coef=workload_coef,
    )
    noising = inverse_as_streaming_matrix(blt)
    return _matrix_factorization_noise(
        grad_template,
        noising,
        stddev=stddev,
        key=key,
    )


__all__ = ["blt_mf_noise"]
