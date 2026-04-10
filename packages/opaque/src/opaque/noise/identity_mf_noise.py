"""Identity matrix noise mechanism (DP-SGD via MF API).

Thin wrapper around ``custom_mf_noise`` using the identity matrix as the
noising matrix. This is equivalent to standard DP-SGD (independent noise
at each step) but expressed through the matrix factorization API, making
it easy to swap in correlated noise mechanisms later.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.noise.custom_mf_noise import custom_mf_noise
from opaque.noise.matrix_factorization.noise import MFNoiseState
from opaque.noise.matrix_factorization.streaming_matrix import identity
from opaque.random import RngKey


def identity_mf_noise(
    grad_template: Any,
    *,
    stddev: float,
    key: RngKey,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create an identity (DP-SGD) noise mechanism via the MF API.

    This adds independent Gaussian noise at each step, equivalent to
    standard DP-SGD. Use this as a drop-in baseline that can be swapped
    for correlated noise mechanisms (``band_mf_noise``)
    without changing the training loop.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank.

    Args:
        grad_template: A pytree with the same structure and shapes as the
            gradients that will be passed to ``noise_fn``.
        stddev: Standard deviation for the base noise.
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> from opaque.random import key
        >>> noise_fn, state = identity_mf_noise(grad_template, stddev=1.0, key=key(42))
        >>> for step in range(100):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    return custom_mf_noise(
        grad_template,
        identity(),
        stddev=stddev,
        key=key,
    )


__all__ = ["identity_mf_noise"]
