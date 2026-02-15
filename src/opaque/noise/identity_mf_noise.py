"""Identity matrix noise mechanism (DP-SGD via MF API).

Thin wrapper around ``custom_mf_noise`` using the identity matrix as the
noising matrix. This is equivalent to standard DP-SGD (independent noise
at each step) but expressed through the matrix factorization API, making
it easy to swap in correlated noise mechanisms later.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.custom_mf_noise import custom_mf_noise
from opaque.noise.matrix_factorization.noise import MFNoiseState
from opaque.noise.matrix_factorization.streaming_matrix import identity


def identity_mf_noise(
    grad_template: Any,
    *,
    stddev: float,
    generator: None | int | torch.Generator = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create an identity (DP-SGD) noise mechanism via the MF API.

    This adds independent Gaussian noise at each step, equivalent to
    standard DP-SGD. Use this as a drop-in baseline that can be swapped
    for correlated noise mechanisms (``band_mf_noise``, ``blt_mf_noise``,
    ``dense_mf_noise``) without changing the training loop.

    Args:
        grad_template: A pytree with the same structure and shapes as the
            gradients that will be passed to ``noise_fn``.
        stddev: Standard deviation for the base noise.
        generator: RNG configuration:
            - ``None``: new unseeded generator (non-reproducible)
            - ``int``: seeded generator (reproducible)
            - ``torch.Generator``: use directly

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> noise_fn, state = identity_mf_noise(grad_template, stddev=1.0, generator=42)
        >>> for step in range(100):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    return custom_mf_noise(
        grad_template,
        identity(),
        stddev=stddev,
        generator=generator,
    )


__all__ = ["identity_mf_noise"]
