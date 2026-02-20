"""BandMF correlated noise mechanism.

Convenience wrapper that optimizes banded Toeplitz coefficients and returns
ready-to-use ``(noise_fn, state)`` for DP-FTRL training.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.gaussian_noise import _create_rng_state
from opaque.random import RngKey
from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    _matrix_factorization_noise,
)
from opaque.noise.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
)
from opaque.noise.matrix_factorization.toeplitz import (
    optimize as optimize_toeplitz,
)


def band_mf_noise(
    grad_template: Any,
    n_steps: int,
    *,
    stddev: float,
    key: RngKey,
    synchronized: str | bool = "auto",
    bands: int | None = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a BandMF correlated noise mechanism.

    Optimizes banded Toeplitz coefficients for ``n_steps`` iterations,
    then wraps the result in the matrix factorization noise API.

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
        bands: Number of bands in the Toeplitz matrix. Defaults to
            ``n_steps`` (full band, equivalent to optimal Fichtenberger init).

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> from opaque.random import key
        >>> noise_fn, state = band_mf_noise(grad_template, 1000, stddev=1.0, key=key(42), bands=10)
        >>> for step in range(1000):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    if bands is None:
        bands = n_steps

    coefs = optimize_toeplitz(n_steps, bands)
    noising = inverse_as_streaming_matrix(coefs)
    gen, resolved_seed, is_sync = _create_rng_state(key, synchronized)
    return _matrix_factorization_noise(
        grad_template,
        noising,
        stddev=stddev,
        gen=gen,
        seed=resolved_seed,
        synchronized=is_sync,
    )


__all__ = ["band_mf_noise"]
