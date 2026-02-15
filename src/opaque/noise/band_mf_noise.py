"""BandMF correlated noise mechanism.

Convenience wrapper that optimizes banded Toeplitz coefficients and returns
ready-to-use ``(init_fn, noise_fn)`` for DP-FTRL training.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    matrix_factorization_noise,
)
from opaque.noise.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
    optimize_banded_toeplitz,
)


def band_mf_noise(
    n_steps: int,
    *,
    stddev: float,
    seed: int | None = None,
    bands: int | None = None,
) -> tuple[
    Callable[[Any], MFNoiseState],
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
]:
    """Create a BandMF correlated noise mechanism.

    Optimizes banded Toeplitz coefficients for ``n_steps`` iterations,
    then wraps the result in the matrix factorization noise API.

    Args:
        n_steps: Number of training iterations.
        stddev: Standard deviation for the base noise.
        seed: Optional random seed for reproducibility.
        bands: Number of bands in the Toeplitz matrix. Defaults to
            ``n_steps`` (full band, equivalent to optimal Fichtenberger init).

    Returns:
        A tuple ``(init_fn, noise_fn)`` where:

        - ``state = init_fn(grad_template)``
        - ``noisy_grads, new_state = noise_fn(clipped_grads, state)``

    Example:
        >>> init_fn, noise_fn = band_mf_noise(1000, stddev=1.0, seed=42, bands=10)
        >>> state = init_fn(grad_template)
        >>> for step in range(1000):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    if bands is None:
        bands = n_steps

    coefs = optimize_banded_toeplitz(n_steps, bands)
    noising = inverse_as_streaming_matrix(coefs)
    return matrix_factorization_noise(noising, stddev=stddev, seed=seed)


__all__ = ["band_mf_noise"]
