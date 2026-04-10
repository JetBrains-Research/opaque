"""BandMF correlated noise mechanism.

Convenience wrapper that optimizes banded Toeplitz coefficients and returns
ready-to-use ``(noise_fn, state)`` for DP-FTRL training.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch

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
from opaque.random import RngKey


def _momentum_workload_coef(momentum: float, n: int) -> torch.Tensor:
    """Compute Toeplitz workload coefficients for momentum-SGD.

    For momentum β, the workload matrix W has entries W[t,s] = β^{t-s}
    for s ≤ t.  The Toeplitz coefficients are [1, β, β², ...].

    Special cases:
        β = 0.0 → [1, 0, 0, ...] (identity workload, equivalent to DP-SGD)
        β = 0.95 → [1, 0.95, 0.9025, ...] (momentum-SGD workload)
        β = 1.0 → [1, 1, 1, ...] (prefix-sum workload, true FTRL)

    Raises:
        ValueError: If momentum < 0.
    """
    if momentum < 0:
        raise ValueError(f"momentum must be >= 0, got {momentum}")
    if momentum == 0.0:
        warnings.warn(
            "momentum=0.0 produces an identity workload — MF noise will "
            "reduce to independent noise with no benefit over standard "
            "Gaussian (DP-SGD). This is useful for testing but not for "
            "production training.",
            stacklevel=3,
        )
        coef = torch.zeros(n, dtype=torch.float64)
        coef[0] = 1.0
        return coef
    return torch.tensor(
        [momentum**i for i in range(n)], dtype=torch.float64
    )


def band_mf_noise(
    grad_template: Any,
    n_steps: int,
    *,
    stddev: float,
    key: RngKey,
    bands: int | None = None,
    momentum: float,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a BandMF correlated noise mechanism.

    Optimizes banded Toeplitz coefficients for ``n_steps`` iterations,
    then wraps the result in the matrix factorization noise API.

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
        bands: Number of bands in the Toeplitz matrix. Defaults to
            ``n_steps`` (full band, equivalent to optimal Fichtenberger init).
        momentum: Polyak momentum coefficient (must be >= 0).
            Determines the optimizer workload ``[1, β, β², ...]``.
            Use β=1.0 for prefix-sum (true FTRL), β<1 for momentum-SGD.
            β=0.0 is allowed for testing (identity workload, equivalent to
            independent noise) but emits a warning.

    Raises:
        ValueError: If ``momentum < 0``.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> from opaque.random import key
        >>> noise_fn, state = band_mf_noise(
        ...     grad_template, 1000, stddev=1.0, key=key(42), bands=10, momentum=0.95,
        ... )
        >>> for step in range(1000):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    if bands is None:
        bands = n_steps

    workload_coef = _momentum_workload_coef(momentum, n_steps)

    coefs = optimize_toeplitz(n_steps, bands, workload_coef=workload_coef)
    noising = inverse_as_streaming_matrix(coefs)
    return _matrix_factorization_noise(
        grad_template,
        noising,
        stddev=stddev,
        key=key,
    )


__all__ = ["band_mf_noise"]
