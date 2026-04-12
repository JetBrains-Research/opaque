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


def _momentum_workload_coef(
    momentum: float,
    n: int,
    lr_schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Toeplitz workload coefficients for momentum-SGD + LR schedule.

    For momentum β and per-step learning rates η_t, the workload matrix W
    has entries W[t,s] = η_t · β^{t-s} for s ≤ t.  The Toeplitz
    coefficients are [η_0, η_1·β, η_2·β², ...].

    When ``lr_schedule=None``, assumes constant η=1 everywhere (the
    original behavior).

    Special cases:
        β = 0.0 → [η_0, 0, 0, ...] (identity workload)
        β = 0.95, lr=None → [1, 0.95, 0.9025, ...] (momentum-SGD)
        β = 1.0, lr=None → [1, 1, 1, ...] (prefix-sum workload, true FTRL)

    Args:
        momentum: Polyak momentum β (must be >= 0).
        n: Number of steps.
        lr_schedule: Optional per-step learning rates, shape [n].
            If None, assumes constant LR (implicit η=1).

    Raises:
        ValueError: If momentum < 0 or lr_schedule has wrong length.
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
        if lr_schedule is not None:
            lr = torch.as_tensor(lr_schedule, dtype=torch.float64)
            coef[0] = lr[0]
        return coef

    base = torch.tensor(
        [momentum**i for i in range(n)], dtype=torch.float64
    )

    if lr_schedule is not None:
        lr = torch.as_tensor(lr_schedule, dtype=torch.float64)
        if lr.shape[0] != n:
            raise ValueError(
                f"lr_schedule length ({lr.shape[0]}) must equal n ({n})"
            )
        return lr * base

    return base


def band_mf_noise(
    grad_template: Any,
    n_steps: int,
    *,
    stddev: float,
    key: RngKey,
    bands: int | None = None,
    momentum: float,
    lr_schedule: torch.Tensor | None = None,
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
        lr_schedule: Optional per-step learning rate schedule, shape [n_steps].
            When provided, the workload becomes ``[η₀, η₁·β, η₂·β², ...]``
            so the noise is optimized for the actual LR trajectory
            (warmup + constant + cosine cooldown) rather than constant LR.

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

    workload_coef = _momentum_workload_coef(momentum, n_steps, lr_schedule=lr_schedule)

    coefs = optimize_toeplitz(n_steps, bands, workload_coef=workload_coef)
    noising = inverse_as_streaming_matrix(coefs)
    return _matrix_factorization_noise(
        grad_template,
        noising,
        stddev=stddev,
        key=key,
    )


__all__ = ["band_mf_noise"]
