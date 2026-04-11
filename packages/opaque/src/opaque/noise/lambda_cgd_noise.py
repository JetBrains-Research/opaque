"""DP-λCGD noise mechanism — correlated noise via PRNG replay.

The DP-λCGD mechanism (Kalinin et al., 2026) uses a lower-triangular
Toeplitz strategy matrix C_λ whose inverse is bidiagonal: 1 on the
diagonal, -λ on the subdiagonal.  The correlated noise at step t is:

    ñ_t = z_t - λ · z_{t-1}          (unnormalized)
    ñ_t = d_t · (z_t - λ · z_{t-1})  (column-normalized, default)

where z_t ~ N(0, σ²I) are i.i.d. Gaussians, and d_t is the column norm
of C_λ at step t:  d_t = √((1 − λ^{2(n−t)}) / (1 − λ²)).

Column normalization (Appendix A of the paper) ensures all columns of
C̃_λ = C_λ · D⁻¹ have unit norm, enabling exact BnB privacy analysis
and strictly improved RMSE (Lemma 9).

Instead of storing z_{t-1}, we regenerate it from the previous step's
PRNG seed — zero additional memory overhead compared to DP-SGD.

References:
    - Kalinin et al. (2026) "DP-λCGD: Leveraging Correlated Gradients
      for Improved DP-SGD" https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import math

import torch

from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    _iid_normal_noise,
    _internal_compute_dtype,
)
from opaque.random import RngKey, generator_from_key
from opaque.random import fold_in as rng_fold_in
from opaque.utils.pytree import tree_map


def _column_norm(lambda_: float, n_steps: int, step: int) -> float:
    """Column norm d_t of C_λ at 0-indexed step t.

    d_t² = (1 − λ^{2(n−t)}) / (1 − λ²)  for the n×n strategy matrix.
    """
    if lambda_ == 0.0:
        return 1.0
    remaining = n_steps - step  # n − t
    lambda2 = lambda_ * lambda_
    lambda2r = lambda2 ** remaining
    if lambda2r < 1e-30:
        return math.sqrt(1.0 / (1.0 - lambda2))
    return math.sqrt((1.0 - lambda2r) / (1.0 - lambda2))


def lambda_cgd_noise(
    grad_template: Any,
    n_steps: int,
    *,
    stddev: float,
    key: RngKey,
    lambda_: float,
    normalized: bool = True,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a DP-λCGD noise mechanism via PRNG replay.

    At each step t, the noise function generates:

    - ``z_t`` from the current step's PRNG seed
    - ``z_{t-1}`` regenerated from the previous step's PRNG seed

    When ``normalized=True`` (default), column-normalized noise is applied:

        ñ_t = d_t · (z_t − λ · z_{t-1})

    where ``d_t = √((1 − λ^{2(n−t)}) / (1 − λ²))`` is the column norm
    of C_λ at step t.  This matches the column-normalized strategy matrix
    C̃_λ = C_λ · D⁻¹ from Appendix A of the paper.

    When ``normalized=False``:

        ñ_t = z_t − λ · z_{t-1}

    The z vectors are never stored — only the PRNG seed is kept.

    Args:
        grad_template: A pytree with the same structure and shapes as the
            gradients that will be passed to ``noise_fn``.
        n_steps: Total number of training steps.  Used to compute column
            norms for normalization.
        stddev: Standard deviation σ for the base i.i.d. noise.
        key: Explicit RNG key for deterministic, functional randomness.
        lambda_: Correlation coefficient in [0, 1). λ=0 is DP-SGD.
        normalized: If True (default), apply column-norm scaling d_t.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> from opaque.random import key
        >>> noise_fn, state = lambda_cgd_noise(
        ...     grad_template, n_steps=15000,
        ...     stddev=1.0, key=key(42), lambda_=0.9,
        ... )
        >>> for step in range(15000):
        ...     noisy_grads, state = noise_fn(clipped_grads, state)
    """
    if lambda_ < 0 or lambda_ >= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1), got {lambda_}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    # Inner state: the previous step counter (for regenerating z_{t-1}).
    # At step 0, there is no previous step, so we use -1 as sentinel.
    state = MFNoiseState(
        _inner_state=None,  # no previous step key
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
    ) -> tuple[Any, MFNoiseState]:
        step = st._step_counter

        # Generate z_t from the current step's PRNG seed
        current_key = rng_fold_in(st._rng_key, step)
        g_current = generator_from_key(current_key)
        z_t = _iid_normal_noise(clipped_grads, stddev, generator=g_current, dtype=dtype)

        if step == 0 or lambda_ == 0.0:
            # First step or no correlation: ñ_0 = z_0  (or d_0·z_0)
            corr_noise = z_t
        else:
            # Regenerate z_{t-1} from the previous step's PRNG seed
            prev_key = rng_fold_in(st._rng_key, step - 1)
            g_prev = generator_from_key(prev_key)
            z_prev = _iid_normal_noise(
                clipped_grads, stddev, generator=g_prev, dtype=dtype
            )

            # Correlated noise: ñ_t = z_t - λ · z_{t-1}
            corr_noise = tree_map(
                lambda zt, zp: zt - lambda_ * zp,
                z_t,
                z_prev,
            )

        # Column-norm scaling: ñ_t *= d_t
        if normalized:
            d_t = _column_norm(lambda_, n_steps, step)
            corr_noise = tree_map(lambda n: n * d_t, corr_noise)

        # Add correlated noise to gradients
        noisy_grads = tree_map(
            lambda grad, n: (grad + n).to(grad.dtype),
            clipped_grads,
            corr_noise,
        )

        new_state = MFNoiseState(
            _inner_state=None,  # no extra state needed (PRNG replay)
            _step_counter=step + 1,
            _rng_key=st._rng_key,
        )
        return noisy_grads, new_state

    return noise_fn, state


__all__ = ["lambda_cgd_noise"]
