"""DP-lambda-CGD strategy and noise -- correlated noise via PRNG replay.

The DP-lambda-CGD mechanism (Kalinin et al., 2026) uses a lower-triangular
Toeplitz strategy matrix C_lambda whose inverse is bidiagonal: 1 on the
diagonal, -lambda on the subdiagonal.  The correlated noise at step t is:

    n_t = z_t - lambda * z_{t-1}                  (unnormalized)
    n_t = d_t * (z_t - lambda * z_{t-1})           (column-normalized, default)

where z_t ~ N(0, sigma^2 I) are i.i.d. Gaussians, and d_t is the column norm
of C_lambda at step t.

Instead of storing z_{t-1}, we regenerate it from the previous step's
PRNG seed -- zero additional memory overhead compared to DP-SGD.

Use ``mf_noise(lambda_cgd_strategy(...), ...)`` to create the noise function.

References:
    - Kalinin et al. (2026) "DP-lambda-CGD: Leveraging Correlated Gradients
      for Improved DP-SGD" https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from opaque.api.dpftrl.noise._strategy_codec import register_strategy
from opaque.pytree import tree_map
from opaque.types import PerGroup
from opaque.random import generator_from_key
from opaque.random.types import RngKey
from opaque.random import fold_in as rng_fold_in

from ._engine import MFNoiseState, _iid_normal_noise


def _native():
    from opaque.api.accounting.core import _native as _n

    return _n


def _column_norm(lambda_: float, n_steps: int, step: int) -> float:
    """Column norm d_t of C_lambda at 0-indexed step t."""
    if lambda_ == 0.0:
        return 1.0
    remaining = n_steps - step
    lambda2 = lambda_ * lambda_
    lambda2r = lambda2**remaining
    if lambda2r < 1e-30:
        return math.sqrt(1.0 / (1.0 - lambda2))
    return math.sqrt((1.0 - lambda2r) / (1.0 - lambda2))


__all__ = ["LambdaCgdStrategy", "lambda_cgd_strategy"]


# ---------------------------------------------------------------------------
# Strategy dataclass and factory
# ---------------------------------------------------------------------------


@register_strategy
@dataclass(frozen=True, slots=True)
class LambdaCgdStrategy:
    """DP-lambda-CGD strategy (PRNG replay noise)."""

    sensitivity: float
    n_steps: int
    lambda_: float
    min_sep: int
    max_participations: int | None
    normalized: bool
    _coefficients: tuple[float, ...]
    _gram_matrix: tuple[float, ...] | None = None
    _max_column_norm: float = 0.0

    def with_horizon(
        self, n_steps: int, max_participations: int | None
    ) -> "LambdaCgdStrategy":
        """Return a fresh strategy regenerated at horizon ``n_steps``.

        Recomputes Gram, sensitivity, and the geometric forward coefficients
        for the new horizon via the same closed-form Rust helpers.
        """
        import dataclasses

        if self.normalized:
            sens_sq = _native().lambda_cgd_normalized_sensitivity_squared(
                self.lambda_, n_steps, self.min_sep, max_participations
            )
            max_column_norm = 1.0
        else:
            sens_sq = _native().lambda_cgd_sensitivity_squared(
                self.lambda_, n_steps, self.min_sep, max_participations
            )
            max_column_norm = float(
                _native().lambda_cgd_max_column_norm(self.lambda_, n_steps)
            )
        new_sensitivity = float(sens_sq**0.5)
        new_coefs = tuple(self.lambda_**i for i in range(n_steps))
        new_gram = tuple(
            _native().lambda_cgd_gram_matrix(
                self.lambda_,
                n_steps,
                self.min_sep,
                max_participations,
                self.normalized,
            )
        )
        return dataclasses.replace(
            self,
            sensitivity=new_sensitivity,
            n_steps=n_steps,
            max_participations=max_participations,
            _coefficients=new_coefs,
            _gram_matrix=new_gram,
            _max_column_norm=max_column_norm,
        )


def lambda_cgd_strategy(
    lambda_: float,
    n_steps: int,
    min_sep: int,
    max_participations: int | None = 1,
    *,
    normalized: bool = True,
) -> LambdaCgdStrategy:
    """Create a DP-lambda-CGD strategy (bandwidth=2, PRNG-replay noise).

    Uses closed-form Rust functions for sensitivity and Gram matrix.

    Note: momentum does not affect lambda-CGD (bandwidth=2). The strategy
    coefficients are always [1, -lambda]. For momentum-aware coefficients,
    use :func:`bisr_strategy` with bandwidth > 2.

    Args:
        lambda_: Correlation coefficient in [0, 1).
        n_steps: Total training steps.
        min_sep: Minimum separation between participations.
        max_participations: Maximum participations per user (default 1).
        normalized: Use column-normalized matrix (default True).

    Returns:
        A :class:`LambdaCgdStrategy` with pre-computed Gram matrix.
    """
    if lambda_ < 0 or lambda_ >= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1), got {lambda_}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    # Sensitivity (closed-form Rust)
    if normalized:
        sens_sq = _native().lambda_cgd_normalized_sensitivity_squared(
            lambda_,
            n_steps,
            min_sep,
            max_participations,
        )
    else:
        sens_sq = _native().lambda_cgd_sensitivity_squared(
            lambda_,
            n_steps,
            min_sep,
            max_participations,
        )
    sensitivity = float(sens_sq**0.5)

    # Coefficients (for inspection): [1, lambda, lambda^2, ...] truncated at n_steps
    coefficients = tuple(lambda_**i for i in range(n_steps))

    # Gram matrix (closed-form Rust)
    gram = _native().lambda_cgd_gram_matrix(
        lambda_,
        n_steps,
        min_sep,
        max_participations,
        normalized,
    )
    gram_matrix = tuple(gram)

    if normalized:
        max_column_norm = 1.0  # all columns have unit norm after normalization
    else:
        max_column_norm = float(_native().lambda_cgd_max_column_norm(lambda_, n_steps))

    return LambdaCgdStrategy(
        sensitivity=sensitivity,
        n_steps=n_steps,
        lambda_=lambda_,
        min_sep=min_sep,
        max_participations=max_participations,
        normalized=normalized,
        _coefficients=coefficients,
        _gram_matrix=gram_matrix,
        _max_column_norm=max_column_norm,
    )


# ---------------------------------------------------------------------------
# Internal noise builder (called by mf_noise() dispatcher)
# ---------------------------------------------------------------------------


def _make_lambda_cgd_noise(
    grad_template: Any,
    strategy: LambdaCgdStrategy,
    *,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """DP-lambda-CGD noise via PRNG replay (zero extra memory)."""
    lambda_ = strategy.lambda_
    n_steps = strategy.n_steps
    normalized = strategy.normalized

    state = MFNoiseState(
        _inner_state=None,
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
        *,
        stddev: float | PerGroup,
    ) -> tuple[Any, MFNoiseState]:
        step = st._step_counter

        current_key = rng_fold_in(st._rng_key, step)
        g_current = generator_from_key(current_key)
        z_t = _iid_normal_noise(
            clipped_grads,
            stddev,
            generator=g_current,
            dtype=dtype,
        )

        if step == 0 or lambda_ == 0.0:
            corr_noise = z_t
        else:
            prev_key = rng_fold_in(st._rng_key, step - 1)
            g_prev = generator_from_key(prev_key)
            z_prev = _iid_normal_noise(
                clipped_grads,
                stddev,
                generator=g_prev,
                dtype=dtype,
            )
            corr_noise = tree_map(
                lambda zt, zp: zt - lambda_ * zp,
                z_t,
                z_prev,
            )

        if normalized:
            d_t = _column_norm(lambda_, n_steps, step)
            corr_noise = tree_map(lambda n: n * d_t, corr_noise)

        noisy_grads = tree_map(
            lambda grad, n: (grad + n).to(grad.dtype),
            clipped_grads,
            corr_noise,
        )

        new_state = MFNoiseState(
            _inner_state=None,
            _step_counter=step + 1,
            _rng_key=st._rng_key,
        )
        return noisy_grads, new_state

    return noise_fn, state
