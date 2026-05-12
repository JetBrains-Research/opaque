"""DP-lambda-CGD strategy and noise — correlated noise via PRNG replay.

The DP-lambda-CGD mechanism (Kalinin et al., 2026) uses a lower-triangular
Toeplitz strategy matrix :math:`C_\\lambda` whose inverse is bidiagonal: 1 on the
diagonal, :math:`-\\lambda` on the subdiagonal.  The correlated noise at step t is::

    n_t = z_t - lambda * z_{t-1}              (unnormalized)
    n_t = d_t * (z_t - lambda * z_{t-1})      (column-normalized, default)

where :math:`z_t \\sim N(0, \\sigma^2 I)` are i.i.d. Gaussians, and :math:`d_t`
is the column norm of :math:`C_\\lambda` at step t.  Instead of storing
:math:`z_{t-1}`, we regenerate it from the previous step's PRNG seed —
zero additional memory overhead compared to DP-SGD.

References:
    - Kalinin et al. (2026) "DP-lambda-CGD: Leveraging Correlated Gradients
      for Improved DP-SGD" https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from opaque.api.dpftrl.noise._strategy_codec import register_strategy
from opaque.pytree import tree_map
from opaque.types import PerGroup
from opaque.random import generator_from_key
from opaque.random.types import RngKey
from opaque.random import fold_in as rng_fold_in

from ._engine import MFNoiseState, _iid_normal_noise
from ._streaming_matrix import StreamingMatrix


def _native():
    from opaque.api.accounting.core import _native as _n

    return _n


def _column_norm(lambda_: float, n_steps: int, step: int) -> float:
    """Column norm :math:`d_t` of :math:`C_\\lambda` at 0-indexed step t."""
    if lambda_ == 0.0:
        return 1.0
    remaining = n_steps - step
    lambda2 = lambda_ * lambda_
    lambda2r = lambda2**remaining
    if lambda2r < 1e-30:
        return math.sqrt(1.0 / (1.0 - lambda2))
    return math.sqrt((1.0 - lambda2r) / (1.0 - lambda2))


@register_strategy
@dataclass(frozen=True, slots=True)
class LambdaCgdStrategy:
    """DP-lambda-CGD strategy — recipe only (PRNG-replay noise)."""

    lambda_: float
    normalized: bool = True
    _gram_memo: dict[tuple[int, int, int | None], tuple[float, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if self.lambda_ < 0 or self.lambda_ >= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1), got {self.lambda_}")

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        # [1, λ, λ², ..., λ^{n_steps-1}].
        return torch.tensor(
            [self.lambda_**i for i in range(n_steps)], dtype=torch.float64
        )

    def gram_matrix(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> tuple[float, ...]:
        key = (n_steps, min_sep, max_participations)
        memo = self._gram_memo
        hit = memo.get(key)
        if hit is not None:
            return hit
        out = tuple(
            _native().lambda_cgd_gram_matrix(
                self.lambda_, n_steps, min_sep, max_participations, self.normalized
            )
        )
        memo[key] = out
        return out

    def streaming_matrix(self, **_) -> StreamingMatrix:
        # Lambda-CGD never materializes a streaming matrix — it uses
        # PRNG replay via :func:`_make_lambda_cgd_noise` instead.  The
        # mf_gaussian_noise dispatcher special-cases this strategy.
        raise NotImplementedError(
            "LambdaCgdStrategy uses PRNG-replay noise; the noise factory "
            "dispatches to _make_lambda_cgd_noise directly."
        )

    def sensitivity(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> float:
        if self.normalized:
            sens_sq = _native().lambda_cgd_normalized_sensitivity_squared(
                self.lambda_, n_steps, min_sep, max_participations
            )
        else:
            sens_sq = _native().lambda_cgd_sensitivity_squared(
                self.lambda_, n_steps, min_sep, max_participations
            )
        return float(sens_sq**0.5)

    def max_column_norm(self, *, n_steps: int) -> float:
        """Max L2 column norm of the strategy matrix at this horizon."""
        if self.normalized:
            return 1.0
        return float(_native().lambda_cgd_max_column_norm(self.lambda_, n_steps))


def lambda_cgd_strategy(
    *,
    lambda_: float,
    normalized: bool = True,
) -> LambdaCgdStrategy:
    """Create a DP-lambda-CGD strategy recipe (bandwidth=2, PRNG-replay noise).

    Args:
        lambda_: Correlation coefficient in [0, 1).
        normalized: Use column-normalized matrix (default True).

    Returns:
        A :class:`LambdaCgdStrategy` recipe.
    """
    return LambdaCgdStrategy(lambda_=lambda_, normalized=normalized)


# ---------------------------------------------------------------------------
# Internal noise builder (called by mf_gaussian_noise() dispatcher)
# ---------------------------------------------------------------------------


def _lambda_cgd_row_l2(strategy: LambdaCgdStrategy, n_steps: int, step: int) -> float:
    """Per-step row L2 norm of the λ-CGD effective C^{-1}.

    The realized per-coordinate noise at step ``t`` is
    ``base_σ · row_l2(t)`` (post-normalization when ``normalized=True``).
    At step 0 there is no previous-step term so the factor is just the
    optional column-norm multiplier; at step t≥1 the unnormalized factor
    is ``sqrt(1 + λ²)`` from ``z_t − λ z_{t−1}``.
    """
    lam = strategy.lambda_
    col = (
        _column_norm(lam, n_steps, min(step, n_steps - 1))
        if strategy.normalized
        else 1.0
    )
    if step == 0 or lam == 0.0:
        return col
    return col * math.sqrt(1.0 + lam * lam)


def _make_lambda_cgd_noise(
    grad_template: Any,
    strategy: LambdaCgdStrategy,
    *,
    n_steps: int,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState]],
    MFNoiseState,
    Callable[[int], float],
]:
    """DP-lambda-CGD noise via PRNG replay (zero extra memory).

    Returns ``(noise_fn, state, row_l2_at)`` where ``row_l2_at(step)``
    gives ``‖row_t(C^{-1})‖`` so the wrapping :func:`mf_gaussian_noise`
    factory can publish the realized per-step σ on
    :class:`NoisedPytree.noise_stddev` (= ``base_σ · row_l2_at(step)``).
    Adam-family bias correction reads that realized σ.
    """
    lambda_ = strategy.lambda_
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

    def row_l2_at(step: int) -> float:
        return _lambda_cgd_row_l2(strategy, n_steps, step)

    return noise_fn, state, row_l2_at


__all__ = ["LambdaCgdStrategy", "lambda_cgd_strategy"]
