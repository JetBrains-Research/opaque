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
    - Kalinin et al. (2026) "DP-λCGD: Efficient Noise Correlation for
      Differentially Private Model Training" https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from opaque.api.dpftrl.noise._plan import (
    MfExecutionPlan,
    lambda_replay_execution_plan,
)
from opaque.api.dpftrl.noise._strategy_codec import register_strategy

from ._schedule_fingerprint import materialize_schedule

if TYPE_CHECKING:
    import numpy as np

    from opaque.api.engine.scheduling.types import Schedule

    from ._streaming_matrix import StreamingMatrix


def _native():
    from opaque.api.accounting.core import _native as _n

    return _n


_lr_key = materialize_schedule


@lru_cache(maxsize=256)
def _lambda_cgd_gram_matrix_cached(
    lambda_: float,
    normalized: bool,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
    lr_key: tuple[float, ...] | None,
) -> tuple[float, ...]:
    """Gram sequence for λ-CGD; cached across repeated σ / PLD probes."""
    if lr_key is not None:
        return tuple(
            _native().lambda_cgd_gram_matrix_lr(
                lambda_,
                0.0,
                n_steps,
                min_sep,
                max_participations,
                normalized,
                list(lr_key),
            )
        )
    return tuple(
        _native().lambda_cgd_gram_matrix(
            lambda_, n_steps, min_sep, max_participations, normalized
        )
    )


def _column_norm(lambda_: float, n_steps: int, step: int) -> float:
    """Column norm :math:`d_t` of :math:`C_\\lambda` at 0-indexed step t.

    ``step`` must be in ``[0, n_steps)``.  At ``step == n_steps`` the
    closed form collapses to 0 (and beyond it is undefined), which would
    zero out the released noise under ``normalized=True``.
    """
    if step < 0 or step >= n_steps:
        raise ValueError(
            f"column-norm step {step} is outside the calibrated horizon [0, {n_steps})."
        )
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
    lr_schedule: Schedule | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.lambda_ < 0 or self.lambda_ >= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1), got {self.lambda_}")

    def execution_plan(self, *, n_steps: int, **_) -> MfExecutionPlan:
        return lambda_replay_execution_plan(
            self.lambda_, n_steps, normalized=self.normalized
        )

    def coefficients(self, *, n_steps: int, **_) -> np.ndarray:
        return self.execution_plan(n_steps=n_steps).coefficients()

    def gram_matrix(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> tuple[float, ...]:
        return _lambda_cgd_gram_matrix_cached(
            self.lambda_,
            self.normalized,
            n_steps,
            min_sep,
            max_participations,
            _lr_key(self.lr_schedule, n_steps),
        )

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
    lr_schedule: Schedule | None = None,
) -> LambdaCgdStrategy:
    """Create a DP-lambda-CGD strategy recipe (bandwidth=2, PRNG-replay noise).

    Args:
        lambda_: Correlation coefficient in [0, 1).
        normalized: Use column-normalized matrix (default True).
        lr_schedule: Optional per-step learning-rate schedule used for
            schedule-weighted Gram accounting. λ-CGD remains momentum-free.

    Returns:
        A :class:`LambdaCgdStrategy` recipe.
    """
    return LambdaCgdStrategy(
        lambda_=lambda_,
        normalized=normalized,
        lr_schedule=lr_schedule,
    )


def _lambda_cgd_row_l2(strategy: LambdaCgdStrategy, n_steps: int, step: int) -> float:
    """Per-step row L2 norm of the λ-CGD effective C^{-1}.

    The realized per-coordinate noise at step ``t`` is
    ``base_σ · row_l2(t)`` (post-normalization when ``normalized=True``).
    At step 0 there is no previous-step term so the factor is just the
    optional column-norm multiplier; at step t≥1 the unnormalized factor
    is ``sqrt(1 + λ²)`` from ``z_t − λ z_{t−1}``.

    ``step`` must be in ``[0, n_steps)`` — the noise function raises on
    past-horizon calls, so this lookup is never asked to invent a factor
    outside the calibrated matrix.
    """
    lam = strategy.lambda_
    col = _column_norm(lam, n_steps, step) if strategy.normalized else 1.0
    if step == 0 or lam == 0.0:
        return col
    return col * math.sqrt(1.0 + lam * lam)


__all__ = ["LambdaCgdStrategy", "lambda_cgd_strategy"]
