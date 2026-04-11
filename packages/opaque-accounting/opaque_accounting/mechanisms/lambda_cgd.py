"""DP-λCGD mechanism — correlated gradient descent accounting.

Provides privacy accounting for the DP-λCGD mechanism (Kalinin et al., 2026).
The strategy matrix C_λ is lower-triangular Toeplitz with entries λ^{i-j},
and its inverse is bidiagonal (1 on diagonal, -λ on subdiagonal), enabling
zero-memory-overhead noise generation via PRNG replay.

References:
    - Kalinin et al. (2026) "DP-λCGD: Leveraging Correlated Gradients
      for Improved DP-SGD" https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    DpProcess,
    Pld,
)
from opaque_accounting.discretization import (
    get_discretization,
)


@dataclass(frozen=True, slots=True)
class LambdaCgd(DpProcess):
    """DP-λCGD mechanism — correlated gradient descent.

    Represents the privacy cost of a DP-λCGD training run.
    The strategy matrix C_λ has entries λ^{i-j} (lower-triangular Toeplitz).
    Sensitivity is computed in closed form from Theorem 1 of the paper.

    When ``normalized=True`` (default), uses column-normalized C̃_λ = C_λ·D⁻¹
    (Appendix A of the paper).  All columns have unit norm, so:
    - Single-participation sensitivity = 1 (exact BnB analysis)
    - RMSE is strictly improved (Lemma 9)
    """

    noise_multiplier: float
    lambda_: float
    n_steps: int
    min_sep: int
    max_participations: int | None
    normalized: bool = True

    @functools.lru_cache(maxsize=1)
    def sensitivity(self) -> float:
        """L2 sensitivity under the configured participation pattern.

        When ``normalized=True``: uses Lemma 8 (column-normalized matrix).
        When ``normalized=False``: uses Theorem 1 eq 15 (unnormalized).
        """
        if self.normalized:
            sens_sq = _native.lambda_cgd_normalized_sensitivity_squared(
                self.lambda_,
                self.n_steps,
                self.min_sep,
                self.max_participations,
            )
        else:
            sens_sq = _native.lambda_cgd_sensitivity_squared(
                self.lambda_,
                self.n_steps,
                self.min_sep,
                self.max_participations,
            )
        return float(sens_sq**0.5)

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )
        return _native.mf_gaussian_pld(
            self.noise_multiplier,
            self.sensitivity(),
            config.to_native(),
        )


def lambda_cgd(
    noise_multiplier: float,
    lambda_: float,
    n_steps: int,
    *,
    min_sep: int = 1,
    max_participations: int | None = 1,
    normalized: bool = True,
) -> LambdaCgd:
    """DP-λCGD mechanism — correlated gradient descent.

    Creates a privacy accounting process for the DP-λCGD mechanism.
    The strategy matrix C_λ is lower-triangular Toeplitz with entries
    λ^{i-j}. Its inverse is bidiagonal (bandwidth 2), enabling
    zero-memory-overhead noise via PRNG replay.

    When ``normalized=True`` (default), the strategy matrix is
    column-normalized (C̃_λ = C_λ·D⁻¹ from Appendix A of the paper).
    This gives:
    - Sensitivity = 1 for single participation
    - Exact BnB amplification (all columns have unit norm)
    - Strictly improved RMSE (Lemma 9)

    Args:
        noise_multiplier: Raw noise standard deviation σ. Must be positive.
        lambda_: Correlation coefficient in [0, 1). λ=0 is DP-SGD.
        n_steps: Number of training iterations. Must be >= 1.
        min_sep: Minimum separation between participations (default 1).
        max_participations: Maximum participations per user (default 1).
            ``None`` means inferred from ``n_steps / min_sep``.
        normalized: If True (default), use column-normalized C̃_λ = C_λ·D⁻¹
            for improved sensitivity and exact BnB analysis.

    Returns:
        A :class:`LambdaCgd` process.

    Example::

        import opaque.accounting as acc

        # With BnB amplification (per-epoch, column-normalized)
        epoch = acc.balls_in_bins(
            acc.lambda_cgd(1.0, lambda_=0.9, n_steps=1875,
                           min_sep=1875, max_participations=1),
            num_bins=1875,
        )
        total = epoch * 8  # 8 epochs
        eps = total.epsilon_at(1e-5)
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if lambda_ < 0 or lambda_ >= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1), got {lambda_}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if min_sep < 1:
        raise ValueError(f"min_sep must be >= 1, got {min_sep}")
    if max_participations is not None and max_participations < 1:
        raise ValueError(
            f"max_participations must be >= 1 or None, got {max_participations}"
        )
    return LambdaCgd(
        noise_multiplier, lambda_, n_steps, min_sep, max_participations, normalized
    )
