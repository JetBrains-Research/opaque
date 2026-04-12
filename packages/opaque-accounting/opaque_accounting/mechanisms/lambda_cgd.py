"""DP-λCGD and BISR mechanism — correlated gradient descent accounting.

Provides privacy accounting for the DP-λCGD mechanism (Kalinin et al., 2026)
and its generalization BISR (Banded Inverse Square Root, Kalinin et al., ICLR
2026). The strategy matrix C_λ is lower-triangular Toeplitz with entries
λ^{i-j}, and its inverse is bidiagonal (bandwidth 2). BISR generalises this
to arbitrary bandwidth p ≥ 2.

References:
    - Kalinin et al. (2026) "DP-λCGD: Leveraging Correlated Gradients
      for Improved DP-SGD" https://arxiv.org/abs/2601.22334
    - Kalinin, McKenna, Upadhyay, Lampert (2026) "Back to Square Roots:
      Banded Inverse Square Root for DP Matrix Factorization"
      https://arxiv.org/abs/2505.12128
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    DpProcess,
    Pld,
)
from opaque_accounting.discretization import (
    get_discretization,
)


def _bisr_inverse_coefficients(bandwidth: int) -> tuple[float, ...]:
    """Compute BISR inverse square-root coefficients c̃_k for prefix-sum workload.

    From Lemma 1 of arxiv:2505.12128. For α=1, β=0 (standard prefix-sum):
        c̃_k = r̃_k  where  r̃_0 = 1, r̃_j = ((j - 3/2) / j) · r̃_{j-1}

    First values: 1, -1/2, 1/8, -1/16, 5/128, -7/256, ...

    Args:
        bandwidth: Number of bands p ≥ 2.

    Returns:
        Tuple of p coefficients (c̃_0, c̃_1, ..., c̃_{p-1}).
    """
    coefs = [0.0] * bandwidth
    coefs[0] = 1.0
    for j in range(1, bandwidth):
        coefs[j] = ((j - 1.5) / j) * coefs[j - 1]
    return tuple(coefs)


@dataclass(frozen=True, slots=True)
class LambdaCgd(DpProcess):
    """DP-λCGD / BISR mechanism — correlated gradient descent.

    Represents the privacy cost of a DP-λCGD or BISR training run.

    For ``bandwidth=2`` (default): standard DP-λCGD with strategy matrix
    C_λ (entries λ^{i-j}, inverse bidiagonal [1, -λ]).

    For ``bandwidth > 2``: BISR mechanism where C^{-1} is banded with
    the given bandwidth. Coefficients are either computed automatically
    (BISR optimal inverse square-root) or provided explicitly.

    When ``normalized=True`` (default), uses column-normalized C̃ = C·D⁻¹.
    All columns have unit norm, so single-participation sensitivity = 1.
    """

    noise_multiplier: float
    lambda_: float
    n_steps: int
    min_sep: int
    max_participations: int | None
    normalized: bool = True
    momentum: float = 0.0
    bandwidth: int = 2
    coefficients: tuple[float, ...] | None = None

    @functools.lru_cache(maxsize=1)
    def _effective_coefficients(self) -> tuple[float, ...]:
        """Compute the C^{-1} band coefficients.

        For bandwidth=2, coefficients=None: [1, -lambda_] (standard λCGD).
        For bandwidth>2, coefficients=None: BISR optimal (inverse square-root).
        For explicit coefficients: use as provided.
        """
        if self.coefficients is not None:
            return self.coefficients
        if self.bandwidth == 2:
            return (1.0, -self.lambda_)
        return _bisr_inverse_coefficients(self.bandwidth)

    def _use_fast_lambda_cgd_path(self) -> bool:
        """Whether to use the optimised closed-form λCGD Rust functions."""
        return self.bandwidth == 2 and self.coefficients is None

    @functools.lru_cache(maxsize=1)
    def sensitivity(self) -> float:
        """L2 sensitivity under the configured participation pattern."""
        if self._use_fast_lambda_cgd_path():
            if self.normalized:
                sens_sq = _native.lambda_cgd_normalized_sensitivity_squared(
                    self.lambda_, self.n_steps, self.min_sep,
                    self.max_participations, self.momentum,
                )
            else:
                sens_sq = _native.lambda_cgd_sensitivity_squared(
                    self.lambda_, self.n_steps, self.min_sep,
                    self.max_participations, self.momentum,
                )
        else:
            coefs = list(self._effective_coefficients())
            if self.normalized:
                sens_sq = _native.bisr_normalized_sensitivity_squared(
                    coefs, self.n_steps, self.min_sep,
                    self.max_participations, self.momentum,
                )
            else:
                sens_sq = _native.bisr_sensitivity_squared(
                    coefs, self.n_steps, self.min_sep,
                    self.max_participations, self.momentum,
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
    momentum: float = 0.0,
    bandwidth: int = 2,
    coefficients: Sequence[float] | None = None,
) -> LambdaCgd:
    """DP-λCGD / BISR mechanism — correlated gradient descent.

    Creates a privacy accounting process for the DP-λCGD mechanism
    (bandwidth=2) or BISR (bandwidth > 2).

    For bandwidth=2 (default): standard λCGD where the strategy matrix
    C_λ is lower-triangular Toeplitz with entries λ^{i-j}, and its inverse
    is bidiagonal [1, -λ].

    For bandwidth > 2: BISR mechanism (arxiv:2505.12128). The inverse
    strategy matrix C^{-1} is banded with the given bandwidth.
    Coefficients default to the BISR optimal inverse square-root formula.

    Args:
        noise_multiplier: Raw noise standard deviation σ. Must be positive.
        lambda_: Correlation coefficient in [0, 1). λ=0 is DP-SGD.
            For BISR with default coefficients, this is ignored (BISR
            coefficients are analytically determined).
        n_steps: Number of training iterations. Must be >= 1.
        min_sep: Minimum separation between participations (default 1).
        max_participations: Maximum participations per user (default 1).
        normalized: If True (default), use column-normalized matrix.
        momentum: Optimizer momentum β in [0, 1). Default 0.
        bandwidth: Bandwidth p of C^{-1} (default 2 = standard λCGD).
            Must be >= 2.
        coefficients: Explicit C^{-1} band coefficients. If None (default),
            auto-computed: [1, -λ] for bandwidth=2, BISR optimal for bandwidth>2.
            If provided, must have length = bandwidth.

    Returns:
        A :class:`LambdaCgd` process.

    Example::

        import opaque_accounting as acc

        # Standard λCGD (bandwidth=2)
        training = acc.balls_in_bins(
            acc.lambda_cgd(1.0, lambda_=0.9, n_steps=15000,
                           min_sep=1875, max_participations=8),
            num_bins=1875, num_epochs=8,
        )

        # BISR (bandwidth=4)
        training = acc.balls_in_bins(
            acc.lambda_cgd(1.0, lambda_=0.5, n_steps=15000,
                           min_sep=1875, max_participations=8,
                           bandwidth=4),
            num_bins=1875, num_epochs=8,
        )
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
    if momentum < 0 or momentum >= 1.0:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}")
    if bandwidth < 2:
        raise ValueError(f"bandwidth must be >= 2, got {bandwidth}")
    if coefficients is not None:
        if len(coefficients) != bandwidth:
            raise ValueError(
                f"coefficients length ({len(coefficients)}) must equal "
                f"bandwidth ({bandwidth})"
            )
        coefficients = tuple(coefficients)

    return LambdaCgd(
        noise_multiplier, lambda_, n_steps, min_sep, max_participations,
        normalized, momentum, bandwidth, coefficients,
    )


def bisr(
    noise_multiplier: float,
    n_steps: int,
    bandwidth: int,
    *,
    min_sep: int = 1,
    max_participations: int | None = 1,
    normalized: bool = True,
    momentum: float = 0.0,
    coefficients: Sequence[float] | None = None,
) -> LambdaCgd:
    """BISR mechanism — Banded Inverse Square Root (arxiv:2505.12128).

    Convenience wrapper for :func:`lambda_cgd` with bandwidth > 2.
    The BISR coefficients are computed from the inverse square-root
    formula (Lemma 1 of the paper).

    BISR with bandwidth=2 is equivalent to DP-λCGD with λ=1/2.

    Args:
        noise_multiplier: Raw noise standard deviation σ. Must be positive.
        n_steps: Number of training iterations.
        bandwidth: Bandwidth p of C^{-1} (≥ 2). Higher = better utility,
            more noise vectors to regenerate.
        min_sep: Minimum separation between participations (default 1).
        max_participations: Maximum participations per user (default 1).
        normalized: If True (default), use column-normalized matrix.
        momentum: Optimizer momentum β in [0, 1). Default 0.
        coefficients: Explicit C^{-1} coefficients. Default: BISR optimal.

    Returns:
        A :class:`LambdaCgd` process.

    Example::

        training = acc.balls_in_bins(
            acc.bisr(1.0, n_steps=15000, bandwidth=4,
                     min_sep=1875, max_participations=8),
            num_bins=1875, num_epochs=8,
        )
    """
    return lambda_cgd(
        noise_multiplier,
        lambda_=0.5,  # canonical BISR value; ignored when coefficients are auto-computed
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        normalized=normalized,
        momentum=momentum,
        bandwidth=bandwidth,
        coefficients=coefficients,
    )
