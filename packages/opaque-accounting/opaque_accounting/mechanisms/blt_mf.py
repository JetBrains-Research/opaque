"""BLT mechanism — Buffered Linear Toeplitz matrix factorization accounting.

Provides privacy accounting for the BLT correlated noise mechanism.
The encoder is optimized for a specific participation pattern (min_sep,
max_participations) and the sensitivity is computed internally from the
optimized BLT.

References:
    - BLT: Choquette-Choo et al. (2024) https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
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
class BltMf(DpProcess):
    """BLT mechanism — Buffered Linear Toeplitz correlated noise.

    Represents the privacy cost of an entire BLT training run.
    The BLT encoder is optimized internally for the given participation
    pattern, and sensitivity is computed from the optimized result.
    """

    noise_multiplier: float
    n_steps: int
    min_sep: int
    max_participations: int | None
    error: str
    max_buffers: int
    momentum: float = 1.0

    @functools.lru_cache(maxsize=1)
    def _optimized_blt(self):
        """Optimize the BLT and cache it."""
        from opaque.noise.band_mf_noise import _momentum_workload_coef
        from opaque.noise.matrix_factorization.buffered_toeplitz import (
            optimize,
        )

        workload_coef = _momentum_workload_coef(self.momentum, self.n_steps)
        return optimize(
            n=self.n_steps,
            min_sep=self.min_sep,
            max_participations=self.max_participations,
            error=self.error,
            max_buffers=self.max_buffers,
            workload_coef=workload_coef,
        )

    @functools.lru_cache(maxsize=1)
    def sensitivity(self) -> float:
        """L2 sensitivity under the configured participation pattern.

        For single participation (min_sep=1, max_participations=1), this
        is the maximum column norm of the BLT encoder matrix.  For
        min-sep participation, uses the min-sep sensitivity computation.
        """
        from opaque.noise.matrix_factorization.buffered_toeplitz import (
            sensitivity_squared,
            toeplitz_coefs,
        )
        from opaque.noise.matrix_factorization.sensitivity import (
            minsep_true_max_participations,
        )
        from opaque.noise.matrix_factorization.toeplitz import (
            minsep_sensitivity_squared,
        )

        blt = self._optimized_blt()
        k = minsep_true_max_participations(
            n=self.n_steps,
            min_sep=self.min_sep,
            max_participations=self.max_participations,
        )

        if k == 1:
            # Single participation: closed-form BLT sensitivity
            sens_sq = sensitivity_squared(blt, n=self.n_steps)
        else:
            # Min-sep participation: via Toeplitz coefficients
            coefs = toeplitz_coefs(blt, self.n_steps)
            sens_sq = minsep_sensitivity_squared(
                strategy_coef=coefs,
                min_sep=self.min_sep,
                max_participations=self.max_participations,
                skip_checks=True,
            )

        return float(sens_sq.sqrt())

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


def blt_mf(
    noise_multiplier: float,
    n_steps: int,
    *,
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = "max",
    max_buffers: int = 10,
    momentum: float = 1.0,
) -> BltMf:
    """BLT mechanism — Buffered Linear Toeplitz correlated noise.

    Creates a privacy accounting process for the BLT mechanism.  The BLT
    encoder is optimized internally for the given participation pattern
    and the sensitivity is computed from the result.

    Args:
        noise_multiplier: Raw noise standard deviation sigma. Must be positive.
        n_steps: Number of training iterations. Must be >= 1.
        min_sep: Minimum separation between participations (default 1).
        max_participations: Maximum participations per user (default 1).
        error: Error metric to optimize: ``'max'`` or ``'mean'``.
        max_buffers: Maximum number of BLT buffers to try (default 10).
        momentum: Polyak momentum coefficient (default 1.0 = prefix-sum).
            Must match the momentum used in the optimizer/noise function.

    Returns:
        A :class:`BltMf` process.

    Example::

        import opaque.accounting as acc

        proc = acc.blt_mf(noise_multiplier=1.0, n_steps=1000)
        eps = proc.epsilon_at(1e-5)
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if min_sep < 1:
        raise ValueError(f"min_sep must be >= 1, got {min_sep}")
    if max_participations is not None and max_participations < 1:
        raise ValueError(
            f"max_participations must be >= 1 or None, got {max_participations}"
        )
    if error not in ("max", "mean"):
        raise ValueError(f"error must be 'max' or 'mean', got {error!r}")
    if max_buffers < 0:
        raise ValueError(f"max_buffers must be >= 0, got {max_buffers}")
    if momentum < 0:
        raise ValueError(f"momentum must be >= 0, got {momentum}")
    return BltMf(
        noise_multiplier, n_steps, min_sep, max_participations, error, max_buffers,
        momentum,
    )
