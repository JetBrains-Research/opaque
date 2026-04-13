"""BandMF mechanism — banded Toeplitz matrix factorization accounting.

Provides privacy accounting for the BandMF correlated noise mechanism.
The encoder matrix is a banded lower-triangular Toeplitz matrix with
column norms normalized to 1, giving single-participation sensitivity = 1.

For cyclic Poisson amplification, wrap with
:func:`~opaque.accounting.amplification.cyclic_poisson.cyclic_poisson`.

References:
    - BandMF: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    DpProcess,
    Pld,
)
from opaque_accounting.discretization import (
    get_discretization,
)


@dataclass(frozen=True, slots=True)
class BandMf(DpProcess):
    """BandMF mechanism — banded Toeplitz correlated noise.

    Represents the privacy cost of an entire BandMF training run
    under single participation. Each user contributes one gradient at
    one step; the sensitivity equals the maximum column norm of the
    optimized encoder matrix (= 1 for the standard Toeplitz optimization).

    When ``lr_schedule`` is provided, the Toeplitz coefficients are
    optimized for the LR-weighted workload ``[η₀, η₁·β, η₂·β², ...]``
    rather than the constant-LR workload ``[1, β, β², ...]``.

    For cyclic Poisson amplification, wrap with
    :func:`~opaque.accounting.amplification.cyclic_poisson.cyclic_poisson`.
    """

    noise_multiplier: float
    n_steps: int
    bands: int
    momentum: float = 1.0
    lr_schedule: tuple[float, ...] | None = None

    @functools.lru_cache(maxsize=1)
    def _optimized_coefs(self):
        """Optimize Toeplitz coefficients and cache them."""
        from opaque.noise.band_mf_noise import _momentum_workload_coef
        from opaque.noise.matrix_factorization.toeplitz import (
            optimize as optimize_toeplitz,
        )

        lr_tensor = (
            torch.tensor(self.lr_schedule, dtype=torch.float64)
            if self.lr_schedule is not None
            else None
        )
        workload_coef = _momentum_workload_coef(
            self.momentum, self.n_steps, lr_schedule=lr_tensor
        )
        return optimize_toeplitz(self.n_steps, self.bands, workload_coef=workload_coef)

    @functools.lru_cache(maxsize=1)
    def strategy_coefficients(self) -> list[float]:
        """Toeplitz strategy coefficients of the optimized encoder.

        Returns the banded Toeplitz coefficients as a list, suitable for
        Gram matrix computation in BnB amplification.
        """
        return self._optimized_coefs().detach().cpu().tolist()

    @functools.lru_cache(maxsize=1)
    def sensitivity(self) -> float:
        """L2 sensitivity under single participation.

        For the standard Toeplitz optimization (coefficients normalized
        to L2 norm 1), this equals 1.0.
        """
        coefs = self._optimized_coefs()
        return float(coefs.norm())

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


def band_mf(
    noise_multiplier: float,
    n_steps: int,
    bands: int,
    momentum: float = 1.0,
    lr_schedule: Sequence[float] | None = None,
) -> BandMf:
    """BandMF mechanism — banded Toeplitz correlated noise.

    Creates a privacy accounting process for the BandMF mechanism under
    single participation (each user contributes one gradient at one step).

    When ``lr_schedule`` is provided, the Toeplitz coefficients are
    optimized for the LR-weighted workload ``[η₀, η₁·β, η₂·β², ...]``
    rather than assuming constant learning rate.

    For cyclic Poisson amplification (the common case), wrap with
    :func:`~opaque.accounting.amplification.cyclic_poisson`::

        proc = acc.cyclic_poisson(acc.band_mf(1.0, 1000, 10), sample_rate=0.01)

    Args:
        noise_multiplier: Raw noise standard deviation sigma. Must be positive.
        n_steps: Number of training iterations. Must be >= 1.
        bands: Number of bands in the Toeplitz matrix. Must be >= 1
            and <= ``n_steps``.
        momentum: Polyak momentum coefficient (default 1.0 = prefix-sum).
            Must match the momentum used in the optimizer/noise function.
        lr_schedule: Optional per-step learning rate schedule. If provided,
            must have length ``n_steps``.

    Returns:
        A :class:`BandMf` process.

    Example::

        import opaque.accounting as acc

        # BandMF without subsampling
        proc = acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10)
        eps = proc.epsilon_at(1e-5)

        # BandMF with cyclic Poisson amplification (recommended)
        proc = acc.cyclic_poisson(acc.band_mf(1.0, 1000, 10), sample_rate=0.01)
        eps = proc.epsilon_at(1e-5)
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if bands < 1 or bands > n_steps:
        raise ValueError(f"bands must be in [1, n_steps={n_steps}], got {bands}")
    if momentum < 0:
        raise ValueError(f"momentum must be >= 0, got {momentum}")
    if lr_schedule is not None:
        if len(lr_schedule) != n_steps:
            raise ValueError(
                f"lr_schedule length ({len(lr_schedule)}) must equal n_steps ({n_steps})"
            )
        lr_schedule = tuple(lr_schedule)
    return BandMf(noise_multiplier, n_steps, bands, momentum, lr_schedule)
