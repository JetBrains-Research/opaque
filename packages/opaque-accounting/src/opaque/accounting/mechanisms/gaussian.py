"""Gaussian mechanism — base noise for DP-SGD."""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass

from .. import _native

from opaque.accounting.base import (
    DpProcess,
    Pld,
)
from opaque.accounting.discretization import (
    get_discretization,
)


@dataclass(frozen=True, slots=True)
class Gaussian(DpProcess):
    """Gaussian mechanism — stores noise_multiplier, computes PLD on demand."""

    noise_multiplier: float

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
        if self.noise_multiplier == 0:
            return _native.non_private_pld(config.to_native())
        return _native.gaussian_pld(self.noise_multiplier, config.to_native())


def gaussian(noise_multiplier: float) -> Gaussian:
    """Gaussian mechanism with noise multiplier σ.

    The Gaussian mechanism adds noise ~ N(0, σ²) to sensitivity-1 queries.
    This is the base mechanism for standard DP-SGD.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            Larger values = more privacy, less utility.  ``0`` is accepted
            (non-private: ε=∞).

    Returns:
        A :class:`Gaussian` process.

    Example::

        # Single Gaussian query
        proc = acc.gaussian(1.1)
        eps = proc.epsilon_at(1e-5)

        # Composed 1000 times
        training = acc.gaussian(1.1) * 1000
        eps = training.epsilon_at(1e-5)

        # Query-time discretization override
        eps = proc.epsilon_at(1e-5, discretization=1e-3)
    """
    if 0 < noise_multiplier < 0.1:
        warnings.warn(
            f"noise_multiplier={noise_multiplier} is very small: epsilon bounds "
            f"may explode and discretization grids may grow unboundedly, "
            f"leading to slow or inaccurate PLD computation.",
            stacklevel=2,
        )
    return Gaussian(noise_multiplier)
