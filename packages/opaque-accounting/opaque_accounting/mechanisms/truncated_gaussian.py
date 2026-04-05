"""Truncated (renormalized) Gaussian mechanism — tightest privacy via bounded support."""

from __future__ import annotations

import functools
import warnings
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
class TruncatedGaussian(DpProcess):
    """Truncated Gaussian mechanism — renormalized density on [−R·σ, R·σ].

    Stores ``noise_multiplier`` and ``radius``, computes PLD on demand.
    """

    noise_multiplier: float
    radius: float

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
        return _native.truncated_gaussian_pld(
            self.noise_multiplier, self.radius, config.to_native()
        )


def truncated_gaussian(
    noise_multiplier: float, radius: float = 3.0
) -> TruncatedGaussian:
    """Truncated (renormalized) Gaussian mechanism.

    Noise is sampled from a Gaussian restricted to [−R·σ, R·σ] with properly
    renormalized density (inverse-CDF sampling).  No point masses.  Provides
    the tightest privacy bounds among bounded Gaussian variants.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            ``0`` is accepted (non-private: ε=∞).
        radius: Support half-width in sigma units (R).  The noise domain is
            [−R·σ, R·σ].  Typical values: 3–10 for meaningful bounding.

    Returns:
        A :class:`TruncatedGaussian` process.

    Example::

        proc = acc.truncated_gaussian(1.1, radius=5.0)
        eps = proc.epsilon_at(1e-5)

        # Compose with Poisson subsampling
        step = acc.poisson(acc.truncated_gaussian(1.1, 5.0), sample_rate=0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if 0 < noise_multiplier < 0.1:
        warnings.warn(
            f"noise_multiplier={noise_multiplier} is very small: epsilon bounds "
            f"may explode and discretization grids may grow unboundedly, "
            f"leading to slow or inaccurate PLD computation.",
            stacklevel=2,
        )
    return TruncatedGaussian(noise_multiplier, radius)
