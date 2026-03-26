"""Rectified (clamped) Gaussian mechanism — tighter privacy via bounded support."""

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
class RectifiedGaussian(DpProcess):
    """Rectified Gaussian mechanism — clamp noise to [−R·σ, R·σ].

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
        return _native.rectified_gaussian_pld(
            self.noise_multiplier, self.radius, config.to_native()
        )


def rectified_gaussian(noise_multiplier: float, radius: float = 3.0) -> RectifiedGaussian:
    """Rectified (clamped) Gaussian mechanism.

    Samples noise from N(0, σ²) and clamps to [−R·σ, R·σ].  Point masses
    accumulate at the boundaries.  Provides tighter privacy bounds than
    the standard (unbounded) Gaussian at identical noise levels.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
        radius: Support half-width in sigma units (R).  The noise domain is
            [−R·σ, R·σ].  Typical values: 3–10 for meaningful bounding.

    Returns:
        A :class:`RectifiedGaussian` process.

    Example::

        proc = acc.rectified_gaussian(1.1, radius=5.0)
        eps = proc.epsilon_at(1e-5)

        # Compose with Poisson subsampling
        step = acc.poisson(acc.rectified_gaussian(1.1, 5.0), sample_rate=0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    return RectifiedGaussian(noise_multiplier, radius)
