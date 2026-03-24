"""Truncated (renormalized) Gaussian mechanism — tightest privacy via bounded support."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    DpProcess,
    Pld,
)
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class TruncatedGaussian(DpProcess):
    """Truncated Gaussian mechanism — renormalized density on [−R·σ, R·σ].

    Stores ``noise_multiplier`` and ``radius``, computes PLD on demand.
    """

    noise_multiplier: float
    radius: float

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> Pld:
        return _native.truncated_gaussian_pld(
            self.noise_multiplier, self.radius, config.to_native()
        )


def truncated_gaussian(noise_multiplier: float, radius: float) -> TruncatedGaussian:
    """Truncated (renormalized) Gaussian mechanism.

    Noise is sampled from a Gaussian restricted to [−R·σ, R·σ] with properly
    renormalized density (inverse-CDF sampling).  No point masses.  Provides
    the tightest privacy bounds among bounded Gaussian variants.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
        radius: Support half-width in sigma units (R).  The noise domain is
            [−R·σ, R·σ].  Typical values: 3–10 for meaningful bounding.

    Returns:
        A :class:`TruncatedGaussian` process.

    Example::

        proc = acc.truncated_gaussian(1.1, radius=5.0)
        eps = proc.pmf(acc.DiscretizationConfig()).epsilon_at(1e-5)
    """
    return TruncatedGaussian(noise_multiplier, radius)
