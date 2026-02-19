"""Accumulated (microbatched) Poisson-subsampled Gaussian mechanism."""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.amplification.poisson import Poisson
from opaque.accounting.base import DiscretizationConfig, DpProcess, Pld


@dataclass(frozen=True, slots=True)
class Accumulated(DpProcess):
    """Accumulated (microbatched) Poisson-subsampled Gaussian mechanism."""

    noise_multiplier: float
    sample_rate: float
    microbatches: int
    config: DiscretizationConfig | None = field(default=None, repr=False)

    def pld(self) -> Pld:
        return _native.accumulated_poisson_gaussian_pld(
            self.noise_multiplier,
            self.sample_rate,
            self.microbatches,
            config=self.config,
        )


def accumulate(
    inner: Poisson,
    microbatches: int,
) -> DpProcess:
    """Gradient accumulation (microbatching) mechanism.

    Process gradients in ``microbatches`` sub-batches, accumulate clipped gradients,
    then add noise once. This improves gradient quality compared to adding noise
    per microbatch while maintaining the same privacy guarantee.

    Args:
        inner: A Poisson process (from :func:`poisson`).
        microbatches: Number of microbatches to accumulate before noising.

    Returns:
        An :class:`Accumulated` process.

    Example::

        # Accumulate 4 microbatches per step
        step = acc.accumulate(
            acc.poisson(acc.gaussian(1.1), 0.01),
            microbatches=4,
        )
        training = step * 500
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, Poisson):
        raise TypeError(
            f"accumulate() requires a Poisson inner mechanism, got {type(inner).__name__}. "
            "Use: acc.accumulate(acc.poisson(acc.gaussian(nm), rate), microbatches=k)"
        )
    return Accumulated(
        inner.noise_multiplier,
        inner.sample_rate,
        microbatches,
        config=inner.config,
    )
