"""BandMF mechanism for cyclic Poisson amplification.

References:
    - BandMF: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.accounting.mechanisms.mf_gaussian import MfGaussian


@dataclass(frozen=True)
class BandMf(MfGaussian):
    """BandMF mechanism for cyclic Poisson amplification.

    Carries ``num_groups`` (= ceil(n_steps / bands)), used by
    :func:`~opaque.accounting.amplification.cyclic_poisson.cyclic_poisson`
    for type dispatch and composition count.
    """

    num_groups: int = 1


def band_mf(
    noise_multiplier: float,
    sensitivity: float,
    num_groups: int = 1,
) -> BandMf:
    """BandMF mechanism for cyclic Poisson amplification.

    Args:
        noise_multiplier: Raw noise standard deviation σ.
        sensitivity: L2 sensitivity of the BandMF strategy.
        num_groups: Number of independent cyclic groups
            (= ceil(n_steps / bands)).

    Returns:
        A :class:`BandMf` process.

    Example::

        proc = acc.cyclic_poisson(
            acc.band_mf(1.0, sensitivity=1.0, num_groups=100),
            sample_rate=0.01,
        )
    """
    return BandMf(noise_multiplier, sensitivity, num_groups)
