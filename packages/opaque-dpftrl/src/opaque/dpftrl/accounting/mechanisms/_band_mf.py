"""BandMF mechanism for cyclic Poisson amplification.

References:
    - BandMF: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.dpftrl.accounting.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class BandMf(MfGaussian):
    """BandMF mechanism — banded Toeplitz Gaussian.

    Mirrors :class:`opaque.dpftrl.noise.BandMfStrategy`: the structural data
    is the strategy's first-column ``coefficients`` (length equals the band
    width).  Length-of-process parameters (``n_steps``) live on the
    amplification factory, not on the mechanism.
    """

    coefficients: tuple[float, ...] = ()

    @property
    def bands(self) -> int:
        """Band width — convenience alias for ``len(coefficients)``."""
        return len(self.coefficients)


def band_mf(
    noise_multiplier: float,
    sensitivity: float,
    coefficients: tuple[float, ...],
) -> BandMf:
    """BandMF mechanism for cyclic Poisson / b-min-sep amplification.

    Args:
        noise_multiplier: Raw noise standard deviation σ.
        sensitivity: L2 sensitivity of the BandMF strategy under its
            participation pattern.
        coefficients: First-column entries of the BandMF strategy matrix
            ``C`` (length equals the band width).  Same tuple as
            :attr:`opaque.dpftrl.noise.BandMfStrategy.coefficients`.

    Returns:
        A :class:`BandMf` process.

    Example::

        import opaque.dpftrl.accounting as ftrl_acc

        proc = ftrl_acc.poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0, 0.5)),
            sample_rate=0.01,
            n_steps=1000,
        )
    """
    return BandMf(noise_multiplier, sensitivity, tuple(coefficients))
