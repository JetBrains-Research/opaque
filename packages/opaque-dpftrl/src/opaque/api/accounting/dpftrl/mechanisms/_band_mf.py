"""BandMF mechanism for cyclic Poisson amplification.

References:
    - BandMF: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian


@dataclass(frozen=True)
class BandMf(MfGaussian):
    """BandMF mechanism — banded Toeplitz Gaussian.

    Mirrors :class:`opaque.dpftrl.noise.BandMfStrategy`: the structural data
    is the strategy's first-column ``coefficients`` (length equals the band
    width).  Length-of-process parameters (``n_steps``) live on the
    amplification factory, not on the mechanism.
    """

    coefficients: tuple[float, ...] = ()

    def __post_init__(self):
        # Empty ``coefficients`` would set ``bands == 0`` and silently zero
        # out downstream amplification (``PoissonMf`` derives
        # ``num_groups = ceil(n_steps / bands)``), so guard direct construction
        # and deserialization the same way the factory does.
        if not self.coefficients:
            raise ValueError(
                "BandMf: coefficients must be a non-empty tuple "
                "(length equals the BandMF band width)."
            )

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
            ``C`` (length equals the band width, must be non-empty).  Same
            tuple as
            :attr:`opaque.dpftrl.noise.BandMfStrategy.coefficients`.

    Returns:
        A :class:`BandMf` process.

    Raises:
        ValueError: If ``coefficients`` is empty.

    Example::

        import opaque.dpftrl.accounting as ftrl_acc

        proc = ftrl_acc.poisson(
            ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=(1.0, 0.5)),
            sample_rate=0.01,
            n_steps=1000,
        )
    """
    return BandMf(noise_multiplier, sensitivity, tuple(coefficients))
