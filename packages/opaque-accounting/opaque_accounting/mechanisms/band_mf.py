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
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    CgfPld,
    DpProcess,
    PmfPld,
)
from opaque_accounting.discretization import _make_native_config


@dataclass(frozen=True, slots=True)
class BandMf(DpProcess):
    """BandMF mechanism — banded Toeplitz correlated noise.

    Represents the privacy cost of an entire BandMF training run
    under single participation.
    """

    noise_multiplier: float
    n_steps: int
    bands: int

    @functools.lru_cache(maxsize=1)
    def _optimized_coefs(self):
        """Optimize Toeplitz coefficients and cache them."""
        from opaque.noise.matrix_factorization.toeplitz import (
            optimize as optimize_toeplitz,
        )

        return optimize_toeplitz(self.n_steps, self.bands)

    @functools.lru_cache(maxsize=1)
    def sensitivity(self) -> float:
        """L2 sensitivity under single participation."""
        coefs = self._optimized_coefs()
        return float(coefs.norm())

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return CgfPld(_native.cgf_mf_gaussian_pld(
            self.noise_multiplier, self.sensitivity()
        ))

    def pmf(self, **kwargs: object) -> PmfPld:
        return PmfPld(_native.mf_gaussian_pld(
            self.noise_multiplier,
            self.sensitivity(),
            _make_native_config(**kwargs),
        ))


def band_mf(
    noise_multiplier: float,
    n_steps: int,
    bands: int,
) -> BandMf:
    """BandMF mechanism — banded Toeplitz correlated noise.

    Creates a privacy accounting process for the BandMF mechanism under
    single participation (each user contributes one gradient at one step).

    For cyclic Poisson amplification (the common case), wrap with
    :func:`~opaque.accounting.amplification.cyclic_poisson`::

        proc = acc.cyclic_poisson(acc.band_mf(1.0, 1000, 10), sample_rate=0.01)

    Args:
        noise_multiplier: Raw noise standard deviation sigma. Must be positive.
        n_steps: Number of training iterations. Must be >= 1.
        bands: Number of bands in the Toeplitz matrix. Must be >= 1
            and <= ``n_steps``.

    Returns:
        A :class:`BandMf` process.

    Example::

        import opaque_accounting as acc

        proc = acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10)
        eps = proc.pmf().epsilon_at(1e-5)
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if bands < 1 or bands > n_steps:
        raise ValueError(f"bands must be in [1, n_steps={n_steps}], got {bands}")
    return BandMf(noise_multiplier, n_steps, bands)
