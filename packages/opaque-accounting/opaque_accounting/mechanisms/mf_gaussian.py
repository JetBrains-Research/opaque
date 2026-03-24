"""Matrix factorization Gaussian mechanism — correlated noise for MF-DP.

Provides privacy accounting for matrix factorization DP mechanisms
(BandMF, BLT, Dense). Unlike standard DP-SGD which composes per-step
Gaussian PLDs, MF mechanisms compute a single PLD for the entire training
run based on the effective noise multiplier σ/S.

References:
    - BandMF: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
    - BLT: Choquette-Choo et al. (2024) https://arxiv.org/abs/2404.16706
    - Dense MF: Denisov et al. (2022) https://arxiv.org/abs/2202.08312
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import _make_native_config


@dataclass(frozen=True, slots=True)
class MfGaussian(DpProcess):
    """MF Gaussian mechanism — stores noise_multiplier and sensitivity.

    Represents the privacy cost of an entire matrix factorization DP
    training run. The privacy reduces to a single Gaussian mechanism
    with effective noise multiplier σ/S.
    """

    noise_multiplier: float
    sensitivity: float

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return CgfPld(_native.cgf_mf_gaussian_pld(
            self.noise_multiplier, self.sensitivity
        ))

    def pmf(self, **kwargs: object) -> PmfPld:
        return PmfPld(_native.mf_gaussian_pld(
            self.noise_multiplier,
            self.sensitivity,
            _make_native_config(**kwargs),
        ))


def mf_gaussian(noise_multiplier: float, sensitivity: float) -> MfGaussian:
    """Matrix factorization Gaussian mechanism.

    Computes the privacy guarantee for the entire MF training run as a
    single Gaussian mechanism with effective noise multiplier σ/S.

    The sensitivity should be pre-computed based on the MF strategy
    (BandMF, BLT, Dense) and participation pattern (single, min-sep,
    fixed-epoch).

    Args:
        noise_multiplier: Raw noise standard deviation σ (before matrix
            factorization). Must be positive.
        sensitivity: L2 sensitivity S of the encoder matrix under the
            given participation pattern. Must be positive.

    Returns:
        An :class:`MfGaussian` process.

    Example::

        import opaque.accounting as acc

        # BandMF with pre-computed sensitivity
        proc = acc.mf_gaussian(noise_multiplier=1.0, sensitivity=2.5)
        eps = proc.cgf().epsilon_at(1e-5)
    """
    return MfGaussian(noise_multiplier, sensitivity)
