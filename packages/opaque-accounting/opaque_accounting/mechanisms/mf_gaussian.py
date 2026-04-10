"""Matrix factorization Gaussian mechanism — correlated noise for MF-DP.

Provides privacy accounting for matrix factorization DP mechanisms
(BandMF). Unlike standard DP-SGD which composes per-step
Gaussian PLDs, MF mechanisms compute a single PLD for the entire training
run based on the effective noise multiplier σ/S.

References:
    - BandMF: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

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
class MfGaussian(DpProcess):
    """MF Gaussian mechanism — stores noise_multiplier and sensitivity.

    Represents the privacy cost of an entire matrix factorization DP
    training run. The privacy reduces to a single Gaussian mechanism
    with effective noise multiplier σ/S.
    """

    noise_multiplier: float
    sensitivity: float

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
        return _native.mf_gaussian_pld(
            self.noise_multiplier,
            self.sensitivity,
            config.to_native(),
        )


def mf_gaussian(noise_multiplier: float, sensitivity: float) -> MfGaussian:
    """Matrix factorization Gaussian mechanism.

    Computes the privacy guarantee for the entire MF training run as a
    single Gaussian mechanism with effective noise multiplier σ/S.

    The sensitivity should be pre-computed based on the MF strategy
    and participation pattern (single, min-sep).

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
        eps = proc.epsilon_at(1e-5)
    """
    return MfGaussian(noise_multiplier, sensitivity)
