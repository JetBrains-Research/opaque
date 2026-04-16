"""MF Gaussian base mechanism — correlated noise for MF-DP.

Provides the base :class:`MfGaussian` type used by all matrix factorization
mechanisms. The privacy reduces to a single Gaussian mechanism with effective
noise multiplier σ/S.

Per-method subclasses live in their own modules:

- :mod:`~opaque_accounting.mechanisms.band_mf` — :class:`BandMf`
- :mod:`~opaque_accounting.mechanisms.blt` — :class:`Blt`
- :mod:`~opaque_accounting.mechanisms.lambda_cgd` — :class:`LambdaCgd`
- :mod:`~opaque_accounting.mechanisms.bisr` — :class:`Bisr`
- :mod:`~opaque_accounting.mechanisms.bsr` — :class:`Bsr`
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
    """MF Gaussian mechanism — internal base type.

    Represents the privacy cost of an entire matrix factorization DP
    training process. The privacy reduces to a single Gaussian mechanism
    with effective noise multiplier σ/S.

    Use one of the per-method factories instead of constructing directly:

    - :func:`~opaque_accounting.mechanisms.band_mf.band_mf` → :class:`~opaque_accounting.mechanisms.band_mf.BandMf`
    - :func:`~opaque_accounting.mechanisms.blt.blt` → :class:`~opaque_accounting.mechanisms.blt.Blt`
    - :func:`~opaque_accounting.mechanisms.lambda_cgd.lambda_cgd` → :class:`~opaque_accounting.mechanisms.lambda_cgd.LambdaCgd`
    - :func:`~opaque_accounting.mechanisms.bisr.bisr` → :class:`~opaque_accounting.mechanisms.bisr.Bisr`
    - :func:`~opaque_accounting.mechanisms.bsr.bsr` → :class:`~opaque_accounting.mechanisms.bsr.Bsr`
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
