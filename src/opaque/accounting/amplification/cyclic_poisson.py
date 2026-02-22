"""Cyclic Poisson amplification for BandMF.

When BandMF uses cyclic Poisson subsampling with band width b, the n
training rounds divide into k = ceil(n/b) independent groups.  Each
group is a Poisson-subsampled Gaussian mechanism.  The total privacy
is the k-fold composition of per-group PLDs.

This wrapper is transparent to the inner :class:`BandMf`: it reaches
into the mechanism's structural parameters (bands, n_steps,
noise_multiplier) and recomputes the privacy analysis from scratch,
using the cyclic structure to determine both the per-group sensitivity
and the composition count.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import opaque_accounting as _native

from opaque.accounting.base import (
    DpProcess,
    Pld,
)
from opaque.accounting.discretization import (
    get_discretization,
)
from opaque.accounting.mechanisms.band_mf import BandMf


@dataclass(frozen=True, slots=True)
class CyclicPoisson(DpProcess):
    """Cyclic Poisson amplification for BandMF.

    Decomposes the BandMF mechanism into ``ceil(n_steps / bands)``
    independent groups, each analyzed as a Poisson-subsampled Gaussian.
    The per-group sensitivity is the single-participation sensitivity
    of the inner BandMf's encoder matrix (typically 1.0 for normalized
    Toeplitz strategies).

    This is NOT a generic wrapper — it only accepts :class:`BandMf`
    because the amplification argument relies on the banded structure.
    """

    inner: BandMf
    sample_rate: float

    @property
    def num_groups(self) -> int:
        """Number of independent groups: ceil(n_steps / bands)."""
        return math.ceil(self.inner.n_steps / self.inner.bands)

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

        # Transparent access to inner BandMf parameters
        sensitivity = self.inner.sensitivity()
        effective_nm = self.inner.noise_multiplier / sensitivity

        per_group_pld = _native.poisson_gaussian_pld(
            effective_nm, self.sample_rate, config.to_native()
        )
        return per_group_pld.self_compose(self.num_groups)


def cyclic_poisson(
    inner: BandMf,
    sample_rate: float,
) -> CyclicPoisson:
    """Cyclic Poisson amplification for BandMF.

    Wraps a :func:`~opaque.accounting.mechanisms.band_mf.band_mf` process
    with cyclic Poisson subsampling.  The ``n_steps`` training rounds are
    divided into ``ceil(n_steps / bands)`` independent groups, each
    analyzed as a Poisson-subsampled Gaussian mechanism.

    This follows the same pattern as :func:`~opaque.accounting.amplification.poisson`:
    the wrapper is typed to only accept :class:`BandMf` and reaches into its
    parameters transparently.

    Args:
        inner: A :class:`BandMf` mechanism (from :func:`band_mf`).
        sample_rate: Poisson sampling probability per group
            (typically ``bands * batch_size / dataset_size``).

    Returns:
        A :class:`CyclicPoisson` process.

    Example::

        import opaque.accounting as acc

        proc = acc.cyclic_poisson(
            acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10),
            sample_rate=0.01,
        )
        eps = proc.epsilon_at(1e-5)
    """
    if not isinstance(inner, BandMf):
        raise TypeError(
            f"cyclic_poisson() requires a BandMf inner mechanism, got "
            f"{type(inner).__name__}. "
            "Use: acc.cyclic_poisson(acc.band_mf(noise_multiplier, n_steps, bands), sample_rate)"
        )
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    return CyclicPoisson(inner=inner, sample_rate=sample_rate)
