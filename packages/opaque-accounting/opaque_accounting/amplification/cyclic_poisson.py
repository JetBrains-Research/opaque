"""Cyclic Poisson amplification for BandMF.

When BandMF uses cyclic Poisson subsampling with band width b, the n
training rounds divide into k = ceil(n/b) independent groups.  Each
group is a Poisson-subsampled Gaussian mechanism.  The total privacy
is the k-fold composition of per-group PLDs.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import _make_native_config
from opaque_accounting.mechanisms.band_mf import BandMf


@dataclass(frozen=True, slots=True)
class CyclicPoisson(DpProcess):
    """Cyclic Poisson amplification for BandMF.

    Decomposes the BandMF mechanism into ``ceil(n_steps / bands)``
    independent groups, each analyzed as a Poisson-subsampled Gaussian.
    """

    inner: BandMf
    sample_rate: float

    @property
    def num_groups(self) -> int:
        """Number of independent groups: ceil(n_steps / bands)."""
        return math.ceil(self.inner.n_steps / self.inner.bands)

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        sensitivity = self.inner.sensitivity()
        effective_nm = self.inner.noise_multiplier / sensitivity
        return CgfPld(
            _native.cgf_poisson_gaussian_pld(effective_nm, self.sample_rate)
            .self_compose(self.num_groups)
        )

    def pmf(self, **kwargs: object) -> PmfPld:
        sensitivity = self.inner.sensitivity()
        effective_nm = self.inner.noise_multiplier / sensitivity

        per_group_pld = _native.poisson_gaussian_pld(
            effective_nm, self.sample_rate, _make_native_config(**kwargs)
        )
        return PmfPld(per_group_pld.self_compose(self.num_groups))


def cyclic_poisson(
    inner: BandMf,
    sample_rate: float,
) -> CyclicPoisson:
    """Cyclic Poisson amplification for BandMF.

    Wraps a :func:`~opaque.accounting.mechanisms.band_mf.band_mf` process
    with cyclic Poisson subsampling.

    Args:
        inner: A :class:`BandMf` mechanism (from :func:`band_mf`).
        sample_rate: Poisson sampling probability per group.

    Returns:
        A :class:`CyclicPoisson` process.

    Example::

        proc = acc.cyclic_poisson(
            acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10),
            sample_rate=0.01,
        )
        eps = proc.pmf().epsilon_at(1e-5)
    """
    if not isinstance(inner, BandMf):
        raise TypeError(
            f"cyclic_poisson() requires a BandMf inner mechanism, got "
            f"{type(inner).__name__}."
        )
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    return CyclicPoisson(inner=inner, sample_rate=sample_rate)
