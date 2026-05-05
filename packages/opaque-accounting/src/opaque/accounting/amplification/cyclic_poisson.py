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
from dataclasses import dataclass

from .. import _native

from opaque.accounting.base import (
    DpProcess,
    Pld,
)
from opaque.accounting.discretization import (
    get_discretization,
)
from opaque.accounting.mechanisms.band_mf import BandMf
from opaque.accounting.transformations.second_moment import SecondMoment

#: Mechanism types accepted by :func:`cyclic_poisson`.
_Inner = BandMf | SecondMoment


@dataclass(frozen=True, slots=True)
class CyclicPoisson(DpProcess):
    """Cyclic Poisson amplification for BandMF.

    Decomposes the BandMF mechanism into ``num_groups`` independent
    groups, each analyzed as a Poisson-subsampled Gaussian.
    """

    inner: _Inner
    sample_rate: float

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

        match self.inner:
            case SecondMoment(inner=BandMf()) as second:
                effective_nm = second.noise_multiplier / second.sensitivity
                num_groups = second.num_groups
            case BandMf():
                effective_nm = self.inner.noise_multiplier / self.inner.sensitivity
                num_groups = self.inner.num_groups
            case _:
                raise TypeError(
                    f"CyclicPoisson requires BandMf or SecondMoment(BandMf) inner, "
                    f"got {type(self.inner).__name__}."
                )

        per_group_pld = _native.poisson_gaussian_pld(
            effective_nm, self.sample_rate, config.to_native()
        )
        return per_group_pld.self_compose(num_groups)


def cyclic_poisson(
    inner: _Inner,
    sample_rate: float,
) -> CyclicPoisson:
    """Cyclic Poisson amplification for BandMF.

    Wraps a BandMF mechanism with cyclic Poisson subsampling.
    The training rounds are divided into ``inner.num_groups`` independent
    groups, each analyzed as a Poisson-subsampled Gaussian mechanism.

    Args:
        inner: A :class:`BandMf` mechanism with ``num_groups`` set.
        sample_rate: Poisson sampling probability per group
            (typically ``bands * batch_size / dataset_size``).

    Returns:
        A :class:`CyclicPoisson` process.

    Example::

        import opaque.accounting as acc

        proc = acc.cyclic_poisson(
            acc.band_mf(1.0, sensitivity=1.0, num_groups=100),
            sample_rate=0.01,
        )
        eps = proc.epsilon_at(1e-5)
    """
    if not isinstance(inner, (BandMf, SecondMoment)):
        raise TypeError(
            f"cyclic_poisson() requires a BandMf or SecondMoment(BandMf) inner mechanism, got "
            f"{type(inner).__name__}. "
            "Use: acc.cyclic_poisson(acc.band_mf(nm, sensitivity, num_groups), sample_rate)"
        )
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    if inner.num_groups < 1:
        raise ValueError(f"inner.num_groups must be >= 1, got {inner.num_groups}")

    return CyclicPoisson(inner=inner, sample_rate=sample_rate)
