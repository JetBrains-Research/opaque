"""Parallel Poisson subsampling mechanism for multi-worker training."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.amplification._poisson import Poisson
from opaque.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian


@dataclass(frozen=True, slots=True)
class ParallelPoisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism under parallel worker execution."""

    inner: Poisson
    num_workers: int

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        from opaque.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )

        native_cfg = config.to_native()

        match self.inner:
            case Poisson(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
            case Poisson(
                inner=Gaussian(noise_multiplier=nm),
                sample_rate=rate,
            ):
                return _native.parallel_poisson_gaussian_pld(
                    nm,
                    rate,
                    self.num_workers,
                    native_cfg,
                )
            case Poisson(
                inner=AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)),
            ):
                return _native.non_private_pld(native_cfg)
            case Poisson(
                inner=AdaClip(inner=Gaussian()) as ac,
                sample_rate=rate,
            ):
                return _native.parallel_poisson_gaussian_pld(
                    ac.effective_noise_multiplier,
                    rate,
                    self.num_workers,
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "ParallelPoisson requires a Poisson inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def parallel_poisson(
    inner: Gaussian | AdaClip | NonPrivate,
    sample_rate: float,
    num_workers: int,
) -> ParallelPoisson:
    """Poisson sampling under parallel worker execution.

    Args:
        inner: A :class:`Gaussian`, :class:`AdaClip`, or :class:`NonPrivate` mechanism.
        sample_rate: Probability of including each example, in (0, 1).
        num_workers: Number of parallel workers running Poisson sampling
            independently.

    Returns:
        A :class:`ParallelPoisson` process.

    Example::

        step = dpsgd_acc.parallel_poisson(
            dpsgd_acc.gaussian(1.1), sample_rate=0.01, num_workers=4,
        )
        eps = (step * 500).epsilon_at(1e-5)
    """
    match inner:
        case Gaussian() | AdaClip() | NonPrivate():
            pass
        case _:
            raise TypeError(
                "parallel_poisson() requires a Gaussian, AdaClip, or NonPrivate "
                f"inner mechanism, got {type(inner).__name__}."
            )
    poisson_inner = Poisson(inner=inner, sample_rate=sample_rate)
    return ParallelPoisson(inner=poisson_inner, num_workers=num_workers)
