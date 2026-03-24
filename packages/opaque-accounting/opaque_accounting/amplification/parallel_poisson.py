"""Parallel Poisson subsampling mechanism for multi-worker training."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.amplification.poisson import Poisson
from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import DiscretizationConfig
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.transformations.adaclip import AdaClip


@dataclass(frozen=True, slots=True)
class ParallelPoisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism under parallel worker execution.

    When Poisson sampling runs independently on multiple workers, unique
    examples can appear in multiple workers' samples. This mechanism
    accounts for that sampling duplication in the privacy calculation.
    """

    inner: Poisson
    num_workers: int

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        match self.inner:
            case Poisson(inner=Gaussian(noise_multiplier=nm), sample_rate=rate):
                return CgfPld(_native.cgf_parallel_poisson_gaussian_pld(
                    nm, rate, self.num_workers
                ))
            case Poisson(inner=AdaClip() as ac, sample_rate=rate):
                z_eff = ac.effective_noise_multiplier
                return CgfPld(_native.cgf_parallel_poisson_gaussian_pld(
                    z_eff, rate, self.num_workers
                ))
            case _:
                raise NotImplementedError(
                    f"CGF not available for ParallelPoisson with "
                    f"{type(self.inner).__name__}"
                )

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> PmfPld:
        match self.inner:
            case Poisson(
                inner=Gaussian(noise_multiplier=nm),
                sample_rate=rate,
            ):
                return PmfPld(_native.parallel_poisson_gaussian_pld(
                    nm,
                    rate,
                    self.num_workers,
                    config.to_native(),
                ))
            case Poisson(
                inner=AdaClip() as ac,
                sample_rate=rate,
            ):
                z_eff = ac.effective_noise_multiplier
                return PmfPld(_native.parallel_poisson_gaussian_pld(
                    z_eff,
                    rate,
                    self.num_workers,
                    config.to_native(),
                ))
            case _:
                raise TypeError(
                    "ParallelPoisson requires a Poisson inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def parallel_poisson(
    inner: Gaussian | AdaClip,
    sample_rate: float,
    num_workers: int,
) -> ParallelPoisson:
    """Poisson sampling under parallel worker execution.

    When Poisson sampling runs on ``num_workers`` parallel workers independently,
    each worker samples its data independently. This causes unique examples to
    appear in multiple workers' batches — this mechanism accounts for that
    sampling duplication in the privacy calculation.

    Args:
        inner: A Gaussian or AdaClip mechanism.
        sample_rate: Probability of including each example, in (0, 1].
        num_workers: Number of parallel workers.

    Returns:
        A :class:`ParallelPoisson` process.

    Example::

        step = acc.parallel_poisson(
            acc.gaussian(1.1), sample_rate=0.01, num_workers=4,
        )
        training = step * 500
        eps = training.pmf(acc.DiscretizationConfig()).epsilon_at(1e-5)
    """
    if not isinstance(inner, (Gaussian, AdaClip)):
        raise TypeError(
            f"parallel_poisson() requires a Gaussian or AdaClip inner mechanism, "
            f"got {type(inner).__name__}."
        )
    poisson_inner = Poisson(inner=inner, sample_rate=sample_rate)
    return ParallelPoisson(inner=poisson_inner, num_workers=num_workers)
