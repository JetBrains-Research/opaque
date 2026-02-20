"""Parallel Poisson subsampling mechanism for multi-worker training."""

from __future__ import annotations

from dataclasses import dataclass

import opaque_accounting as _native

from opaque.accounting.amplification.poisson import Poisson
from opaque.accounting.base import DpProcess, Pld
from opaque.accounting.mechanisms.gaussian import Gaussian
from opaque.accounting.transformations.adaclip import AdaClip


@dataclass(frozen=True, slots=True)
class ParallelPoisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism under parallel worker execution.

    When Poisson sampling runs independently on multiple workers (e.g., in
    multi-worker PyTorch DataLoader or DDP training), unique examples can
    appear in multiple workers' samples. This mechanism accounts for that
    sampling duplication in the privacy calculation.
    """

    inner: Poisson
    num_workers: int

    def pld(self) -> Pld:
        match self.inner:
            case Poisson(
                inner=Gaussian(noise_multiplier=nm, config=cfg),
                sample_rate=rate,
            ):
                return _native.accumulated_poisson_gaussian_pld(
                    nm,
                    rate,
                    self.num_workers,
                    config=cfg,
                )
            case Poisson(
                inner=AdaClip(
                    inner=Gaussian(noise_multiplier=nm, config=cfg),
                    quantile_noise_std=quantile_noise_std,
                ),
                sample_rate=rate,
            ):
                s = _native.combined_sensitivity(nm, quantile_noise_std)
                z_eff = 1.0 / s
                return _native.accumulated_poisson_gaussian_pld(
                    z_eff,
                    rate,
                    self.num_workers,
                    config=cfg,
                )
            case _:
                raise TypeError(
                    "ParallelPoisson requires a Poisson inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "ParallelPoisson",
            "num_workers": self.num_workers,
            "inner": self.inner.state_dict(),
        }


def parallel_poisson(
    inner: Poisson,
    num_workers: int,
) -> ParallelPoisson:
    """Poisson sampling under parallel worker execution.

    When Poisson sampling runs on ``num_workers`` parallel workers independently,
    each worker samples its data independently. This causes unique examples to
    appear in multiple workers' batches—this mechanism accounts for that
    sampling duplication in the privacy calculation.

    This is the accounting mechanism for parallel training setups where:
    - Multi-worker PyTorch DataLoader with Poisson sampling on each worker
    - DDP training where each rank runs Poisson sampling independently
    - Any other parallel training where the same Poisson sampler runs on N workers

    Args:
        inner: A Poisson process (from :func:`poisson`).
        num_workers: Number of parallel workers running Poisson sampling independently.

    Returns:
        A :class:`ParallelPoisson` process.

    Example::

        # Multi-worker training with 4 workers
        step = acc.parallel_poisson(
            acc.poisson(acc.gaussian(1.1), 0.01),
            num_workers=4,
        )
        training = step * 500
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, Poisson):
        raise TypeError(
            f"parallel_poisson() requires a Poisson inner mechanism, got {type(inner).__name__}. "
            "Use: acc.parallel_poisson(acc.poisson(acc.gaussian(nm), rate), num_workers=k)"
        )
    return ParallelPoisson(inner=inner, num_workers=num_workers)
