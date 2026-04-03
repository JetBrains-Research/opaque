"""Parallel Poisson subsampling mechanism for multi-worker training."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.amplification.poisson import Poisson
from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.mechanisms.nonprivate import NonPrivate
from opaque_accounting.transformations.adaclip import AdaClip


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

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        from opaque_accounting.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )

        native_cfg = config.to_native()

        match self.inner:
            case Poisson(inner=NonPrivate()):
                return _native.eps_delta_pld(0.0, 1.0, native_cfg)
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
                inner=AdaClip(inner=Gaussian()) as ac,
                sample_rate=rate,
            ):
                # Tight: z_eff combines both into one Gaussian.
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
    inner: Gaussian | AdaClip,
    sample_rate: float,
    num_workers: int,
) -> ParallelPoisson:
    """Poisson sampling under parallel worker execution.

    When Poisson sampling runs on ``num_workers`` parallel workers independently,
    each worker samples its data independently. This causes unique examples to
    appear in multiple workers' batches — this mechanism accounts for that
    sampling duplication in the privacy calculation.

    This is the accounting mechanism for parallel training setups where:

    - Multi-worker PyTorch DataLoader with Poisson sampling on each worker
    - DDP training where each rank runs Poisson sampling independently
    - Any other parallel training where the same Poisson sampler runs on N
      workers

    Like :func:`poisson` and :func:`truncated_poisson`, this is a full wrapper:
    pass the inner Gaussian mechanism and sample rate directly.

    Args:
        inner: A Gaussian or AdaClip mechanism (from :func:`gaussian` or
            :func:`adaclip`).
        sample_rate: Probability of including each example, in (0, 1).
        num_workers: Number of parallel workers running Poisson sampling
            independently.

    Notes:
        Truncation is selected automatically inside the Rust implementation
        from query-time discretization settings
        (``log_x_mass_truncation_bound``) to balance speed and conservativeness.

    Returns:
        A :class:`ParallelPoisson` process.

    Example::

        step = acc.parallel_poisson(
            acc.gaussian(1.1), sample_rate=0.01, num_workers=4,
        )
        training = step * 500
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, (Gaussian, AdaClip, NonPrivate)):
        raise TypeError(
            f"parallel_poisson() requires a Gaussian, AdaClip, or NonPrivate "
            f"inner mechanism, got {type(inner).__name__}. "
            "Use: acc.parallel_poisson(acc.gaussian(nm), sample_rate=q, num_workers=k)"
        )
    poisson_inner = Poisson(inner=inner, sample_rate=sample_rate)
    return ParallelPoisson(inner=poisson_inner, num_workers=num_workers)
