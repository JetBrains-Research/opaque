"""Parallel Poisson subsampling mechanism for multi-worker training."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import _native

from opaque.accounting.amplification._poisson import Poisson
from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.mechanisms._gaussian import Gaussian
from opaque.accounting.mechanisms._mf_gaussian import MfGaussian
from opaque.accounting.mechanisms._nonprivate import NonPrivate
from opaque.accounting.transformations._adaclip import AdaClip
from opaque.accounting.transformations._second_moment import SecondMoment


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
        from opaque.accounting.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
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
                # Tight: z_eff combines both into one Gaussian.
                return _native.parallel_poisson_gaussian_pld(
                    ac.effective_noise_multiplier,
                    rate,
                    self.num_workers,
                    native_cfg,
                )
            case (
                Poisson(
                    inner=SecondMoment(inner=Gaussian(noise_multiplier=0)),
                )
                | Poisson(
                    inner=SecondMoment(inner=NonPrivate()),
                )
            ):
                return _native.non_private_pld(native_cfg)
            case Poisson(
                inner=SecondMoment(inner=Gaussian()) as sm,
                sample_rate=rate,
            ):
                # Tight: SecondMoment changes the joint sensitivity, so
                # parallel-Poisson amplification reduces to a Gaussian with
                # effective_nm = σ ÷ joint sensitivity.
                return _native.parallel_poisson_gaussian_pld(
                    sm.noise_multiplier / sm.sensitivity,
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
    inner: Gaussian | AdaClip | NonPrivate | SecondMoment,
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
        inner: A :class:`Gaussian`, :class:`AdaClip` (with Gaussian inner),
            :class:`NonPrivate`, or :class:`SecondMoment` (with Gaussian
            inner) mechanism — produced by the corresponding ``acc.gaussian``,
            ``acc.adaclip``, ``acc.nonprivate``, or ``acc.second_moment``
            factory.  ``SecondMoment`` with an ``MfGaussian`` inner is
            rejected — pass the matching MF amplification (``cyclic_poisson``,
            ``b_min_sep``) instead.
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
    if not isinstance(inner, (Gaussian, AdaClip, NonPrivate, SecondMoment)):
        raise TypeError(
            f"parallel_poisson() requires a Gaussian, AdaClip, NonPrivate, or "
            f"SecondMoment(Gaussian) inner mechanism, got {type(inner).__name__}. "
            "Use: acc.parallel_poisson(acc.gaussian(nm), sample_rate=q, num_workers=k)"
        )
    if isinstance(inner, SecondMoment) and isinstance(inner.inner, MfGaussian):
        raise TypeError(
            "parallel_poisson() does not support SecondMoment(MfGaussian) — pass "
            "an MF-aware amplification (cyclic_poisson, b_min_sep) instead."
        )
    poisson_inner = Poisson(inner=inner, sample_rate=sample_rate)
    return ParallelPoisson(inner=poisson_inner, num_workers=num_workers)
