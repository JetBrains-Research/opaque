"""Truncated Poisson-subsampled Gaussian mechanism — production DP-SGD."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.accounting import _native

from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.mechanisms._nonprivate import NonPrivate
from opaque.accounting.transformations._second_moment import SecondMoment
from opaque.dpsgd.accounting.mechanisms._adaclip import AdaClip
from opaque.dpsgd.accounting.mechanisms._gaussian import Gaussian

#: Mechanism types accepted by :func:`truncated_poisson`.
_Inner = Gaussian | AdaClip | NonPrivate | SecondMoment


@dataclass(frozen=True, slots=True)
class TruncatedPoisson(DpProcess):
    """Truncated Poisson-subsampled Gaussian mechanism."""

    inner: _Inner
    sample_rate: float
    batch_size_cap: int
    dataset_size: int

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
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return _native.non_private_pld(native_cfg)
            case Gaussian(noise_multiplier=nm):
                return _native.truncated_poisson_gaussian_pld(
                    nm,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    native_cfg,
                )
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
            case AdaClip(inner=Gaussian()) as ac:
                return _native.truncated_poisson_gaussian_pld(
                    ac.effective_noise_multiplier,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    native_cfg,
                )
            case (
                SecondMoment(inner=Gaussian(noise_multiplier=0))
                | SecondMoment(inner=NonPrivate())
                | SecondMoment(inner=AdaClip(inner=Gaussian(noise_multiplier=0)))
                | SecondMoment(inner=AdaClip(inner=NonPrivate()))
            ):
                return _native.non_private_pld(native_cfg)
            case (
                SecondMoment(inner=Gaussian())
                | SecondMoment(inner=AdaClip(inner=Gaussian()))
            ) as sm:
                return _native.truncated_poisson_gaussian_pld(
                    sm.noise_multiplier / sm.sensitivity,
                    self.sample_rate,
                    self.batch_size_cap,
                    self.dataset_size,
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "TruncatedPoisson requires a Gaussian, AdaClip(Gaussian), "
                    "SecondMoment(Gaussian), SecondMoment(AdaClip(Gaussian)), "
                    "or NonPrivate inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def truncated_poisson(
    inner: _Inner,
    sample_rate: float,
    batch_size_cap: int,
    dataset_size: int,
) -> DpProcess:
    """Truncated Poisson sampling (production DP-SGD with capped batch size).

    Args:
        inner: The base Gaussian mechanism, an :func:`adaclip` transform, or a
            :func:`second_moment` transform (with Gaussian inner).
        sample_rate: Probability of including each example.
        batch_size_cap: Maximum batch size.
        dataset_size: Total number of examples in the dataset.

    Returns:
        A :class:`TruncatedPoisson` process.

    Example::

        n, batch = 50_000, 250
        step = dpsgd_acc.truncated_poisson(dpsgd_acc.gaussian(0.8), batch / n, batch, n)
        eps = (step * 1000).epsilon_at(1e-5)
    """
    match inner:
        case Gaussian() | AdaClip() | NonPrivate():
            pass
        case SecondMoment(inner=Gaussian() | NonPrivate() | AdaClip()):
            pass
        case _:
            raise TypeError(
                f"truncated_poisson() requires a Gaussian, AdaClip, NonPrivate, or "
                f"SecondMoment(Gaussian) inner mechanism, got {type(inner).__name__}."
            )
    return TruncatedPoisson(
        inner=inner,
        sample_rate=sample_rate,
        batch_size_cap=batch_size_cap,
        dataset_size=dataset_size,
    )
