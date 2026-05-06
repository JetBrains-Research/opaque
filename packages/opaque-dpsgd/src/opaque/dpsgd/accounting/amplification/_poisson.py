"""Poisson-subsampled Gaussian mechanism — standard DP-SGD step."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.accounting import _native

from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.mechanisms._nonprivate import NonPrivate
from opaque.accounting.transformations._second_moment import SecondMoment
from opaque.dpsgd.accounting.mechanisms._adaclip import AdaClip
from opaque.dpsgd.accounting.mechanisms._gaussian import Gaussian

#: Mechanism types accepted by :func:`poisson`.
_Inner = Gaussian | AdaClip | NonPrivate | SecondMoment


@dataclass(frozen=True, slots=True)
class Poisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism."""

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
                return _native.poisson_gaussian_pld(nm, self.sample_rate, native_cfg)
            case AdaClip(inner=Gaussian()) as ac:
                return _native.poisson_gaussian_pld(
                    ac.effective_noise_multiplier,
                    self.sample_rate,
                    native_cfg,
                )
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
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
                # Tight: SecondMoment changes the joint sensitivity, so
                # amplification reduces to Poisson on a Gaussian with
                # effective_nm = σ ÷ joint sensitivity.
                return _native.poisson_gaussian_pld(
                    sm.noise_multiplier / sm.sensitivity,
                    self.sample_rate,
                    native_cfg,
                )
            case _:
                raise TypeError(
                    "Poisson requires a Gaussian, AdaClip(Gaussian), "
                    "SecondMoment(Gaussian), SecondMoment(AdaClip(Gaussian)), "
                    "or NonPrivate inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def poisson(
    inner: _Inner,
    sample_rate: float,
) -> Poisson:
    """Poisson-subsampled Gaussian mechanism (standard DP-SGD step).

    Args:
        inner: The base mechanism — :func:`gaussian`, :func:`adaclip`,
            or :func:`second_moment` (with Gaussian inner).
        sample_rate: Probability of including each example (batch_size / dataset_size).

    Returns:
        A :class:`Poisson` process.

    Example::

        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    match inner:
        case Gaussian() | AdaClip() | NonPrivate():
            pass
        case SecondMoment(inner=Gaussian() | NonPrivate() | AdaClip()):
            pass
        case _:
            raise TypeError(
                f"poisson() requires a Gaussian, AdaClip, NonPrivate, or "
                f"SecondMoment(Gaussian) inner mechanism, got {type(inner).__name__}. "
                "Example: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), rate)"
            )
    return Poisson(inner=inner, sample_rate=sample_rate)
