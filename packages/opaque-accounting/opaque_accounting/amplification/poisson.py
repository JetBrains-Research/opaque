"""Poisson-subsampled Gaussian mechanism — standard DP-SGD step."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.mechanisms.nonprivate import NonPrivate
from opaque_accounting.transformations.adaclip import AdaClip

#: Mechanism types accepted by :func:`poisson`.
_Inner = Gaussian | AdaClip | NonPrivate


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
        from opaque_accounting.discretization import get_discretization

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
                # Tight: Theorem 1 z_eff folds both into one
                # Gaussian before amplification.
                return _native.poisson_gaussian_pld(
                    ac.effective_noise_multiplier,
                    self.sample_rate,
                    native_cfg,
                )
            case AdaClip(
                inner=NonPrivate() | Gaussian(noise_multiplier=0)
            ):
                return _native.non_private_pld(native_cfg)
            case _:
                raise TypeError(
                    "Poisson requires a Gaussian or AdaClip inner "
                    f"mechanism, got {type(self.inner).__name__}."
                )


def poisson(
    inner: _Inner,
    sample_rate: float,
) -> Poisson:
    """Poisson-subsampled Gaussian mechanism (standard DP-SGD step).

    Each training step selects examples independently with probability ``sample_rate``,
    computes gradients, clips them, adds Gaussian noise, and updates the model.

    This is the **standard DP-SGD mechanism** used in most deep learning privacy work.

    Args:
        inner: The base mechanism — :func:`gaussian` or an :func:`adaclip` transform.
        sample_rate: Probability of including each example (batch_size / dataset_size).

    Returns:
        A :class:`Poisson` process.

    Example::

        step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, (Gaussian, AdaClip, NonPrivate)):
        raise TypeError(
            f"poisson() requires a Gaussian, AdaClip, or NonPrivate "
            f"inner mechanism, got {type(inner).__name__}. "
            "Example: acc.poisson(acc.gaussian(nm), rate)"
        )
    return Poisson(inner=inner, sample_rate=sample_rate)
