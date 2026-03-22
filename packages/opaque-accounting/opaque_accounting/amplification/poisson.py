"""Poisson-subsampled Gaussian mechanism — standard DP-SGD step."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.mechanisms.mip_gaussian import MipGaussian
from opaque_accounting.mechanisms.rectified_gaussian import RectifiedGaussian
from opaque_accounting.mechanisms.truncated_gaussian import TruncatedGaussian
from opaque_accounting.transformations.adaclip import AdaClip

#: Mechanism types accepted by :func:`poisson`.
_Inner = Gaussian | RectifiedGaussian | TruncatedGaussian | MipGaussian | AdaClip


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

        match self.inner:
            case Gaussian(noise_multiplier=nm):
                return _native.poisson_gaussian_pld(
                    nm, self.sample_rate, config.to_native()
                )
            case RectifiedGaussian(noise_multiplier=nm, radius=r):
                return _native.poisson_rectified_gaussian_pld(
                    nm, r, self.sample_rate, config.to_native()
                )
            case TruncatedGaussian(noise_multiplier=nm, radius=r):
                return _native.poisson_truncated_gaussian_pld(
                    nm, r, self.sample_rate, config.to_native()
                )
            case MipGaussian(noise_multiplier=nm, sensitivities=s, weights=w):
                return _native.poisson_mip_gaussian_pld(
                    nm,
                    self.sample_rate,
                    list(s),
                    list(w),
                    config.to_native(),
                )
            case AdaClip(inner=MipGaussian() as mg):
                z_eff = self.inner.effective_noise_multiplier
                return _native.poisson_mip_gaussian_pld(
                    z_eff,
                    self.sample_rate,
                    list(mg.sensitivities),
                    list(mg.weights),
                    config.to_native(),
                )
            case AdaClip():
                z_eff = self.inner.effective_noise_multiplier
                return _native.poisson_gaussian_pld(
                    z_eff, self.sample_rate, config.to_native()
                )
            case _:
                raise TypeError(
                    "Poisson requires a Gaussian, RectifiedGaussian, "
                    "TruncatedGaussian, MipGaussian, or AdaClip inner "
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
        inner: The base mechanism — :func:`gaussian`, :func:`rectified_gaussian`,
            :func:`truncated_gaussian`, or an :func:`adaclip` transform.
        sample_rate: Probability of including each example (batch_size / dataset_size).

    Returns:
        A :class:`Poisson` process.

    Example::

        # Standard Gaussian
        step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)

        # Tighter bounds with rectified Gaussian
        step = acc.poisson(acc.rectified_gaussian(1.1, 5.0), sample_rate=0.01)

        # Tightest bounds with truncated Gaussian
        step = acc.poisson(acc.truncated_gaussian(1.1, 5.0), sample_rate=0.01)

        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(
        inner, (Gaussian, RectifiedGaussian, TruncatedGaussian, MipGaussian, AdaClip)
    ):
        raise TypeError(
            f"poisson() requires a Gaussian, RectifiedGaussian, TruncatedGaussian, "
            f"MipGaussian, or AdaClip inner mechanism, got {type(inner).__name__}. "
            "Examples: acc.poisson(acc.gaussian(nm), rate), "
            "acc.poisson(acc.mip_gaussian(nm, norms), rate)"
        )
    return Poisson(inner=inner, sample_rate=sample_rate)
