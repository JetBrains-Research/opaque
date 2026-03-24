"""Poisson-subsampled Gaussian mechanism — standard DP-SGD step."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, Pld
from opaque_accounting.discretization import DiscretizationConfig
from opaque_accounting.mechanisms.gaussian import Gaussian
from opaque_accounting.mechanisms.rectified_gaussian import RectifiedGaussian
from opaque_accounting.mechanisms.truncated_gaussian import TruncatedGaussian
from opaque_accounting.transformations.adaclip import AdaClip

#: Mechanism types accepted by :func:`poisson`.
_Inner = Gaussian | RectifiedGaussian | TruncatedGaussian | AdaClip


@dataclass(frozen=True, slots=True)
class Poisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism."""

    inner: _Inner
    sample_rate: float

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> Pld:
        match self.inner:
            case Gaussian(noise_multiplier=nm):
                return _native.cgf_poisson_gaussian_pld(nm, self.sample_rate)
            case AdaClip():
                z_eff = self.inner.effective_noise_multiplier
                return _native.cgf_poisson_gaussian_pld(z_eff, self.sample_rate)
            case _:
                raise NotImplementedError(
                    f"CGF not available for Poisson-subsampled "
                    f"{type(self.inner).__name__}"
                )

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> Pld:
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
            case AdaClip():
                z_eff = self.inner.effective_noise_multiplier
                return _native.poisson_gaussian_pld(
                    z_eff, self.sample_rate, config.to_native()
                )
            case _:
                raise TypeError(
                    "Poisson requires a Gaussian, RectifiedGaussian, "
                    "TruncatedGaussian, or AdaClip inner mechanism, got "
                    f"{type(self.inner).__name__}."
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

        step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)
        training = step * 1000
        eps = training.cgf().epsilon_at(1e-5)
    """
    if not isinstance(inner, (Gaussian, RectifiedGaussian, TruncatedGaussian, AdaClip)):
        raise TypeError(
            f"poisson() requires a Gaussian, RectifiedGaussian, TruncatedGaussian, "
            f"or AdaClip inner mechanism, got {type(inner).__name__}."
        )
    return Poisson(inner=inner, sample_rate=sample_rate)
