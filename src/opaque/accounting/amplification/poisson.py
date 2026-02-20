"""Poisson-subsampled Gaussian mechanism — standard DP-SGD step."""

from __future__ import annotations

from dataclasses import dataclass

import opaque_accounting as _native

from opaque.accounting.base import DpProcess, Pld
from opaque.accounting.mechanisms.gaussian import Gaussian
from opaque.accounting.transformations.adaclip import AdaClip


@dataclass(frozen=True, slots=True)
class Poisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism."""

    inner: Gaussian | AdaClip
    sample_rate: float

    def pld(self) -> Pld:
        match self.inner:
            case Gaussian(noise_multiplier=nm, config=cfg):
                return _native.poisson_gaussian_pld(
                    nm, self.sample_rate, config=cfg
                )
            case AdaClip(
                inner=Gaussian(noise_multiplier=nm, config=cfg),
                quantile_noise_std=quantile_noise_std,
            ):
                s = _native.combined_sensitivity(nm, quantile_noise_std)
                z_eff = 1.0 / s
                return _native.poisson_gaussian_pld(
                    z_eff, self.sample_rate, config=cfg
                )
            case _:
                raise TypeError(
                    "Poisson requires a Gaussian or AdaClip inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "Poisson",
            "sample_rate": self.sample_rate,
            "inner": self.inner.state_dict(),
        }


def poisson(
    inner: Gaussian | AdaClip,
    sample_rate: float,
) -> Poisson:
    """Poisson-subsampled Gaussian mechanism (standard DP-SGD step).

    Each training step selects examples independently with probability ``sample_rate``,
    computes gradients, clips them, adds Gaussian noise, and updates the model.

    This is the **standard DP-SGD mechanism** used in most deep learning privacy work.

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`) or
            an :func:`adaclip` transform applied to a Gaussian.
        sample_rate: Probability of including each example (batch_size / dataset_size).

    Returns:
        A :class:`Poisson` process.

    Example::

        # One training step
        step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)

        # 1000 steps of training
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, (Gaussian, AdaClip)):
        raise TypeError(
            f"poisson() requires a Gaussian or AdaClip inner mechanism, got {type(inner).__name__}. "
            "Use: acc.poisson(acc.gaussian(noise_multiplier), sample_rate)"
        )
    return Poisson(inner=inner, sample_rate=sample_rate)
