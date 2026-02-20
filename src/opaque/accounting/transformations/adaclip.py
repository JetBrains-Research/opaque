"""Adaptive clipping transformation (Andrew et al. 2021)."""

from __future__ import annotations

from dataclasses import dataclass

import opaque_accounting as _native

from opaque.accounting.base import DpProcess, Pld
from opaque.accounting.mechanisms.gaussian import Gaussian


@dataclass(frozen=True, slots=True)
class AdaClip(DpProcess):
    """Adaptive clipping transformation (Andrew et al. 2021)."""

    inner: Gaussian
    quantile_noise_std: float

    def pld(self) -> Pld:
        match self.inner:
            case Gaussian(noise_multiplier=nm, config=cfg):
                s = _native.combined_sensitivity(nm, self.quantile_noise_std)
                z_eff = 1.0 / s
                return _native.gaussian_pld(z_eff, config=cfg)
            case _:
                raise TypeError(
                    "AdaClip requires a Gaussian inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "AdaClip",
            "quantile_noise_std": self.quantile_noise_std,
            "inner": self.inner.state_dict(),
        }


def adaclip(
    inner: Gaussian,
    quantile_noise_std: float,
) -> AdaClip:
    """Adaptive clipping mechanism (Andrew et al. 2021).

    Adaptive clipping adjusts the clipping threshold based on the empirical
    distribution of gradient norms. The quantile estimation uses a noisy mechanism,
    adding extra privacy cost.

    The total privacy cost uses the combined sensitivity formula:
    ``z_eff = 1 / sqrt(1/z² + 1/(4·σ_b²))``

    where z is the base noise multiplier and σ_b is the quantile noise std.

    The result is an :class:`AdaClip` process with the effective noise multiplier,
    so it can be composed with :func:`poisson` or :func:`truncated_poisson`::

        step = acc.poisson(acc.adaclip(acc.gaussian(1.1), 50.0), 0.01)

    Args:
        inner: The base Gaussian mechanism (from :func:`gaussian`).
        quantile_noise_std: Noise std for quantile estimation.
            Larger = more private quantile, less accurate clipping.

    Returns:
        An :class:`AdaClip` process.

    Example::

        step = acc.adaclip(acc.gaussian(1.1), quantile_noise_std=50.0)
        eps = step.epsilon_at(1e-5)
    """
    if not isinstance(inner, Gaussian):
        raise TypeError(
            f"adaclip() requires a Gaussian inner mechanism, got {type(inner).__name__}."
        )
    return AdaClip(inner=inner, quantile_noise_std=quantile_noise_std)
