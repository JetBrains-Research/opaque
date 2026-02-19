"""Gaussian mechanism — base noise for DP-SGD."""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import DiscretizationConfig, DpProcess, Pld
from opaque.accounting.discretization import resolve_pld_config


@dataclass(frozen=True, slots=True)
class Gaussian(DpProcess):
    """Gaussian mechanism — stores noise_multiplier, computes PLD on demand."""

    noise_multiplier: float
    config: DiscretizationConfig | None = field(default=None, repr=False)

    def pld(self) -> Pld:
        return _native.gaussian_pld(self.noise_multiplier, config=self.config)


def gaussian(
    noise_multiplier: float,
    *,
    discretization: None | float | DiscretizationConfig = None,
) -> Gaussian:
    """Gaussian mechanism with noise multiplier σ.

    The Gaussian mechanism adds noise ~ N(0, σ²) to sensitivity-1 queries.
    This is the base mechanism for standard DP-SGD.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            Larger values = more privacy, less utility.
        discretization: PLD precision config (keyword-only). Can be:
            - None: use module default (see :func:`set_discretization`)
            - float: use as grid spacing
            - DiscretizationConfig: full config

    Returns:
        A :class:`Gaussian` process.

    Example::

        # Single Gaussian query
        proc = acc.gaussian(1.1)
        eps = proc.epsilon_at(1e-5)

        # Composed 1000 times
        training = acc.gaussian(1.1) * 1000
        eps = training.epsilon_at(1e-5)
    """
    config = resolve_pld_config(discretization)
    return Gaussian(noise_multiplier, config=config)
