"""Gaussian mechanism — base noise for DP-SGD."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import CgfPld, DpProcess, PmfPld
from opaque_accounting.discretization import _make_native_config


@dataclass(frozen=True, slots=True)
class Gaussian(DpProcess):
    """Gaussian mechanism — stores noise_multiplier, computes PLD on demand."""

    noise_multiplier: float

    @functools.lru_cache(maxsize=1)
    def cgf(self) -> CgfPld:
        return CgfPld(_native.cgf_gaussian_pld(self.noise_multiplier))

    def pmf(self, **kwargs: object) -> PmfPld:
        return PmfPld(_native.gaussian_pld(self.noise_multiplier, _make_native_config(**kwargs)))


def gaussian(noise_multiplier: float) -> Gaussian:
    """Gaussian mechanism with noise multiplier σ.

    The Gaussian mechanism adds noise ~ N(0, σ²) to sensitivity-1 queries.
    This is the base mechanism for standard DP-SGD.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            Larger values = more privacy, less utility.

    Returns:
        A :class:`Gaussian` process.

    Example::

        # Single Gaussian query
        proc = acc.gaussian(1.1)
        eps = proc.cgf().epsilon_at(1e-5)

        # Composed 1000 times
        training = acc.gaussian(1.1) * 1000
        eps = training.cgf().epsilon_at(1e-5)
    """
    return Gaussian(noise_multiplier)
