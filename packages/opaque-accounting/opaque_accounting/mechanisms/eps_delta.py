"""Fixed (ε, δ)-DP mechanism."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import DpProcess, PmfPld
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class EpsDelta(DpProcess):
    """Fixed (ε, δ) mechanism."""

    epsilon: float
    delta: float

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> PmfPld:
        return PmfPld(_native.eps_delta_pld(self.epsilon, self.delta, config.to_native()))


def eps_delta(epsilon: float, delta: float = 0.0) -> DpProcess:
    """Fixed (ε, δ)-DP guarantee (for composition with other mechanisms).

    Useful when you have an external mechanism with known privacy parameters
    that you want to compose with other tracked processes.

    Args:
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ. Default: 0.0 (pure ε-DP).

    Returns:
        A :class:`DpProcess` wrapping an (ε, δ) PLD.

    Example::

        external = acc.eps_delta(3.0, 1e-5)
        training = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
        total = external | training
        eps = total.pmf(acc.DiscretizationConfig()).epsilon_at(1e-5)
    """
    return EpsDelta(epsilon, delta)
