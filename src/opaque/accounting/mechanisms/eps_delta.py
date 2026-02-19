"""Fixed (ε, δ)-DP mechanism."""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import DiscretizationConfig, DpProcess, Pld
from opaque.accounting.discretization import resolve_pld_config


@dataclass(frozen=True, slots=True)
class EpsDelta(DpProcess):
    """Fixed (ε, δ) mechanism."""

    epsilon: float
    delta: float
    config: DiscretizationConfig | None = field(default=None, repr=False)

    def pld(self) -> Pld:
        return _native.eps_delta_pld(self.epsilon, self.delta, config=self.config)


def eps_delta(
    epsilon: float,
    delta: float = 0.0,
    *,
    discretization: None | float | DiscretizationConfig = None,
) -> DpProcess:
    """Fixed (ε, δ)-DP guarantee (for composition with other mechanisms).

    Useful when you have an external mechanism with known privacy parameters
    that you want to compose with other tracked processes.

    Args:
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ. Default: 0.0 (pure ε-DP).
        discretization: PLD precision config (keyword-only).

    Returns:
        A :class:`DpProcess` wrapping an (ε, δ) PLD.

    Example::

        # External mechanism with (3.0, 1e-5)-DP
        external = acc.eps_delta(3.0, 1e-5)

        # Compose with DP-SGD
        training = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
        total = external | training
        eps = total.epsilon_at(1e-5)
    """
    config = resolve_pld_config(discretization)
    return EpsDelta(epsilon, delta, config=config)
