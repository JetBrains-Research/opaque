"""Fixed (ε, δ)-DP mechanism."""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import (
    DpProcess,
    Pld,
)
from opaque.accounting.discretization import (
    get_discretization,
)


@dataclass(frozen=True, slots=True)
class EpsDelta(DpProcess):
    """Fixed (ε, δ) mechanism."""

    epsilon: float
    delta: float

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )
        return _native.eps_delta_pld(self.epsilon, self.delta, config.to_native())


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

        # External mechanism with (3.0, 1e-5)-DP
        external = acc.eps_delta(3.0, 1e-5)

        # Compose with DP-SGD
        training = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
        total = external | training
        eps = total.epsilon_at(1e-5)
    """
    return EpsDelta(epsilon, delta)
