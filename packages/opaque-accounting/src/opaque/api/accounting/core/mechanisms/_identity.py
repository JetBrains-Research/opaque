"""Identity mechanism — zero privacy loss, composition identity element."""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._pld_cache import pld_cache
from opaque.api.accounting.core.discretization import get_discretization

from .. import _native


@dataclass(frozen=True, slots=True)
class Identity(DpProcess):
    """Identity mechanism — zero privacy loss.

    Identity element of composition:
    ``Identity() | a`` → ``a`` and ``a | Identity()`` → ``a``.
    """

    @pld_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        num_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            num_mc_samples=num_mc_samples,
            seed=seed,
        )
        return _native.identity_pld(config.to_native())


def identity() -> DpProcess:
    """Identity mechanism (zero privacy loss).

    Useful as a placeholder or identity element in composition.

    Returns:
        An :class:`Identity` process (ε=0 for any δ).

    Example::

        # Identity has ε=0 for any δ
        proc = acc.identity()
        eps = proc.epsilon_at(1e-5)  # ~0
    """
    return Identity()
