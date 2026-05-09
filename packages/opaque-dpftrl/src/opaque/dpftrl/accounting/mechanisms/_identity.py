"""MF identity mechanism — uncorrelated Gaussian under the MF training API.

Pairs with :func:`opaque.dpftrl.noise.identity_strategy` (encoder
:math:`C^{-1} = I`, sensitivity ``1.0``).  As a *mechanism*, the standalone
:class:`IdentityMf` PLD models a single sensitivity-1 Gaussian release
(``gaussian_pld(noise_multiplier)``).  Realistic FTRL training uses one of
the FTRL amplifications on top:

- :func:`opaque.dpftrl.accounting.cyclic_poisson` for per-step Poisson
  inclusion (``T`` independent rounds).
- :func:`opaque.dpftrl.accounting.balls_in_bins` for fixed-partition
  multi-epoch sampling — tight identity-aware reduction inside
  :class:`BallsInBins`.

Unsubsampled training is the existing :class:`opaque.accounting.composition.types.Repeated`
form: ``mf_identity(nm) * num_steps`` composes plain Gaussian.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.accounting import _native

from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.discretization import get_discretization


@dataclass(frozen=True, slots=True)
class IdentityMf(DpProcess):
    """MF identity (uncorrelated noise) mechanism — a single Gaussian release.

    Sensitivity is fixed at ``1.0`` because the encoder is :math:`I`.  Stand-
    alone composition (``IdentityMf(...) * T``) gives unsubsampled
    Gaussian-over-T-rounds accounting; wrap with
    :func:`opaque.dpftrl.accounting.cyclic_poisson` /
    :func:`opaque.dpftrl.accounting.balls_in_bins` for FTRL-native subsampled
    or fixed-partition analyses.
    """

    noise_multiplier: float

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
        native_cfg = config.to_native()
        if self.noise_multiplier == 0:
            return _native.non_private_pld(native_cfg)
        return _native.gaussian_pld(float(self.noise_multiplier), native_cfg)


def mf_identity(noise_multiplier: float) -> IdentityMf:
    """MF identity mechanism factory (sensitivity ``1.0``).

    Pairs with :func:`opaque.dpftrl.noise.identity_strategy`.  Use as the inner
    of an FTRL amplification factory for the subsampled / fixed-partition
    accounting story you want:

    - Per-step Poisson over ``T`` rounds::

        ftrl_acc.cyclic_poisson(
            ftrl_acc.mf_identity(nm),
            sample_rate=p,
            num_steps=T,
        )

    - Fixed-partition (Balls-in-Bins) over ``E`` epochs::

        ftrl_acc.balls_in_bins(
            ftrl_acc.mf_identity(nm),
            num_bins=k,
            num_epochs=E,
        )

    - Unamplified composition::

        ftrl_acc.mf_identity(nm) * T

    Args:
        noise_multiplier: Gaussian noise multiplier ``σ / Δ`` with sensitivity
            ``Δ = 1`` (identity encoder).  ``0`` is non-private (``ε = ∞``).

    Returns:
        An :class:`IdentityMf` mechanism dataclass.
    """
    return IdentityMf(noise_multiplier=float(noise_multiplier))


__all__ = ["IdentityMf", "mf_identity"]
