"""Random-allocation amplification from a base PLD.

This wraps the native :func:`opaque_accounting.random_allocation_pld` helper
(Feldman & Shenfeld, 2026, arXiv:2602.17284 — conservative deterministic path).

Example::

    import opaque_accounting as acc

    step = acc.gaussian(1.0)
    pld_step = step.pld()
    training = acc.random_allocation_pld(pld_step, t=1000, k=1, target_delta=1e-5)
    eps = training.epsilon_at(1e-5)
"""

from __future__ import annotations

from .. import opaque_accounting as _native

from opaque_accounting.base import Pld
from opaque_accounting.discretization import get_discretization


def random_allocation_pld(
    base_pld: Pld,
    t: int,
    k: int = 1,
    *,
    target_delta: float = 1e-5,
    discretization: float | None = None,
    log_x_mass_truncation_bound: float | None = None,
    pessimistic_estimate: bool | None = None,
    max_grid_size: int | None = None,
) -> Pld:
    """Amplify ``base_pld`` under ``k``-out-of-``t`` random allocation.

    Args:
        base_pld: Privacy loss distribution for one full run of the base
            mechanism (e.g. one DP-SGD epoch without amplification).
        t: Total number of steps in the allocation sequence.
        k: Number of times each record participates (default 1, balls-in-bins).
        target_delta: Delta used internally when lifting the base PLD to
            an :math:`(\\varepsilon,\\delta)` surrogate before amplification.
        discretization: Optional PLD grid override (same as other ``pld()`` calls).
        log_x_mass_truncation_bound: Optional tail truncation override.
        pessimistic_estimate: Whether to use pessimistic grid rounding.
        max_grid_size: Optional max PLD grid size.

    Returns:
        A :class:`Pld` for the amplified process.
    """
    cfg = get_discretization(
        discretization=discretization,
        log_x_mass_truncation_bound=log_x_mass_truncation_bound,
        pessimistic_estimate=pessimistic_estimate,
        max_grid_size=max_grid_size,
    )
    return _native.random_allocation_pld(
        base_pld, t, k, target_delta, cfg.to_native()
    )
