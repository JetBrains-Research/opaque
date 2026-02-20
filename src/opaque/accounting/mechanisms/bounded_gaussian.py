"""Bounded Gaussian mechanism — Add/Remove adjacency DP accounting."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import opaque_accounting as _native

from opaque.accounting.base import DpProcess, Pld
from opaque.accounting.discretization import get_discretization


@dataclass(frozen=True, slots=True)
class BoundedGaussian(DpProcess):
    """Bounded Gaussian mechanism (Add/Remove adjacency, wide-bound approximation).

    The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds truncated Gaussian
    noise to keep outputs within a bounded domain.  For DP-SGD the mechanism is
    analysed under **Add/Remove** adjacency with sensitivity 1 — the same as the
    standard Gaussian mechanism.

    When the truncation bounds are wide relative to σ (the standard DP-SGD case),
    the PLD is approximately equal to the standard Gaussian PLD.
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
        return _native.bounded_gaussian_pld(self.noise_multiplier, config.to_native())


def bounded_gaussian(noise_multiplier: float) -> BoundedGaussian:
    """Bounded Gaussian mechanism (Add/Remove adjacency, wide-bound approximation).

    The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds noise from a
    truncated normal distribution, bounding outputs to a fixed domain.

    For DP-SGD with gradient clipping the standard adjacency is **Add/Remove**
    with sensitivity 1.  When the truncation bounds are wide relative to σ (at
    least ~3σ from every possible query value), the PLD is approximately equal
    to the standard Gaussian PLD, so ``bounded_gaussian_pld(nm)`` is used as a
    conservative upper bound on ε.

    Note:
        The exact PLD of a truncated Gaussian includes a log-normalisation
        correction term that depends on the truncation bounds and the query value.
        For narrow bounds or query values near the boundaries the approximation
        degrades.  Future API versions will accept the truncation bounds
        explicitly for exact accounting.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            Valid range: [0.1, 1.2] — same as :func:`gaussian`.

    Returns:
        A :class:`BoundedGaussian` process.

    Example::

        # Single bounded Gaussian query
        proc = acc.bounded_gaussian(1.1)
        eps = proc.epsilon_at(1e-5)

        # Composed 1000 times
        training = acc.bounded_gaussian(1.1) * 1000
        eps = training.epsilon_at(1e-5)

        # Query-time discretization override
        eps = proc.epsilon_at(1e-5, discretization=1e-3)

    References:
        Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
        Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
        https://arxiv.org/abs/2211.17230
    """
    return BoundedGaussian(noise_multiplier)
