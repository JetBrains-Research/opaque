"""Bounded Gaussian mechanism — Add/Remove adjacency DP accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import DiscretizationConfig, DpProcess, Pld
from opaque.accounting.discretization import resolve_pld_config


@dataclass(frozen=True, slots=True)
class BoundedGaussian(DpProcess):
    """Bounded Gaussian mechanism (Add/Remove adjacency, wide-bound approximation).

    The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds truncated Gaussian
    noise to keep outputs within a bounded domain.  For DP-SGD the mechanism is
    analysed under **Add/Remove** adjacency with sensitivity 1 — the same as the
    standard Gaussian mechanism.

    When the truncation bounds are wide relative to σ (the standard DP-SGD case),
    the PLD is approximately equal to the standard Gaussian PLD.  The ``pld()``
    method returns ``gaussian_pld(noise_multiplier)`` as a conservative
    (upper-bound) approximation.
    """

    noise_multiplier: float
    config: DiscretizationConfig | None = field(default=None, repr=False)

    def pld(self) -> Pld:
        return _native.bounded_gaussian_pld(self.noise_multiplier, config=self.config)


def bounded_gaussian(
    noise_multiplier: float,
    *,
    discretization: None | float | DiscretizationConfig = None,
) -> BoundedGaussian:
    """Bounded Gaussian mechanism (Add/Remove adjacency, wide-bound approximation).

    The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds noise from a
    truncated normal distribution, bounding outputs to a fixed domain.

    For DP-SGD with gradient clipping the standard adjacency is **Add/Remove**
    with sensitivity 1.  When the truncation bounds are wide relative to σ (at
    least ~3σ from every possible query value), the PLD is approximately equal
    to the standard Gaussian PLD.  This function returns an accounting process
    that uses ``gaussian_pld(noise_multiplier)`` as a conservative upper bound
    on ε.

    Note:
        The exact PLD of a truncated Gaussian includes a log-normalisation
        correction term that depends on the truncation bounds and the query value.
        For narrow bounds or query values near the boundaries the approximation
        degrades.  Future API versions will accept the truncation bounds
        explicitly for exact accounting.

    Args:
        noise_multiplier: Noise standard deviation divided by sensitivity (σ/Δ).
            Valid range: [0.1, 1.2] — same as :func:`gaussian`.
        discretization: PLD precision config (keyword-only). Can be:
            - None: use module default (see :func:`set_discretization`)
            - float: use as grid spacing
            - DiscretizationConfig: full config

    Returns:
        A :class:`BoundedGaussian` process.

    Example::

        # Single bounded Gaussian query
        proc = acc.bounded_gaussian(1.1)
        eps = proc.epsilon_at(1e-5)

        # Composed 1000 times
        training = acc.bounded_gaussian(1.1) * 1000
        eps = training.epsilon_at(1e-5)

    References:
        Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
        Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
        https://arxiv.org/abs/2211.17230
    """
    config = resolve_pld_config(discretization)
    return BoundedGaussian(noise_multiplier, config=config)
