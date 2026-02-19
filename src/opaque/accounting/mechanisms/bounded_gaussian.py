"""Bounded Gaussian mechanism — Replace-adjacency DP accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import DiscretizationConfig, DpProcess, Pld
from opaque.accounting.discretization import resolve_pld_config


@dataclass(frozen=True, slots=True)
class BoundedGaussian(DpProcess):
    """Bounded Gaussian mechanism (Replace adjacency).

    The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds truncated Gaussian
    noise to keep outputs within a bounded domain.  Under **Replace** adjacency,
    sensitivity is 2Δ, so the PLD is equivalent to a standard Gaussian with
    ``effective_σ = noise_multiplier / 2``.
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
    """Bounded Gaussian mechanism (Replace adjacency).

    The Bounded Gaussian Mechanism (Chen & Hale, 2024) adds noise from a
    truncated normal distribution, bounding outputs to a fixed domain.
    Under **Replace** adjacency (one record swapped), sensitivity is 2Δ,
    which is accounted for internally.

    The ``noise_multiplier`` parameter has the same meaning and valid range
    as for :func:`gaussian`: σ/Δ with Δ = 1 (unit sensitivity).  For the same
    ``noise_multiplier``, this mechanism has higher ε than :func:`gaussian`
    because Replace adjacency doubles the effective sensitivity.

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
