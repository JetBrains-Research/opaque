"""Identity MF strategy — uncorrelated noise baseline (DP-SGD via MF API).

Use ``mf_noise(template, identity_strategy(), ...)`` for standard DP-SGD
with independent noise at each step.
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.accounting.core._process_codec import register_strategy


@register_strategy
@dataclass(frozen=True)
class IdentityStrategy:
    """Identity (uncorrelated) MF strategy — independent noise per step.

    Encoder ``C = I`` so every column has unit L2 norm: ``sensitivity = 1.0``.
    Horizon-invariant: :meth:`with_horizon` returns ``self``.
    """

    sensitivity: float = 1.0
    _max_column_norm: float = 1.0

    def with_horizon(
        self, n_steps: int, max_participations: int | None
    ) -> "IdentityStrategy":
        return self


def identity_strategy() -> IdentityStrategy:
    """Create an identity (DP-SGD) noise strategy.

    Returns:
        An :class:`IdentityStrategy` for use with :func:`mf_noise` and
        :func:`opaque.dpftrl.accounting.mf_gaussian`.
    """
    return IdentityStrategy()


__all__ = ["IdentityStrategy", "identity_strategy"]
