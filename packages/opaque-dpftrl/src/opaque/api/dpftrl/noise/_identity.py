"""Identity noise strategy — independent Gaussian noise (DP-SGD via MF API).

Use ``mf_noise(template, identity_mf_strategy(), ...)`` for standard DP-SGD
with independent noise at each step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityMfStrategy:
    """Identity (DP-SGD) strategy — independent noise at each step.

    The identity matrix has column norms of 1, so ``sensitivity = 1.0``.
    """

    sensitivity: float = 1.0
    _max_column_norm: float = 1.0


def identity_mf_strategy() -> IdentityMfStrategy:
    """Create an identity (DP-SGD) noise strategy.

    Returns:
        An :class:`IdentityMfStrategy` for use with :func:`mf_noise`.

    Example:
        >>> from opaque.types import clipped
        >>> from opaque.api.dpftrl.noise import mf_noise, identity_mf_strategy
        >>> from opaque.random import key
        >>> noise_fn, state = mf_noise(
        ...     template, identity_mf_strategy(), noise_multiplier=1.0, key=key(42)
        ... )
        >>> noised, state = noise_fn(clipped(grads, max_norm=1.0), state)
    """
    return IdentityMfStrategy()


__all__ = ["IdentityMfStrategy", "identity_mf_strategy"]
