"""Identity noise strategy — independent Gaussian noise (DP-SGD via MF API).

Use ``mf_noise(template, identity_strategy(), ...)`` for standard DP-SGD
with independent noise at each step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityStrategy:
    """Identity (DP-SGD) strategy — independent noise at each step.

    The identity matrix has column norms of 1, so ``sensitivity = 1.0``.
    """

    sensitivity: float = 1.0
    _max_column_norm: float = 1.0


def identity_strategy() -> IdentityStrategy:
    """Create an identity (DP-SGD) noise strategy.

    Returns:
        An :class:`IdentityStrategy` for use with :func:`mf_noise`.

    Example:
        >>> from opaque.mf.noise import mf_noise, identity_strategy
        >>> from opaque.core.random import key
        >>> noise_fn, state = mf_noise(template, identity_strategy(), stddev=1.0, key=key(42))
    """
    return IdentityStrategy()


__all__ = ["IdentityStrategy", "identity_strategy"]
