"""Identity noise strategy — independent Gaussian noise (DP-SGD via MF API).

Use ``mf_noise(identity_strategy(), ...)`` or the convenience wrapper
``identity_mf_noise(...)`` for standard DP-SGD with independent noise
at each step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opaque.noise.custom_mf_noise import custom_mf_noise
from opaque.noise.matrix_factorization.noise import MFNoiseState
from opaque.noise.matrix_factorization.streaming_matrix import identity
from opaque.random import RngKey


@dataclass(frozen=True)
class IdentityStrategy:
    """Identity (DP-SGD) strategy — independent noise at each step.

    The identity matrix has column norms of 1, so ``sensitivity = 1.0``.
    """

    sensitivity: float = 1.0


def identity_strategy() -> IdentityStrategy:
    """Create an identity (DP-SGD) noise strategy.

    Returns:
        An :class:`IdentityStrategy` for use with :func:`mf_noise`.

    Example:
        >>> from opaque.noise import mf_noise, identity_strategy
        >>> from opaque.random import key
        >>> noise_fn, state = mf_noise(template, identity_strategy(), stddev=1.0, key=key(42))
    """
    return IdentityStrategy()


def identity_mf_noise(
    grad_template: Any,
    *,
    stddev: float,
    key: RngKey,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create an identity (DP-SGD) noise mechanism via the MF API.

    Convenience wrapper equivalent to
    ``mf_noise(grad_template, identity_strategy(), stddev=stddev, key=key)``.

    Args:
        grad_template: A pytree with the same structure and shapes as the
            gradients that will be passed to ``noise_fn``.
        stddev: Standard deviation for the base noise.
        key: Explicit RNG key for deterministic, functional randomness.

    Returns:
        A tuple ``(noise_fn, state)`` for the training loop.
    """
    return custom_mf_noise(
        grad_template,
        identity(),
        stddev=stddev,
        key=key,
    )


__all__ = ["IdentityStrategy", "identity_strategy", "identity_mf_noise"]
