"""Generic noise-mechanism base type.

Only the ``NoiseState`` abstract base lives here, plus a few helpers
that concrete noise-state sync implementations share. Concrete DP-SGD
mechanisms (Gaussian, truncated Gaussian, per-group) live in
:mod:`opaque.dpsgd.noise`; DP-FTRL matrix-factorization mechanisms in
:mod:`opaque.dpftrl.noise`.
"""

from __future__ import annotations

import math
from abc import ABC
from typing import Any, NamedTuple

from opaque.random import RngKey


DEFAULT_SECOND_MOMENT_OVERHEAD = math.sqrt(3.0 / 2.0)
"""Default first-stream noise overhead for private second moments.

This is the add/remove-DP d >= 2 value from the joint first+second
moment analysis.  It multiplies the first-moment sensitivity when the
noise mechanism also releases a private element-wise squared-gradient
stream.
"""


class NoiseState(ABC):
    """Base class for noise state.

    All noise functions (Gaussian and matrix factorization) return a state
    object that inherits from this class, providing a unified interface for
    step tracking and RNG key management.

    Attributes:
        _step_counter: Number of noise_fn calls made.
        _rng_key: Immutable RNG key for deterministic per-step derivation.
    """

    _step_counter: int
    """Number of noise_fn calls made."""

    _rng_key: RngKey
    """Immutable RNG key for deterministic per-step derivation."""


class SecondMomentNoiseOutput(NamedTuple):
    """Noise output with private first and second moment streams.

    Attributes:
        noisy_grads: Noisy clipped gradients for the optimizer's first
            moment / update direction.
        noisy_squared_grads: Noisy element-wise squared clipped gradients
            for optimizers with a second moment accumulator.
    """

    noisy_grads: Any
    noisy_squared_grads: Any


def resolve_second_moment_overhead(second_moment: bool | float) -> float:
    """Resolve ``second_moment`` to a first-moment noise overhead.

    ``True`` selects the default paper value ``sqrt(3/2)`` for d >= 2.
    A float supplies the overhead directly and must be greater than 1.
    """
    if isinstance(second_moment, bool):
        if not second_moment:
            raise ValueError("second_moment=False has no overhead to resolve")
        return DEFAULT_SECOND_MOMENT_OVERHEAD

    overhead = float(second_moment)
    if overhead <= 1.0:
        raise ValueError(
            f"second_moment overhead must be greater than 1.0, got {second_moment}"
        )
    return overhead


def second_moment_joint_sensitivity(
    c1_max_column_norm: float,
    sensitivity: float,
    *,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> float:
    """Sensitivity of the first stream when releasing a second moment.

    ``sensitivity`` is the clipped-gradient sensitivity before applying
    the mechanism strategy (for averaged gradients this is typically
    ``clipping_norm / batch_size``).  ``c1_max_column_norm`` is the
    first strategy's max column norm.
    """
    if c1_max_column_norm <= 0:
        raise ValueError(
            f"c1_max_column_norm must be positive, got {c1_max_column_norm}"
        )
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")
    if first_moment_overhead <= 1.0:
        raise ValueError(
            "first_moment_overhead must be greater than 1.0, "
            f"got {first_moment_overhead}"
        )
    return sensitivity * c1_max_column_norm * first_moment_overhead


def second_moment_noise_scale(
    c1_max_column_norm: float,
    c2_max_column_norm: float,
    sensitivity: float,
    *,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> float:
    """Scale from first-stream stddev to second-stream stddev.

    The public API is expressed in terms of the first-moment overhead
    rather than the paper's allocation variable.  For overhead ``rho``:

    ``second_stddev = first_stddev * sensitivity * c2 / (c1 * sqrt(rho^2 - 1))``.
    """
    if c1_max_column_norm <= 0:
        raise ValueError(
            f"c1_max_column_norm must be positive, got {c1_max_column_norm}"
        )
    if c2_max_column_norm <= 0:
        raise ValueError(
            f"c2_max_column_norm must be positive, got {c2_max_column_norm}"
        )
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")
    if first_moment_overhead <= 1.0:
        raise ValueError(
            "first_moment_overhead must be greater than 1.0, "
            f"got {first_moment_overhead}"
        )
    return (
        sensitivity
        * c2_max_column_norm
        / (c1_max_column_norm * math.sqrt(first_moment_overhead**2 - 1.0))
    )


def second_moment_stddevs(
    noise_multiplier: float,
    sensitivity: float,
    *,
    c1_max_column_norm: float = 1.0,
    c2_max_column_norm: float = 1.0,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> tuple[float, float]:
    """Return ``(first_stddev, second_stddev)`` for private moments."""
    if noise_multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    first_sensitivity = second_moment_joint_sensitivity(
        c1_max_column_norm,
        sensitivity,
        first_moment_overhead=first_moment_overhead,
    )
    first_stddev = noise_multiplier * first_sensitivity
    second_stddev = first_stddev * second_moment_noise_scale(
        c1_max_column_norm,
        c2_max_column_norm,
        sensitivity,
        first_moment_overhead=first_moment_overhead,
    )
    return first_stddev, second_stddev


# ---- Distributed sync helpers (shared across mechanisms) ----

# Field-level ops for ``opaque.distributed.sync_object`` applied to any
# ``NoiseState`` subclass.  All concrete noise mechanisms use the same
# step-counter convention, so this is centralized here.
NOISE_STATE_FIELD_OPS: dict[str, str] = {
    "_step_counter": "assert_equal",
}


def assert_rng_key_equal(state: NoiseState, state_name: str) -> None:
    """Assert that a ``NoiseState``'s RNG key seed matches across ranks.

    Shared across ``sync_gaussian_noise_state`` (opaque-dpsgd) and
    ``sync_mf_noise_state`` (opaque-dpftrl).
    """
    from opaque.distributed.state import assert_scalar_equal

    assert_scalar_equal(int(state._rng_key.seed), name=f"{state_name}.seed")


__all__ = [
    "DEFAULT_SECOND_MOMENT_OVERHEAD",
    "NOISE_STATE_FIELD_OPS",
    "NoiseState",
    "SecondMomentNoiseOutput",
    "assert_rng_key_equal",
    "resolve_second_moment_overhead",
    "second_moment_joint_sensitivity",
    "second_moment_noise_scale",
    "second_moment_stddevs",
]
