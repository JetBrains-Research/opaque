"""Generic noise-mechanism base type and the ``NoisedPytree`` wrapper.

The ``NoiseState`` abstract base, the ``NoisedPytree`` post-noise wrapper,
and shared math helpers for joint first+second moment release live here.
Concrete DP-SGD mechanisms (Gaussian, truncated Gaussian, per-group) live in
:mod:`opaque.dpsgd.noise`; DP-FTRL matrix-factorization mechanisms in
:mod:`opaque.dpftrl.noise`.
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

from opaque.clipping.types import (
    ClippedPytree,
    MaxNorm,
    _scale_max_norm,
    _scale_tensor_leaves,
)
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


# ---------------------------------------------------------------------------
# NoisedPytree — the post-mechanism counterpart of ClippedPytree
# ---------------------------------------------------------------------------


NoiseStddev = Any


def _scale_stddev(stddev: NoiseStddev, factor: float) -> NoiseStddev:
    if stddev is None:
        return None
    return abs(factor) * stddev


@dataclass(frozen=True)
class NoisedPytree(ClippedPytree):
    """A privatised pytree carrying max-norm and realised noise metadata.

    Extends :class:`opaque.clipping.types.ClippedPytree` with the per-step
    ``noise_stddev`` recorded by the noise mechanism.  ``max_norm`` still
    describes the original record-impact max_norm, not a max_norm on the noised
    output values.
    """

    noise_stddev: NoiseStddev = None

    def _scaled(self, scalar: float) -> NoisedPytree:
        return replace(
            self,
            pytree=_scale_tensor_leaves(self.pytree, scalar),
            max_norm=_scale_max_norm(self.max_norm, scalar),
            noise_stddev=_scale_stddev(self.noise_stddev, scalar),
        )


def noised(
    pytree: Any,
    *,
    max_norm: MaxNorm,
    noise_stddev: NoiseStddev,
) -> NoisedPytree:
    """Manually wrap an already-privatised pytree with noise metadata."""
    return NoisedPytree(
        pytree=pytree, max_norm=max_norm, noise_stddev=noise_stddev
    )


# ---------------------------------------------------------------------------
# SecondMomentNoiseOutput — paired post-noise streams
# ---------------------------------------------------------------------------


class SecondMomentClippingOutput(NamedTuple):
    """Pre-noise paired-stream input to a noise mechanism.

    Symmetric with :class:`SecondMomentNoiseOutput` but on the
    *pre-noise* side: where the output pairs two ``NoisedPytree``s
    (post-noise), this pairs two ``ClippedPytree``s (pre-noise).
    Each carries its own ``max_norm``.

    Constructed by clipping when the user requests paired-stream
    output (per-example squaring inside the vmap loop).  The presence
    of this type at a noise mechanism's input switches the mechanism
    into paired-stream mode without an explicit constructor flag.

    Attributes:
        grads: Clipped per-example summed gradients (Σᵢ gᵢ).
        squared_grads: Clipped per-example summed squared gradients
            (Σᵢ gᵢ²) — the per-example squaring is what makes this
            useful as Adam's second moment under DP, distinct from
            squaring the noised summed gradient.
    """

    grads: ClippedPytree
    squared_grads: ClippedPytree


class SecondMomentNoiseOutput(NamedTuple):
    """Noise output with private first and second moment streams.

    Attributes:
        noisy_grads: Noised clipped gradients for the optimizer's first
            moment / update direction.
        noisy_squared_grads: Noised element-wise squared clipped gradients
            for optimizers with a second moment accumulator.
    """

    noisy_grads: NoisedPytree
    noisy_squared_grads: NoisedPytree


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
    first_max_norm: float,
    squared_max_norm: float,
    *,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> float:
    """Scale from first-stream stddev to second-stream stddev.

    For overhead ``rho`` and per-record sensitivities ``Δ_first`` /
    ``Δ_second`` on the two streams, the joint Gaussian allocation gives::

        σ_second = σ_first · (c2 · Δ_second) / (c1 · Δ_first · sqrt(rho² − 1))

    so the scale returned here is the right-hand factor.  Pass
    ``squared_max_norm`` from the second-stream's contribution bound
    directly — for per-example correct DP-Adam it is ``C² / n`` (not
    ``(C/n)²``); the two differ by a factor of ``n``, and the
    distinction matters for getting the noise ratio right.
    """
    if c1_max_column_norm <= 0:
        raise ValueError(
            f"c1_max_column_norm must be positive, got {c1_max_column_norm}"
        )
    if c2_max_column_norm <= 0:
        raise ValueError(
            f"c2_max_column_norm must be positive, got {c2_max_column_norm}"
        )
    if first_max_norm <= 0:
        raise ValueError(f"first_max_norm must be positive, got {first_max_norm}")
    if squared_max_norm <= 0:
        raise ValueError(
            f"squared_max_norm must be positive, got {squared_max_norm}"
        )
    if first_moment_overhead <= 1.0:
        raise ValueError(
            "first_moment_overhead must be greater than 1.0, "
            f"got {first_moment_overhead}"
        )
    return (
        squared_max_norm
        * c2_max_column_norm
        / (
            first_max_norm
            * c1_max_column_norm
            * math.sqrt(first_moment_overhead**2 - 1.0)
        )
    )


def second_moment_stddevs(
    noise_multiplier: float,
    first_max_norm: float,
    squared_max_norm: float,
    *,
    c1_max_column_norm: float = 1.0,
    c2_max_column_norm: float = 1.0,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> tuple[float, float]:
    """Return ``(first_stddev, second_stddev)`` for private moments.

    ``first_max_norm`` is the per-record contribution bound on the
    first stream (``C / n`` for averaged clipped grads).
    ``squared_max_norm`` is the analogous bound on the second stream —
    for per-example correct ``Σᵢ gᵢ²`` it is ``C² / n``, *not*
    ``first_max_norm²``.  Confusing the two under-samples the
    second-stream noise by a factor of ``n``.
    """
    if noise_multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    first_sensitivity = second_moment_joint_sensitivity(
        c1_max_column_norm,
        first_max_norm,
        first_moment_overhead=first_moment_overhead,
    )
    first_stddev = noise_multiplier * first_sensitivity
    second_stddev = first_stddev * second_moment_noise_scale(
        c1_max_column_norm,
        c2_max_column_norm,
        first_max_norm,
        squared_max_norm,
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
    "NoiseStddev",
    "NoisedPytree",
    "SecondMomentClippingOutput",
    "SecondMomentNoiseOutput",
    "assert_rng_key_equal",
    "noised",
    "resolve_second_moment_overhead",
    "second_moment_joint_sensitivity",
    "second_moment_noise_scale",
    "second_moment_stddevs",
]
