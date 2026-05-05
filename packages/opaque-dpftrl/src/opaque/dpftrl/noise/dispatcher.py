"""MF strategy types and unified noise dispatcher.

Each mechanism file defines its own strategy dataclass and factory:
- ``identity.py``: :class:`IdentityStrategy`, :func:`identity_strategy`
- ``band_mf.py``: :class:`BandMfStrategy`, :func:`band_mf_strategy`
- ``blt.py``: :class:`BltStrategy`, :func:`blt_strategy`
- ``lambda_cgd.py``: :class:`LambdaCgdStrategy`, :func:`lambda_cgd_strategy`
- ``bisr.py``: :class:`BisrStrategy`, :func:`bisr_strategy`
- ``bsr.py``: :class:`BsrStrategy`, :func:`bsr_strategy`

The :func:`mf_noise` function dispatches on the strategy type to create
the appropriate noise mechanism.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from opaque.types import PerGroup
from opaque.types import ClippedPytree
from opaque.core.noise import DEFAULT_SECOND_MOMENT_OVERHEAD, second_moment_joint_sensitivity, second_moment_noise_scale, second_moment_stddevs
from opaque.types import NoisedPytree, SecondMomentClippingOutput, SecondMomentNoiseOutput
from opaque.random import RngKey
from opaque.random import fold_in as rng_fold_in

from .band_mf import BandMfStrategy, band_mf_strategy
from .bisr import BisrStrategy, bisr_strategy
from .bsr import BsrStrategy, bsr_strategy
from .blt import BltStrategy, blt_strategy
from .identity import IdentityStrategy, identity_strategy
from .lambda_cgd import (
    LambdaCgdStrategy,
    _make_lambda_cgd_noise,
    lambda_cgd_strategy,
)
from ._engine import (
    MFNoiseState,
    _matrix_factorization_noise,
)
from ._streaming_matrix import (
    identity,
)
from .second_moment import SecondMomentMFNoiseState

MfStrategy = (
    BandMfStrategy
    | BltStrategy
    | LambdaCgdStrategy
    | BisrStrategy
    | BsrStrategy
    | IdentityStrategy
)


def mf_noise(
    grad_template: Any,
    strategy: MfStrategy,
    *,
    noise_multiplier: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
    second_moment_strategy: MfStrategy | None = None,
    first_moment_overhead: float = DEFAULT_SECOND_MOMENT_OVERHEAD,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState | SecondMomentMFNoiseState]],
    MFNoiseState | SecondMomentMFNoiseState,
]:
    """Create a correlated noise mechanism for the given MF strategy.

    Dispatches on the strategy type:
    - :class:`LambdaCgdStrategy`: PRNG replay noise (zero extra memory).
    - :class:`BandMfStrategy`, :class:`BltStrategy`, :class:`BisrStrategy`,
      :class:`BsrStrategy`:
      StreamingMatrix-based noise.

    The returned ``noise_fn`` polymorphically dispatches on its input
    type:

    - ``ClippedPytree`` → ``NoisedPytree`` (single-stream noise).
    - ``SecondMomentClippingOutput`` → ``SecondMomentNoiseOutput``
      (paired-stream noise; only available when
      ``second_moment_strategy`` was supplied at construction).

    Args:
        grad_template: Pytree with same structure/shapes as gradients.
        strategy: MF strategy from one of the factory functions.
        noise_multiplier: Gaussian noise multiplier. The clipped-gradient
            ``max_norm`` is read from each ``ClippedPytree`` input.
        key: Explicit RNG key for deterministic randomness.
        dtype: Optional dtype for intermediate noise computation.
        second_moment_strategy: Optional explicit strategy for the
            squared-gradient stream.  When supplied the mechanism allocates
            a second noise stream and only accepts
            ``SecondMomentClippingOutput`` inputs at call time.  When
            ``None`` (default) only single-stream ``ClippedPytree`` inputs
            are accepted.
        first_moment_overhead: First-stream noise overhead used when
            paired-stream output is requested.  Defaults to ``√(3/2)``
            (the d ≥ 2 add/remove DP value).  Ignored when
            ``second_moment_strategy`` is ``None``.

    Returns:
        A tuple ``(noise_fn, state)`` for the training loop.
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)
    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    if second_moment_strategy is not None:
        return _make_second_moment_mf_noise(
            grad_template,
            strategy,
            second_moment_strategy,
            noise_multiplier=resolved_noise_multiplier,
            key=key,
            dtype=dtype,
            first_moment_overhead=first_moment_overhead,
        )

    raw_noise_fn, raw_state = _make_raw_mf_noise(
        grad_template,
        strategy,
        key=key,
        dtype=dtype,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
    ) -> tuple[NoisedPytree, MFNoiseState]:
        if isinstance(clipped_grads, SecondMomentClippingOutput):
            raise TypeError(
                "mf_noise was constructed without `second_moment_strategy` and "
                "cannot consume SecondMomentClippingOutput inputs.  Either pass "
                "a single-stream ClippedPytree, or rebuild the noise function "
                "with `second_moment_strategy=...`."
            )
        clipped_grads = _expect_clipped(clipped_grads, op="mf_noise")
        max_norm = _validate_constant_max_norm(
            clipped_grads, st._first_max_norm, op="mf_noise"
        )
        base_stddev = resolved_noise_multiplier * max_norm
        noisy_tree, new_state = raw_noise_fn(
            clipped_grads.pytree,
            st,
            stddev=base_stddev,
        )
        return (
            NoisedPytree(
                pytree=noisy_tree,
                max_norm=clipped_grads.max_norm,
                noise_stddev=base_stddev,
            ),
            replace(new_state, _first_max_norm=max_norm),
        )

    return noise_fn, raw_state


def _make_raw_mf_noise(
    grad_template: Any,
    strategy: MfStrategy,
    *,
    key: RngKey,
    dtype: torch.dtype | None,
) -> tuple[Callable[..., tuple[Any, MFNoiseState]], MFNoiseState]:
    match strategy:
        case IdentityStrategy():
            return _matrix_factorization_noise(
                grad_template,
                identity(),
                key=key,
                dtype=dtype,
            )
        case LambdaCgdStrategy():
            return _make_lambda_cgd_noise(
                grad_template,
                strategy,
                key=key,
                dtype=dtype,
            )
        case BandMfStrategy() | BltStrategy() | BisrStrategy() | BsrStrategy():
            if strategy._streaming_matrix is None:
                raise ValueError(
                    "Strategy must have a _streaming_matrix for noise generation."
                )
            return _matrix_factorization_noise(
                grad_template,
                strategy._streaming_matrix,
                key=key,
                dtype=dtype,
            )
        case _:
            raise TypeError(f"Unknown strategy type: {type(strategy).__name__}")


def _resolve_noise_multiplier(noise_multiplier: float) -> float:
    multiplier = float(noise_multiplier)
    if multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    return multiplier


def _expect_clipped(value: Any, *, op: str) -> ClippedPytree:
    if isinstance(value, NoisedPytree):
        raise TypeError(
            f"{op} expects ClippedPytree inputs, not NoisedPytree values that "
            "have already passed through a noise mechanism."
        )
    if not isinstance(value, ClippedPytree):
        raise TypeError(
            f"{op} expects ClippedPytree inputs. Wrap manual values with "
            "opaque.clipping.types.clipped(...)."
        )
    if isinstance(value.max_norm, PerGroup):
        raise TypeError(
            f"{op} does not support PerGroup bounds. Per-group noise allocation "
            "for matrix-factorization mechanisms has not been validated; the "
            "interaction of per-group clipping with correlated noise across "
            "time is an open question. Use a scalar clipping_norm with "
            "MF mechanisms."
        )
    return value


def _validate_constant_max_norm(
    grads: ClippedPytree,
    first_max_norm: float | None,
    *,
    op: str,
) -> float:
    """Latch the per-step max_norm and reject changes across calls.

    MF privacy analyses calibrate noise from a sensitivity that is constant
    across the sequence; varying ``ClippedPytree.max_norm`` per call (e.g. from
    adaptive clipping) breaks the proof.  The dispatcher latches the
    first-call max_norm in the state and rejects any subsequent call whose
    max_norm differs.
    """
    sensitivity = grads.sensitivity
    if sensitivity < 0:
        raise ValueError(
            f"ClippedPytree max_norm must be non-negative, got {grads.max_norm}"
        )
    max_norm = float(sensitivity)
    if first_max_norm is not None and max_norm != first_max_norm:
        raise ValueError(
            f"{op} saw a varying ClippedPytree.max_norm across calls "
            f"(first={first_max_norm}, now={max_norm}). MF privacy proofs assume a "
            "constant per-step sensitivity; adaptive clipping with MF noise "
            "is not supported."
        )
    return max_norm


def _make_second_moment_mf_noise(
    grad_template: Any,
    first_strategy: MfStrategy,
    second_strategy: MfStrategy,
    *,
    noise_multiplier: float,
    key: RngKey,
    dtype: torch.dtype | None,
    first_moment_overhead: float,
) -> tuple[
    Callable[
        [Any, SecondMomentMFNoiseState],
        tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState],
    ],
    SecondMomentMFNoiseState,
]:
    if first_moment_overhead <= 1.0:
        raise ValueError(
            "first_moment_overhead must be greater than 1.0, "
            f"got {first_moment_overhead}"
        )

    first_fn, first_state = _make_raw_mf_noise(
        grad_template,
        first_strategy,
        key=rng_fold_in(key, 0),
        dtype=dtype,
    )
    second_fn, second_state = _make_raw_mf_noise(
        grad_template,
        second_strategy,
        key=rng_fold_in(key, 1),
        dtype=dtype,
    )

    init_state = SecondMomentMFNoiseState(
        _first_state=first_state,
        _second_state=second_state,
    )

    def noise_fn(
        clipped_input: Any,
        st: SecondMomentMFNoiseState,
    ) -> tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState]:
        if not isinstance(clipped_input, SecondMomentClippingOutput):
            raise TypeError(
                "mf_noise was constructed with `second_moment_strategy` and "
                "expects SecondMomentClippingOutput inputs (paired-stream).  "
                "Build the paired form upstream via "
                "`clipped_grad(..., second_moment=True)`, or rebuild the "
                "noise function without `second_moment_strategy` for "
                "single-stream mode."
            )
        first_clipped = _expect_clipped(clipped_input.grads, op="mf_noise")
        second_clipped = _expect_clipped(
            clipped_input.squared_grads, op="mf_noise (squared stream)"
        )
        max_norm = _validate_constant_max_norm(
            first_clipped, st._first_state._first_max_norm, op="mf_noise"
        )
        squared_max_norm = _validate_constant_max_norm(
            second_clipped,
            st._second_state._first_max_norm,
            op="mf_noise (squared stream)",
        )
        first_stddev, second_stddev = second_moment_stddevs(
            noise_multiplier,
            first_max_norm=max_norm,
            squared_max_norm=squared_max_norm,
            c1_max_column_norm=first_strategy._max_column_norm,
            c2_max_column_norm=second_strategy._max_column_norm,
            first_moment_overhead=first_moment_overhead,
        )
        noisy_grads, new_first = first_fn(
            first_clipped.pytree,
            st._first_state,
            stddev=first_stddev,
        )
        noisy_squared, new_second = second_fn(
            second_clipped.pytree,
            st._second_state,
            stddev=second_stddev,
        )
        return (
            SecondMomentNoiseOutput(
                NoisedPytree(
                    pytree=noisy_grads,
                    max_norm=first_clipped.max_norm,
                    noise_stddev=first_stddev,
                ),
                NoisedPytree(
                    pytree=noisy_squared,
                    max_norm=second_clipped.max_norm,
                    noise_stddev=second_stddev,
                ),
            ),
            SecondMomentMFNoiseState(
                _first_state=replace(new_first, _first_max_norm=max_norm),
                _second_state=replace(new_second, _first_max_norm=squared_max_norm),
            ),
        )

    return noise_fn, init_state


__all__ = [
    "BandMfStrategy",
    "BltStrategy",
    "IdentityStrategy",
    "LambdaCgdStrategy",
    "BisrStrategy",
    "BsrStrategy",
    "DEFAULT_SECOND_MOMENT_OVERHEAD",
    "MfStrategy",
    "SecondMomentMFNoiseState",
    "SecondMomentNoiseOutput",
    "band_mf_strategy",
    "blt_strategy",
    "identity_strategy",
    "lambda_cgd_strategy",
    "bisr_strategy",
    "bsr_strategy",
    "second_moment_joint_sensitivity",
    "second_moment_noise_scale",
    "second_moment_stddevs",
    "mf_noise",
]
