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
from opaque.api.engine.noise_allocation import paired_noise_stddevs, per_group_noise_stddev
from opaque.types import (
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)
from opaque.random.types import RngKey
from opaque.random import fold_in as rng_fold_in

from ._band_mf import BandMfStrategy, band_mf_strategy
from ._bisr import BisrStrategy, bisr_strategy
from ._bsr import BsrStrategy, bsr_strategy
from ._blt import BltStrategy, blt_strategy
from ._identity import IdentityStrategy, identity_strategy
from ._lambda_cgd import LambdaCgdStrategy, _make_lambda_cgd_noise, lambda_cgd_strategy
from ._distributed import (
    fingerprint_per_group_max_norm,
    fingerprint_scalar_max_norm,
)
from ._engine import MFNoiseState, _matrix_factorization_noise
from ._streaming_matrix import identity
from ._second_moment import SecondMomentMFNoiseState


def _mf_per_group_sync_fingerprint_for_latch(
    prior: MFNoiseState,
    max_norm: float | PerGroup,
) -> int | None:
    """Integer fingerprint for distributed sync; set on every latched ``max_norm``."""
    if prior._first_max_norm is None:
        if isinstance(max_norm, PerGroup):
            return fingerprint_per_group_max_norm(max_norm)
        return fingerprint_scalar_max_norm(float(max_norm))
    return prior._first_max_norm_sync_fingerprint


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

    The paired-stream release uses the sensitivity-proportional joint
    Mahalanobis allocation (``opaque.api.engine.noise_allocation.paired_noise_stddevs``),
    with the MF translation ``nm / ‖C₁‖`` as the joint effective multiplier so the
    joint PLD matches the single-stream MF Gaussian accountant at
    ``(noise_multiplier, ‖C₁‖)``.  ``PerGroup`` ``max_norm`` is supported on
    both streams (same joint allocation as DP-SGD Gaussian, then MF correlation).
    The joint privacy budget collapses to the same first-moment-only mechanism
    at the given ``noise_multiplier``, so calibration is identical to a
    first-moment-only release.

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
        if isinstance(max_norm, PerGroup):
            base_stddev = per_group_noise_stddev(max_norm, resolved_noise_multiplier)
        else:
            base_stddev = resolved_noise_multiplier * max_norm
        noisy_tree, new_state = raw_noise_fn(
            clipped_grads.pytree,
            st,
            stddev=base_stddev,
        )
        sync_fp = _mf_per_group_sync_fingerprint_for_latch(st, max_norm)
        return (
            NoisedPytree(
                pytree=noisy_tree,
                max_norm=clipped_grads.max_norm,
                noise_stddev=base_stddev,
            ),
            replace(
                new_state,
                _first_max_norm=max_norm,
                _first_max_norm_sync_fingerprint=sync_fp,
            ),
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
            "opaque.types.clipped(...)."
        )
    return value


def _validate_constant_max_norm(
    grads: ClippedPytree,
    first_max_norm: float | PerGroup | None,
    *,
    op: str,
) -> float | PerGroup:
    """Latch the per-step max_norm and reject changes across calls.

    MF privacy analyses calibrate noise from a sensitivity that is constant
    across the sequence.  Fixed clipping (:func:`opaque.dpftrl.clipping.clipped_grad`)
    and AUTO-S clipping (:func:`opaque.dpftrl.clipping.auto_clipped_grad`) both
    produce a constant ``ClippedPytree.max_norm`` and pass this latch.
    Adaptive clipping (:func:`opaque.dpsgd.clipping.adaptive_clipped_grad`)
    varies its threshold across steps, which breaks the proof; the
    dispatcher latches the first-call max_norm in the state and rejects any
    subsequent call whose max_norm differs.
    """
    max_norm = grads.max_norm
    if isinstance(max_norm, PerGroup):
        for group_name, value in max_norm.values.items():
            if value < 0:
                raise ValueError(
                    f"ClippedPytree max_norm must be non-negative for all groups, "
                    f"got {value} for group '{group_name}'."
                )
    else:
        if float(max_norm) < 0:
            raise ValueError(
                f"ClippedPytree max_norm must be non-negative, got {grads.max_norm}"
            )
    if first_max_norm is not None and max_norm != first_max_norm:
        raise ValueError(
            f"{op} saw a varying ClippedPytree.max_norm across calls "
            f"(first={first_max_norm}, now={max_norm}). MF privacy proofs "
            "assume a constant per-step sensitivity; this is satisfied by "
            "fixed and AUTO-S clipping but not by adaptive clipping, which "
            "is therefore unsupported with MF noise."
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
) -> tuple[
    Callable[
        [Any, SecondMomentMFNoiseState],
        tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState],
    ],
    SecondMomentMFNoiseState,
]:
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
        # Strategy norms enter the per-record sensitivity on each stream
        # before the joint Mahalanobis allocation: Δ¹ = ζ·‖C₁‖, Δ² = ζ²·‖C₂‖.
        # The single-stream MF release has effective Gaussian noise
        # multiplier ``nm / ‖C₁‖`` on the encoded stream (`MfGaussian` PLD
        # is ``gaussian_pld(nm / sensitivity)``).  To match that calibration
        # the joint paired Mahalanobis budget must equal ``(‖C₁‖ / nm)²``
        # — pass ``nm / ‖C₁‖`` as the joint effective multiplier so the
        # allocator's ``1 / nm²`` budget identity hits the right value.
        c1 = first_strategy._max_column_norm
        c2 = second_strategy._max_column_norm
        first_stddev, second_stddev = paired_noise_stddevs(
            noise_multiplier / c1,
            first=max_norm * c1,
            second=squared_max_norm * c2,
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
        sync_fp_first = _mf_per_group_sync_fingerprint_for_latch(
            st._first_state, max_norm
        )
        sync_fp_second = _mf_per_group_sync_fingerprint_for_latch(
            st._second_state, squared_max_norm
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
                _first_state=replace(
                    new_first,
                    _first_max_norm=max_norm,
                    _first_max_norm_sync_fingerprint=sync_fp_first,
                ),
                _second_state=replace(
                    new_second,
                    _first_max_norm=squared_max_norm,
                    _first_max_norm_sync_fingerprint=sync_fp_second,
                ),
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
    "MfStrategy",
    "SecondMomentMFNoiseState",
    "SecondMomentNoiseOutput",
    "band_mf_strategy",
    "blt_strategy",
    "identity_strategy",
    "lambda_cgd_strategy",
    "bisr_strategy",
    "bsr_strategy",
    "mf_noise",
]
