"""Gaussian and bounded-Gaussian DP-SGD noise mechanisms."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

from opaque.api.engine import ops, runtime
from opaque.api.engine.noise_allocation import (
    PAIRED_FIRST_STREAM_FOLD,
    PAIRED_SECOND_STREAM_FOLD,
    per_group_noise_stddev,
    resolve_paired_clipped,
)
from opaque.api.engine.pytree import tree_flatten_with_paths, tree_unflatten
from opaque.random import fold_in as rng_fold_in
from opaque.random import normal
from opaque.random.types import RngKey
from opaque.types import (
    ClippedPytree,
    NoisedPytree,
    NoiseState,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

if TYPE_CHECKING:
    from opaque.api.dpsgd.noise._types import GaussianNoiseFn

# Root of every key this mechanism derives.  A mechanism's whole key space
# hangs off one namespaced string, so a caller who hands the same base key to
# two mechanisms still gets independent noise from each.  Without a root the
# obvious derivation — ``fold_in(key, step)`` — is what *every* mechanism
# writes, and two of them drawing from it produce byte-identical noise with
# nothing to signal it.  See ``docs/reference/rng.md``.
GAUSSIAN_STREAM_FOLD = "opaque.dpsgd.gaussian"

_SQRT2 = math.sqrt(2.0)


@dataclasses.dataclass(frozen=True)
class GaussianNoiseState(NoiseState):
    """Immutable state for Gaussian noise generation.

    Holds an immutable RNG key for deterministic per-step derivation.
    Noise for step ``t`` is generated from
    ``fold_in(_rng_key, GAUSSIAN_STREAM_FOLD, t)``.

    Attributes:
        _step_counter: Number of noise_fn calls made.
        _rng_key: Immutable RNG key for deterministic derivation.
    """

    _step_counter: int
    _rng_key: RngKey


def _validate_noise_stddev(noise_stddev: float | PerGroup) -> None:
    """Validate that realized noise standard deviation is non-negative."""
    if isinstance(noise_stddev, PerGroup):
        for gname, value in noise_stddev.values.items():
            if value < 0:
                raise ValueError(
                    "noise standard deviation must be non-negative for all groups, "
                    f"got {value} for group '{gname}'"
                )
    else:
        if noise_stddev < 0:
            raise ValueError(
                f"noise standard deviation must be non-negative, got {noise_stddev}"
            )


def _resolve_noise_multiplier(noise_multiplier: float | None) -> float:
    if noise_multiplier is None:
        raise ValueError("gaussian_noise() requires noise_multiplier.")
    multiplier = float(noise_multiplier)
    if multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    return multiplier


def _resolve_bound(
    bound: float | tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    """Normalize the user-facing ``bound`` argument to ``(low, high)`` or None.

    Accepts ``None`` (unbounded), a positive scalar ``B`` interpreted as the
    symmetric interval ``[-B, B]``, or a 2-tuple/2-list ``(low, high)`` with
    ``low < high``.  Bounds must straddle zero — the support has to contain
    the unbiased mechanism centre — so ``low <= 0 <= high`` is required.
    """
    if bound is None:
        return None
    if isinstance(bound, (tuple, list)):
        if len(bound) != 2:  # noqa: PLR2004 - bound is the documented (low, high) pair
            raise ValueError(
                f"bound must be a 2-tuple (low, high) when given a sequence, "
                f"got length {len(bound)}"
            )
        low, high = float(bound[0]), float(bound[1])
    else:
        b = float(bound)
        if b <= 0:
            raise ValueError(
                f"scalar bound must be positive (interpreted as [-B, B]), got {b}"
            )
        low, high = -b, b
    if not low < high:
        raise ValueError(f"bound must satisfy low < high, got ({low}, {high})")
    if not (low <= 0.0 <= high):
        raise ValueError(
            "bound must straddle zero (low <= 0 <= high) so the support "
            f"contains the unbiased mechanism centre, got ({low}, {high})"
        )
    return low, high


def gaussian_noise(
    *,
    noise_multiplier: float,
    key: RngKey,
    bound: float | tuple[float, float] | list[float] | None = None,
    compute_dtype: object | None = None,
) -> tuple[
    GaussianNoiseFn,
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise to :class:`opaque.types.ClippedPytree` inputs and returns updated
    state.  The realized standard deviation is
    ``noise_multiplier * clipped.max_norm``.  The output is a
    :class:`opaque.types.NoisedPytree` carrying that realized
    ``noise_stddev`` metadata.

    When ``bound`` is set, the output is sampled from a Gaussian renormalized
    over the per-coordinate interval and clamped to it — the *bounded
    Gaussian mechanism* of Chen and Hale (2024).

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank. For independent noise, derive a per-rank key::

        from opaque.random import key, fold_in
        my_key = fold_in(key(42), rank)  # different noise per rank
        noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=my_key)

    Args:
        noise_multiplier: Gaussian noise multiplier.  The realized standard
            deviation is ``noise_multiplier * clipped.max_norm``.
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.
        bound: Optional per-coordinate output bound.  ``None`` (default) ⇒
            unbounded standard Gaussian.  A positive scalar ``B`` ⇒ symmetric
            interval ``[-B, B]``.  A ``(low, high)`` tuple/list ⇒ asymmetric
            interval; must satisfy ``low <= 0 <= high``.  Units are absolute
            (same scale as the gradient / clip norm), not multiples of σ.
        compute_dtype: Provider-native internal sampling dtype. Defaults to the
            active provider's ``float32``; finite precision discretizes and bounds
            the representable tails. Output is cast back to the input dtype.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(clipped_grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example:
        >>> from opaque.types import clipped
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=key(42))
        >>> grads = ...  # native array from the active provider
        >>> noisy_grads, state = noise_fn(clipped(grads, max_norm=1.0), state)

    Example (bounded output — symmetric ``[-3, 3]``):
        >>> noise_fn, state = gaussian_noise(
        ...     noise_multiplier=1.0, bound=3.0, key=key(42),
        ... )

    Example (bounded output — asymmetric ``[0, 10]``):
        >>> noise_fn, state = gaussian_noise(
        ...     noise_multiplier=1.0, bound=(0.0, 10.0), key=key(42),
        ... )

    References:
        Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
        Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
        https://arxiv.org/abs/2211.17230
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)
    resolved_bound = _resolve_bound(bound)

    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    state = GaussianNoiseState(
        _step_counter=0,
        _rng_key=key,
    )

    def _sample(center: Any, std: float, sample_key: RngKey) -> Any:
        """Sample N(center, std²), optionally truncated by inverse CDF.

        ``compute_dtype`` upcast/downcast preserves the type-stable boundary.
        Bounded sampling derives uniforms from a standard-normal CDF so every
        random draw remains available through the portable keyed normal contract.
        """
        if not ops.is_array(center):
            raise TypeError(
                "gaussian_noise expects native array leaves; "
                f"got {type(center).__name__}."
            )

        in_dtype = ops.dtype(center)

        if std == 0:
            if resolved_bound is None:
                return center
            low, high = resolved_bound
            return ops.clamp(center, lo=low, hi=high)

        sample_dtype = compute_dtype if compute_dtype is not None else ops.float32()
        center_c = (
            ops.astype(center, sample_dtype) if in_dtype != sample_dtype else center
        )
        standard_normal = normal(
            sample_key,
            ops.shape(center_c),
            dtype=sample_dtype,
            like=center_c,
        )
        if resolved_bound is None:
            sample = ops.add(center_c, ops.multiply(standard_normal, std))
        else:
            low, high = resolved_bound
            z_low = ops.divide(ops.subtract(low, center_c), std)
            z_high = ops.divide(ops.subtract(high, center_c), std)
            alpha = ops.multiply(ops.add(ops.erf(ops.divide(z_low, _SQRT2)), 1.0), 0.5)
            beta = ops.multiply(ops.add(ops.erf(ops.divide(z_high, _SQRT2)), 1.0), 0.5)
            uniform = ops.multiply(
                ops.add(ops.erf(ops.divide(standard_normal, _SQRT2)), 1.0),
                0.5,
            )
            uniform = ops.add(alpha, ops.multiply(uniform, ops.subtract(beta, alpha)))
            # Clamp away from 0 and 1 so ``erfinv`` cannot return ±inf.
            eps = ops.finfo_eps(sample_dtype)
            uniform = ops.clamp(uniform, lo=eps, hi=1.0 - eps)
            sample = ops.add(
                center_c,
                ops.multiply(
                    ops.erfinv(ops.subtract(ops.multiply(uniform, 2.0), 1.0)),
                    std * _SQRT2,
                ),
            )
            sample = ops.clamp(sample, lo=low, hi=high)

        return ops.astype(sample, in_dtype) if in_dtype != sample_dtype else sample

    def _add_noise_tree(
        grads: Any,
        effective_stddev: float | PerGroup,
        stream_key: RngKey,
    ) -> Any:
        _validate_noise_stddev(effective_stddev)

        paths, leaves, treedef = tree_flatten_with_paths(grads)
        noised_leaves = []
        for leaf_index, (path, leaf) in enumerate(zip(paths, leaves, strict=True)):
            if not ops.is_array(leaf):
                raise TypeError(
                    "gaussian_noise expects native array leaves; "
                    f"got {type(leaf).__name__} at path {path!r}."
                )
            leaf_stddev = (
                effective_stddev.for_path(path)
                if isinstance(effective_stddev, PerGroup)
                else effective_stddev
            )
            noised_leaves.append(
                _sample(
                    leaf,
                    leaf_stddev,
                    rng_fold_in(stream_key, "gaussian_noise_leaf", leaf_index),
                )
            )
        return tree_unflatten(treedef, noised_leaves)

    def _clipped_stddev(clipped: ClippedPytree) -> float | PerGroup:
        if isinstance(clipped.max_norm, PerGroup):
            return per_group_noise_stddev(clipped.max_norm, resolved_noise_multiplier)
        # Non-private run: force std = 0 before the ``nm * max_norm`` product so
        # a disabled clip (max_norm = +inf) gives ``0`` rather than ``0 * inf =
        # NaN`` (the ``std == 0`` short-circuit in ``_sample`` then fires).
        effective = (
            0.0
            if resolved_noise_multiplier == 0.0
            else resolved_noise_multiplier * clipped.max_norm
        )
        _validate_noise_stddev(effective)
        return effective

    def _add_paired(
        clipped_input: SecondMomentClippingOutput, st: GaussianNoiseState
    ) -> tuple[SecondMomentNoiseOutput, GaussianNoiseState]:
        first_clipped, second_clipped, first_stddev, second_stddev = (
            resolve_paired_clipped(
                clipped_input,
                noise_multiplier=resolved_noise_multiplier,
            )
        )
        # Two independent noise streams, both beneath this mechanism's root
        # so they cannot collide with each other or with the single-stream
        # derivation (``fold_in(_rng_key, GAUSSIAN_STREAM_FOLD, step)``).
        first_step_key = rng_fold_in(
            st._rng_key,
            GAUSSIAN_STREAM_FOLD,
            PAIRED_FIRST_STREAM_FOLD,
            st._step_counter,
        )
        second_step_key = rng_fold_in(
            st._rng_key,
            GAUSSIAN_STREAM_FOLD,
            PAIRED_SECOND_STREAM_FOLD,
            st._step_counter,
        )
        noisy_grads = _add_noise_tree(
            first_clipped.pytree,
            first_stddev,
            first_step_key,
        )
        noisy_squared = _add_noise_tree(
            second_clipped.pytree,
            second_stddev,
            second_step_key,
        )
        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
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
            next_state,
        )

    def noise_fn(grads, st):
        """Add Gaussian noise to a clipped pytree (or paired stream)."""
        with runtime.trace_scope("opaque::gaussian_noise"):
            return _noise_fn_impl(grads, st)

    def _noise_fn_impl(grads, st):
        if isinstance(grads, SecondMomentClippingOutput):
            return _add_paired(grads, st)

        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )

        if isinstance(grads, NoisedPytree):
            raise TypeError(
                "gaussian_noise expects ClippedPytree inputs, not NoisedPytree "
                "values that have already passed through a noise mechanism."
            )

        if not isinstance(grads, ClippedPytree):
            raise TypeError(
                "gaussian_noise expects ClippedPytree inputs. Wrap manual "
                "values with opaque.types.clipped(...)."
            )

        effective_stddev = _clipped_stddev(grads)
        step_key = rng_fold_in(st._rng_key, GAUSSIAN_STREAM_FOLD, st._step_counter)
        noisy_tree = _add_noise_tree(
            grads.pytree,
            effective_stddev,
            step_key,
        )
        return (
            NoisedPytree(
                pytree=noisy_tree,
                max_norm=grads.max_norm,
                noise_stddev=effective_stddev,
            ),
            next_state,
        )

    return noise_fn, state


__all__ = ["GaussianNoiseState", "gaussian_noise"]
