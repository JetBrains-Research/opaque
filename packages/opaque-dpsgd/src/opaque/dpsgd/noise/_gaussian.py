"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian noise
to clipped DP query values.

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.random import key
    >>> from opaque.types import clipped
    >>> from opaque.dpsgd.noise import gaussian_noise
    >>> noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
    >>> noisy_grads, state = noise_fn(clipped(grads, max_norm=1.0), state)

The constructor takes a noise multiplier, not a raw standard deviation.
Per-step sensitivity flows through the input ``ClippedPytree.max_norm`` metadata,
and the returned ``NoisedPytree`` carries the realized ``noise_stddev`` for
downstream optimizers.

The noise function is **purely local** — it uses exactly the key you provide.
For synchronized noise in distributed training, pass the same key on every rank.
For independent noise, derive a per-rank key with ``fold_in(key, rank)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from opaque.types import (
    ClippedPytree,
    NoiseState,
    NoisedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)
from opaque.random import generator_from_key
from opaque.random.types import RngKey
from opaque.random import fold_in as rng_fold_in
from opaque.pytree import tree_map
from opaque._noise_allocation import (
    PAIRED_FIRST_STREAM_FOLD,
    PAIRED_SECOND_STREAM_FOLD,
    per_group_noise_stddev,
    resolve_paired_clipped,
)


@dataclasses.dataclass(frozen=True)
class GaussianNoiseState(NoiseState):
    """Immutable state for Gaussian noise generation.

    Holds an immutable RNG key for deterministic per-step derivation.
    Noise for step ``t`` is generated from ``fold_in(_rng_key, t)``.

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


def gaussian_noise(
    *,
    noise_multiplier: float,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Callable[..., tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise to :class:`opaque.types.ClippedPytree` inputs and returns updated
    state.  The realized standard deviation is
    ``noise_multiplier * clipped.max_norm``.  The output is a
    :class:`opaque.types.NoisedPytree` carrying that realized
    ``noise_stddev`` metadata.

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
        compute_dtype: Internal dtype for ``torch.randn`` and the
            scale-and-add arithmetic.  Defaults to ``torch.float32`` because
            the Gaussian-mechanism privacy guarantee requires sampling from a
            true Gaussian — ``torch.randn(dtype=torch.bfloat16)`` samples
            from a coarsely-discretized lattice that does not satisfy the
            standard analysis.  The type-stable boundary is preserved: the
            input's dtype is matched on output (input upcast to
            ``compute_dtype``, noise added, downcast at return).

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(clipped_grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example:
        >>> import torch
        >>> from opaque.types import clipped
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=key(42))
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(clipped(grads, max_norm=1.0), state)

    Example (distributed — independent noise per rank):
        >>> from opaque.random import key, fold_in
        >>> rank = torch.distributed.get_rank()
        >>> noise_fn, state = gaussian_noise(
        ...     noise_multiplier=1.1, key=fold_in(key(42), rank)
        ... )
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)

    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    state = GaussianNoiseState(
        _step_counter=0,
        _rng_key=key,
    )

    def _add_noise(tensor: torch.Tensor, std: float, generator) -> torch.Tensor:
        """Sample noise in compute_dtype, add, downcast to input dtype."""
        noise = torch.randn(
            tensor.shape,
            dtype=compute_dtype,
            generator=generator,
        ).to(device=tensor.device)
        if tensor.dtype == compute_dtype:
            return tensor + noise * std
        # Type-stable boundary: upcast input, add in compute_dtype, downcast.
        return (tensor.to(compute_dtype) + noise * std).to(dtype=tensor.dtype)

    def _add_noise_tree(grads, effective_stddev, generator):
        _validate_noise_stddev(effective_stddev)

        # Per-group σ: ClippedPytree.pytree must be a flat dict[path_key, Tensor].
        if isinstance(effective_stddev, PerGroup):
            if all(v == 0 for v in effective_stddev.values.values()):
                return grads

            noised = {}
            for param_key, tensor in grads.items():
                group_std = effective_stddev.for_key(param_key)
                noised[param_key] = _add_noise(tensor, group_std, generator)

            return noised

        # Global (scalar) noise path
        if effective_stddev == 0:
            return grads

        return tree_map(lambda t: _add_noise(t, effective_stddev, generator), grads)

    def _clipped_stddev(clipped: ClippedPytree) -> float | PerGroup:
        if isinstance(clipped.max_norm, PerGroup):
            return per_group_noise_stddev(clipped.max_norm, resolved_noise_multiplier)
        effective = resolved_noise_multiplier * clipped.max_norm
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
        # Two independent noise streams; fold-in tags namespace them so they
        # don't collide with the single-stream key derivation
        # (``fold_in(_rng_key, _step_counter)``).
        first_step_key = rng_fold_in(
            rng_fold_in(st._rng_key, PAIRED_FIRST_STREAM_FOLD), st._step_counter
        )
        second_step_key = rng_fold_in(
            rng_fold_in(st._rng_key, PAIRED_SECOND_STREAM_FOLD), st._step_counter
        )
        noisy_grads = _add_noise_tree(
            first_clipped.pytree,
            first_stddev,
            generator_from_key(first_step_key),
        )
        noisy_squared = _add_noise_tree(
            second_clipped.pytree,
            second_stddev,
            generator_from_key(second_step_key),
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
        step_key = rng_fold_in(st._rng_key, st._step_counter)
        noisy_tree = _add_noise_tree(
            grads.pytree,
            effective_stddev,
            generator_from_key(step_key),
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


__all__ = ["gaussian_noise", "GaussianNoiseState"]
