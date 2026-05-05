"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian noise
to bounded DP query values.

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.random import key
    >>> from opaque.bounded import bounded
    >>> from opaque.dpsgd.noise.gaussian import gaussian_noise
    >>> noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
    >>> noisy_grads, state = noise_fn(bounded(grads, bound=1.0), state)

The constructor takes a noise multiplier, not a raw standard deviation.
Per-step sensitivity flows through the input ``BoundedPytree.bound`` metadata,
and the returned ``NoisyPytree`` carries the realized ``noise_stddev`` for
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

from opaque.bounded import BoundedPytree, NoisyPytree
from opaque.clipping.per_group import PerGroup
from opaque.distributed import (
    is_distributed,
    register_sync_type,
    sync_object,
)
from opaque.core.noise import (
    NOISE_STATE_FIELD_OPS,
    NoiseState,
    assert_rng_key_equal,
)
from opaque.random import RngKey, generator_from_key
from opaque.random import (
    fold_in as rng_fold_in,
)
from opaque.core.pytree import tree_map
from opaque.dpsgd.noise.per_group_noise import per_group_noise_stddev


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
    noise to :class:`opaque.bounded.BoundedPytree` inputs and returns updated
    state.  The realized standard deviation is
    ``noise_multiplier * bounded.bound``.  The output is a
    :class:`opaque.bounded.NoisyPytree` carrying that realized
    ``noise_stddev`` metadata.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank. For independent noise, derive a per-rank key::

        from opaque.random import key, fold_in
        my_key = fold_in(key(42), rank)  # different noise per rank
        noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=my_key)

    Args:
        noise_multiplier: Gaussian noise multiplier.  The realized standard
            deviation is ``noise_multiplier * bounded.bound``.
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

        - ``noise_fn(bounded_grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example:
        >>> import torch
        >>> from opaque.bounded import bounded
        >>> from opaque.dpsgd.noise.gaussian import gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=key(42))
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(bounded(grads, bound=1.0), state)

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

        # Per-group noise path
        if isinstance(effective_stddev, PerGroup):
            if all(v == 0 for v in effective_stddev.values.values()):
                return grads

            noisy = {}
            for param_key, tensor in grads.items():
                group_std = effective_stddev.for_key(param_key)
                noisy[param_key] = _add_noise(tensor, group_std, generator)

            return noisy

        # Global (scalar) noise path
        if effective_stddev == 0:
            return grads

        return tree_map(lambda t: _add_noise(t, effective_stddev, generator), grads)

    def _bounded_stddev(bounded: BoundedPytree) -> float | PerGroup:
        if isinstance(bounded.bound, PerGroup):
            return per_group_noise_stddev(bounded.bound, resolved_noise_multiplier)
        effective = resolved_noise_multiplier * bounded.bound
        _validate_noise_stddev(effective)
        return effective

    def noise_fn(grads, st):
        """Add Gaussian noise to a bounded pytree."""
        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )

        if isinstance(grads, NoisyPytree):
            raise TypeError(
                "gaussian_noise expects BoundedPytree inputs, not NoisyPytree "
                "values that have already passed through a noise mechanism."
            )

        if not isinstance(grads, BoundedPytree):
            raise TypeError(
                "gaussian_noise expects BoundedPytree inputs. Wrap manual "
                "values with opaque.bounded.bounded(...)."
            )

        effective_stddev = _bounded_stddev(grads)
        step_key = rng_fold_in(st._rng_key, st._step_counter)
        noisy_tree = _add_noise_tree(
            grads.pytree,
            effective_stddev,
            generator_from_key(step_key),
        )
        return (
            NoisyPytree(
                pytree=noisy_tree,
                bound=grads.bound,
                noise_stddev=effective_stddev,
            ),
            next_state,
        )

    return noise_fn, state


# ---- Distributed state validation ----


def sync_gaussian_noise_state(state: GaussianNoiseState) -> GaussianNoiseState:
    """Validate Gaussian noise state consistency across ranks.

    Asserts that all ranks share the same seed and step counter.  No-op
    outside ``torch.distributed``.  Registered automatically with
    :func:`opaque.distributed.sync`.
    """
    if not is_distributed():
        return state
    assert_rng_key_equal(state, "GaussianNoiseState")
    return sync_object(state, field_ops=NOISE_STATE_FIELD_OPS)


register_sync_type(GaussianNoiseState, sync_gaussian_noise_state)


__all__ = ["gaussian_noise", "GaussianNoiseState", "sync_gaussian_noise_state"]
