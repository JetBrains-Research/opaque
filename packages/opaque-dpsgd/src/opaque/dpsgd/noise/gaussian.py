"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian noise
to gradients in DP-SGD (Differentially Private Stochastic Gradient Descent).

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.random import key
    >>> from opaque.dpsgd.noise.gaussian import gaussian_noise
    >>> noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
    >>> noisy_grads, state = noise_fn(grads, state)

The ``stddev`` can be overridden per call for adaptive clipping:

    >>> noisy_grads, state = noise_fn(grads, state, stddev=new_stddev)

The noise function is **purely local** — it uses exactly the key you provide.
For synchronized noise in distributed training, pass the same key on every rank.
For independent noise, derive a per-rank key with ``fold_in(key, rank)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

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
from opaque.clipping.per_group import PerGroup
from opaque.core.pytree import tree_map


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


def _validate_stddev(stddev: float | PerGroup) -> None:
    """Validate that stddev is non-negative (scalar or per-group)."""
    if isinstance(stddev, PerGroup):
        for gname, val in stddev.values.items():
            if val < 0:
                raise ValueError(
                    f"stddev must be non-negative for all groups, "
                    f"got {val} for group '{gname}'"
                )
    else:
        if stddev < 0:
            raise ValueError(f"stddev must be non-negative, got {stddev}")


def gaussian_noise(
    stddev: float | PerGroup,
    *,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Callable[..., tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise N(0, stddev²) to gradients and returns updated state.

    The ``stddev`` provided here is the default. It can be overridden on each
    call via ``noise_fn(grads, state, stddev=new_stddev)`` — useful when the
    noise scale changes between steps (e.g., with adaptive clipping).

    When ``stddev`` is a :class:`~opaque.utils.per_group.PerGroup`, each
    parameter receives noise scaled by its group's stddev value.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank. For independent noise, derive a per-rank key::

        from opaque.random import key, fold_in
        my_key = fold_in(key(42), rank)  # different noise per rank
        noise_fn, state = gaussian_noise(stddev=1.1, key=my_key)

    Args:
        stddev: Standard deviation of Gaussian noise
            (usually ``noise_multiplier * clip_state.sensitivity``).
            When ``PerGroup``, each parameter group gets its own noise scale.
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

        - ``noise_fn(grads, state, *, stddev=None) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example:
        >>> import torch
        >>> from opaque.dpsgd.noise.gaussian import gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = gaussian_noise(stddev=1.1, key=key(42))
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(grads, state)

    Example (per-call override for adaptive clipping):
        >>> noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
        >>> noisy, state = noise_fn(grads, state, stddev=0.8)  # override this step

    Example (distributed — independent noise per rank):
        >>> from opaque.random import key, fold_in
        >>> rank = torch.distributed.get_rank()
        >>> noise_fn, state = gaussian_noise(stddev=1.1, key=fold_in(key(42), rank))
    """
    _validate_stddev(stddev)

    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    state = GaussianNoiseState(
        _step_counter=0,
        _rng_key=key,
    )

    default_stddev = stddev

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

    def noise_fn(grads, st, *, stddev=None):
        """Add Gaussian noise to gradients."""
        effective_stddev = stddev if stddev is not None else default_stddev
        _validate_stddev(effective_stddev)

        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )

        # Per-group noise path
        if isinstance(effective_stddev, PerGroup):
            if all(v == 0 for v in effective_stddev.values.values()):
                return grads, next_state

            step_key = rng_fold_in(st._rng_key, st._step_counter)
            g = generator_from_key(step_key)

            noisy = {}
            for param_key, tensor in grads.items():
                group_std = effective_stddev.for_key(param_key)
                noisy[param_key] = _add_noise(tensor, group_std, g)

            return noisy, next_state

        # Global (scalar) noise path
        if effective_stddev == 0:
            return grads, next_state

        step_key = rng_fold_in(st._rng_key, st._step_counter)
        g = generator_from_key(step_key)

        noisy = tree_map(lambda t: _add_noise(t, effective_stddev, g), grads)

        return noisy, next_state

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
