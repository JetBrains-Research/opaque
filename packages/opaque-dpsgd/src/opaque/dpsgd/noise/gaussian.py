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
import math
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
    SecondMomentNoiseOutput,
    assert_rng_key_equal,
    resolve_second_moment_overhead,
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


def _resolve_stddev(
    stddev: float | PerGroup | None,
) -> float | PerGroup:
    """Resolve the required Gaussian noise standard deviation."""
    if stddev is None:
        raise ValueError("gaussian_noise() requires stddev.")
    return stddev


def _scale_stddev(stddev: float | PerGroup, factor: float) -> float | PerGroup:
    if isinstance(stddev, PerGroup):
        return PerGroup(
            stddev.groups,
            {name: value * factor for name, value in stddev.values.items()},
        )
    return float(stddev) * factor


def _second_moment_stddev_from_base(
    stddev: float | PerGroup,
    sensitivity: float | PerGroup | None,
    overhead: float,
) -> tuple[float | PerGroup, float | PerGroup]:
    if sensitivity is None:
        raise ValueError(
            "second_moment=True requires stddev and sensitivity so both moment "
            "streams can be calibrated."
        )

    first_stddev = _scale_stddev(stddev, overhead)
    denom = math.sqrt(overhead**2 - 1.0)

    if isinstance(stddev, PerGroup):
        if isinstance(sensitivity, PerGroup):
            second_values = {
                name: stddev.values[name] * sensitivity.values[name] * overhead / denom
                for name in stddev.values
            }
        else:
            second_values = {
                name: value * float(sensitivity) * overhead / denom
                for name, value in stddev.values.items()
            }
        return first_stddev, PerGroup(stddev.groups, second_values)

    if isinstance(sensitivity, PerGroup):
        return first_stddev, PerGroup(
            sensitivity.groups,
            {
                name: float(stddev) * value * overhead / denom
                for name, value in sensitivity.values.items()
            },
        )

    return first_stddev, float(stddev) * float(sensitivity) * overhead / denom


def _resolve_second_moment_stddevs(
    stddev: float | PerGroup | None,
    sensitivity: float | PerGroup | None,
    overhead: float,
) -> tuple[float | PerGroup, float | PerGroup]:
    return _second_moment_stddev_from_base(
        _resolve_stddev(stddev),
        sensitivity,
        overhead,
    )


def gaussian_noise(
    stddev: float | PerGroup | None = None,
    *,
    key: RngKey,
    sensitivity: float | PerGroup | None = None,
    second_moment: bool | float = False,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Callable[..., tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise to gradients and returns updated state.

    The noise scale is supplied as ``stddev``.  Without ``second_moment``, the
    scale can be overridden on each call via
    ``noise_fn(grads, state, stddev=new_stddev)``.

    When ``second_moment`` is ``True`` or a float overhead, ``noise_fn`` returns
    :class:`opaque.core.noise.SecondMomentNoiseOutput` containing both noisy
    gradients and noisy element-wise squared gradients.  In that mode,
    ``sensitivity`` is also required so the squared-gradient stream can be
    calibrated.  Per-call overrides can pass both ``stddev`` and ``sensitivity``
    to track adaptive clipping changes.

    When ``stddev`` is a :class:`~opaque.utils.per_group.PerGroup`, each
    parameter receives noise scaled by its group's stddev value.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank. For independent noise, derive a per-rank key::

        from opaque.random import key, fold_in
        my_key = fold_in(key(42), rank)  # different noise per rank
        noise_fn, state = gaussian_noise(stddev=1.1, key=my_key)

    Args:
        stddev: Standard deviation of Gaussian noise.
        sensitivity: Clipped-gradient sensitivity.  Required when using
            ``second_moment`` because the squared-gradient stream sensitivity is
            derived from it.
        second_moment: ``False`` for the regular single stream.  ``True``
            enables the default private second-moment overhead ``sqrt(3/2)``;
            a float supplies the first-stream overhead directly and must be
            greater than 1.
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
    second_moment_enabled = second_moment is not False
    if second_moment_enabled:
        overhead = resolve_second_moment_overhead(second_moment)
        default_stddev, default_second_stddev = _resolve_second_moment_stddevs(
            stddev,
            sensitivity,
            overhead,
        )
    else:
        default_stddev = _resolve_stddev(stddev)
        default_second_stddev = None

    _validate_stddev(default_stddev)
    if default_second_stddev is not None:
        _validate_stddev(default_second_stddev)

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
        _validate_stddev(effective_stddev)

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

    def noise_fn(
        grads,
        st,
        *,
        stddev=None,
        sensitivity=None,
    ):
        """Add Gaussian noise to gradients."""
        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )

        if second_moment_enabled:
            effective_stddev = stddev if stddev is not None else locals_default_stddev
            effective_sensitivity = (
                sensitivity if sensitivity is not None else locals_default_sensitivity
            )
            first_stddev, second_stddev = _resolve_second_moment_stddevs(
                effective_stddev,
                effective_sensitivity,
                overhead,
            )
            step_key = rng_fold_in(st._rng_key, st._step_counter)
            first_key = rng_fold_in(step_key, 0)
            second_key = rng_fold_in(step_key, 1)
            noisy = _add_noise_tree(grads, first_stddev, generator_from_key(first_key))
            squared = tree_map(lambda grad: grad * grad, grads)
            noisy_squared = _add_noise_tree(
                squared,
                second_stddev,
                generator_from_key(second_key),
            )
            return SecondMomentNoiseOutput(noisy, noisy_squared), next_state

        effective_stddev = stddev if stddev is not None else default_stddev

        step_key = rng_fold_in(st._rng_key, st._step_counter)
        return _add_noise_tree(
            grads, effective_stddev, generator_from_key(step_key)
        ), next_state

    locals_default_stddev = stddev
    locals_default_sensitivity = sensitivity

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
