"""Gradient / pytree reduction helpers for distributed DP training.

Provides ``reduce_pytree`` for generic per-leaf all-reduce and
``sum_gradients`` as a thin alias scoped to clipped gradients.

``ClippedPytree`` and ``NoisedPytree`` support the linear reductions whose DP
metadata semantics are determined: ``sum`` and ``mean``.  Summing disjoint local
clipped queries preserves the per-record max_norm; averaging divides it by world
size.  Summing independent Gaussian-noised local queries scales ``noise_stddev``
by ``sqrt(world_size)``; averaging scales it by ``1 / sqrt(world_size)``.

Paired-stream wrappers (:class:`~opaque.types.SecondMomentClippingOutput`,
:class:`~opaque.types.SecondMomentNoiseOutput`) recurse into both child
wrappers so both streams participate in the collective.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from opaque.api.engine import ops, runtime
from opaque.api.engine.pytree import tree_map
from opaque.api.engine.types import (
    ClippedPytree,
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

from ._state import assert_scalar_equal
from .collectives import get_world_size, is_distributed


def _is_noised(pytree: Any) -> bool:
    return isinstance(pytree, NoisedPytree)


_WRAPPER_REDUCTION_OPS = {"sum", "mean"}


def _assert_object_equal(value: Any, *, name: str) -> None:
    if not is_distributed():
        return
    gathered = runtime.distributed_all_gather_object(value)
    mismatched = [idx for idx, other in enumerate(gathered) if other != value]
    if mismatched:
        raise RuntimeError(
            f"{name} mismatch across ranks: mismatched ranks={mismatched}."
        )


def _assert_public_metadata_equal(value: Any, *, name: str) -> None:
    if not is_distributed():
        return
    if value is None:
        _assert_object_equal(value, name=name)
        return
    groups = getattr(value, "groups", None)
    values = getattr(value, "values", None)
    if isinstance(groups, dict) and isinstance(values, dict):
        _assert_object_equal(groups, name=f"{name}.groups")
        _assert_object_equal(set(values), name=f"{name}.values.keys")
        for group_name, value in values.items():
            assert_scalar_equal(value, name=f"{name}.values[{group_name!r}]")
        return
    assert_scalar_equal(value, name=name)


def _assert_wrapper_reduction_supported(pytree: ClippedPytree, op: str) -> None:
    if op not in _WRAPPER_REDUCTION_OPS:
        raise TypeError(
            f"{type(pytree).__name__} distributed reduction only supports "
            "op='sum' or op='mean'. "
            "Use `.pytree` and reconstruct with an explicit max_norm for other reductions."
        )


def _metadata_world_size() -> int:
    return get_world_size() if is_distributed() else 1


def _scale_public_metadata(value: Any, factor: float) -> Any:
    if value is None:
        return None
    return factor * value


def _reduced_metadata(pytree: ClippedPytree, op: str, world_size: int) -> ClippedPytree:
    if op == "sum":
        max_norm = pytree.max_norm
        if _is_noised(pytree):
            noise_stddev = _scale_public_metadata(
                pytree.noise_stddev,
                math.sqrt(float(world_size)),
            )
            return replace(pytree, max_norm=max_norm, noise_stddev=noise_stddev)
        return replace(pytree, max_norm=max_norm)

    if op == "mean":
        max_norm = _scale_public_metadata(pytree.max_norm, 1.0 / float(world_size))
        if _is_noised(pytree):
            noise_stddev = _scale_public_metadata(
                pytree.noise_stddev,
                1.0 / math.sqrt(float(world_size)),
            )
            return replace(pytree, max_norm=max_norm, noise_stddev=noise_stddev)
        return replace(pytree, max_norm=max_norm)

    raise AssertionError(f"Unsupported wrapper reduction op: {op}")


def reduce_pytree(pytree: Any, op: str = "sum") -> Any:
    """Return a pytree with each tensor leaf reduced; input unchanged.

    When passed ``ClippedPytree`` or ``NoisedPytree``, preserves and updates the
    wrapper metadata for supported ``sum`` and ``mean`` reductions.  Paired
    second-moment wrappers recurse into both child streams.
    """
    if isinstance(pytree, SecondMomentClippingOutput):
        return SecondMomentClippingOutput(
            grads=reduce_pytree(pytree.grads, op=op),
            squared_grads=reduce_pytree(pytree.squared_grads, op=op),
        )

    if isinstance(pytree, SecondMomentNoiseOutput):
        return SecondMomentNoiseOutput(
            noisy_grads=reduce_pytree(pytree.noisy_grads, op=op),
            noisy_squared_grads=reduce_pytree(pytree.noisy_squared_grads, op=op),
        )

    if isinstance(pytree, ClippedPytree):
        _assert_wrapper_reduction_supported(pytree, op)
        _assert_public_metadata_equal(pytree.max_norm, name="ClippedPytree.max_norm")
        if _is_noised(pytree):
            _assert_public_metadata_equal(
                pytree.noise_stddev,
                name="NoisedPytree.noise_stddev",
            )
        reduced = replace(pytree, pytree=reduce_pytree(pytree.pytree, op=op))
        return _reduced_metadata(reduced, op, _metadata_world_size())

    def _reduce(leaf: Any) -> Any:
        if ops.is_array(leaf):
            return runtime.distributed_all_reduce(leaf, runtime.ReduceOp(op))
        raise TypeError(
            f"reduce_pytree expects tensor leaves after wrapper dispatch; "
            f"got {type(leaf).__name__}. Unwrap paired/custom containers "
            f"explicitly or register a reduction branch."
        )

    return tree_map(_reduce, pytree)


def sum_gradients(gradients: Any) -> Any:
    """DP-specific alias for ``reduce_pytree(op="sum")``."""
    return reduce_pytree(gradients, op="sum")


__all__ = [
    "reduce_pytree",
    "sum_gradients",
]
