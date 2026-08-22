# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""In-place Torch collectives for DP gradient reduction.

The neutral engine exposes out-of-place reductions (``reduce_pytree``,
``sum_gradients``) that clone before reducing so results are safe on every
provider.  Torch DDP training loops can avoid that extra gradient-tree
allocation with the in-place variants here, which mutate tensor leaves via
``torch.distributed.all_reduce`` directly.

In-place wrapper reductions are accepted only when the wrapper metadata stays
unchanged; reductions that rescale ``max_norm`` / ``noise_stddev`` (noised
``sum``, clipped/noised ``mean``) must go through the engine's out-of-place
``reduce_pytree``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from opaque.api.engine.distributed.gradients import assert_public_metadata_equal
from opaque.api.engine.pytree import tree_map
from opaque.api.engine.types import (
    ClippedPytree,
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

_OP_MAP = {
    "sum": dist.ReduceOp.SUM,
    # "mean" reduces as SUM and divides by world size afterwards: gloo has
    # no ReduceOp.AVG, and the two-step form matches AVG's semantics.
    "mean": dist.ReduceOp.SUM,
    "max": dist.ReduceOp.MAX,
    "min": dist.ReduceOp.MIN,
    "product": dist.ReduceOp.PRODUCT,
}

_WRAPPER_REDUCTION_OPS = {"sum", "mean"}


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _is_noised(pytree: Any) -> bool:
    return isinstance(pytree, NoisedPytree)


def _resolve_op(op: str) -> dist.ReduceOp:
    try:
        return _OP_MAP[op]
    except KeyError:
        raise ValueError(
            f"Invalid reduction operation: {op}. Must be one of: {list(_OP_MAP)}"
        ) from None


def all_reduce_(tensor: torch.Tensor, op: str = "sum") -> None:
    """All-reduce ``tensor`` across ranks in place.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialized.
        ValueError: If ``op`` is not a recognized reduction.
    """
    reduce_op = _resolve_op(op)
    if not _is_distributed():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Call torch.distributed.init_process_group() first."
        )
    dist.all_reduce(tensor, op=reduce_op)
    if op == "mean":
        tensor.div_(dist.get_world_size())


def _assert_wrapper_reduction_supported(pytree: ClippedPytree, op: str) -> None:
    if op not in _WRAPPER_REDUCTION_OPS:
        raise TypeError(
            f"{type(pytree).__name__} distributed reduction only supports "
            "op='sum' or op='mean'. "
            "Use `.pytree` and reconstruct with an explicit max_norm for other reductions."
        )


def _in_place_wrapper_metadata_changes(pytree: ClippedPytree, op: str) -> bool:
    if _world_size() == 1:
        return False
    if op == "mean":
        return True
    return _is_noised(pytree) and pytree.noise_stddev is not None


def reduce_pytree_(pytree: Any, op: str = "sum") -> None:
    """All-reduce tensor leaves in place.

    In-place wrapper reductions are accepted only when the wrapper metadata
    stays unchanged.  Use the engine's ``opaque.distributed.gradients.reduce_pytree``
    for reductions such as noised ``sum`` or clipped/noised ``mean`` that
    need updated metadata.
    """
    if isinstance(pytree, SecondMomentClippingOutput):
        reduce_pytree_(pytree.grads, op=op)
        reduce_pytree_(pytree.squared_grads, op=op)
        return

    if isinstance(pytree, SecondMomentNoiseOutput):
        reduce_pytree_(pytree.noisy_grads, op=op)
        reduce_pytree_(pytree.noisy_squared_grads, op=op)
        return

    if isinstance(pytree, ClippedPytree):
        _assert_wrapper_reduction_supported(pytree, op)
        if _in_place_wrapper_metadata_changes(pytree, op):
            raise TypeError(
                f"In-place {type(pytree).__name__} reduction would change metadata; "
                "use reduce_pytree() instead."
            )
        assert_public_metadata_equal(pytree.max_norm, name="ClippedPytree.max_norm")
        if _is_noised(pytree):
            assert_public_metadata_equal(
                pytree.noise_stddev,
                name="NoisedPytree.noise_stddev",
            )
        reduce_pytree_(pytree.pytree, op=op)
        return

    if not _is_distributed():
        return

    def _reduce(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            all_reduce_(leaf, op=op)
            return leaf
        raise TypeError(
            f"reduce_pytree_ expects tensor leaves after wrapper dispatch; "
            f"got {type(leaf).__name__}. Unwrap paired/custom containers "
            f"explicitly or register a reduction branch."
        )

    tree_map(_reduce, pytree)


def sum_gradients_(gradients: Any) -> None:
    """DP-specific alias for ``reduce_pytree_(op="sum")``."""
    reduce_pytree_(gradients, op="sum")


__all__ = [
    "all_reduce_",
    "reduce_pytree_",
    "sum_gradients_",
]
