"""Distributed façade — re-exports from ``opaque.api.engine.distributed``.

Submodule façades ``opaque.distributed.collectives`` and
``opaque.distributed.gradients`` mirror the impl tree's
``opaque.api.engine.distributed.{collectives,gradients}`` for callers
that import the submodules by name (``from opaque.distributed.collectives
import all_reduce``).
"""

from opaque.api.engine.distributed import (
    all_reduce,
    get_rank,
    get_world_size,
    is_distributed,
    local_shard,
    sum_gradients,
    sum_gradients_,
    sync,
)
from opaque.api.engine.distributed.gradients import reduce_pytree, reduce_pytree_

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "sum_gradients",
    "sum_gradients_",
    "reduce_pytree",
    "reduce_pytree_",
    "sync",
    "local_shard",
]
