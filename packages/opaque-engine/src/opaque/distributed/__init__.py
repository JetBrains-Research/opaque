"""Distributed primitives for DP training.

DDP detection (``is_distributed``, ``get_rank``, ``get_world_size``),
collectives (``all_reduce``, ``barrier``), gradient reductions
(``reduce_pytree``, ``sum_gradients``), wrapper-aware ``sync()``
dispatch, and ``local_shard`` for dataset partitioning.

The ``opaque.distributed.collectives`` and
``opaque.distributed.gradients`` submodules expose the lower-level
power-user primitives directly.
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
