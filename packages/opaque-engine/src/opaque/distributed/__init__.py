"""Distributed primitives for DP training.

DDP detection (``is_distributed``, ``get_rank``, ``get_world_size``),
collectives (``all_reduce``, ``barrier``), scalar and pytree gathers
(``reduce_scalar``, ``gather_pytree``), cross-rank sanity checks
(``assert_scalar_equal``, ``assert_string_equal``), gradient reductions
(``reduce_pytree``, ``sum_gradients``), wrapper-aware ``sync()``
dispatch, and ``local_shard`` for dataset partitioning.

The ``opaque.distributed.collectives`` and
``opaque.distributed.gradients`` submodules expose the lower-level
power-user primitives directly.
"""

from opaque.api.engine.distributed import (
    all_reduce,
    assert_scalar_equal,
    assert_string_equal,
    barrier,
    gather_for_metrics,
    gather_pytree,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    local_shard,
    num_processes,
    process_index,
    reduce_scalar,
    sum_gradients,
    sync,
    wait_for_everyone,
)
from opaque.api.engine.distributed.gradients import reduce_pytree

__all__ = [
    "all_reduce",
    "assert_scalar_equal",
    "assert_string_equal",
    "barrier",
    "gather_for_metrics",
    "gather_pytree",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "local_shard",
    "num_processes",
    "process_index",
    "reduce_pytree",
    "reduce_scalar",
    "sum_gradients",
    "sync",
    "wait_for_everyone",
]
