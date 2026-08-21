"""Distributed training helpers for differential privacy.

Headline DP-DDP flow:

- **detection**: :func:`is_distributed`, :func:`get_rank`, :func:`get_world_size`
- **collectives**: :func:`all_reduce`
- **gradient aggregation**: :func:`sum_gradients` (return-based).
- **state sync**: :func:`sync` (type-dispatched; handles clipping + noise
  states, registered DP runtime objects).
- **dataset sharding**: :func:`local_shard`.

The two documented power-user submodules collect lower-level primitives:

- :mod:`opaque.distributed.collectives` — ``all_reduce``,
  ``barrier``, and the detection helpers.
- :mod:`opaque.distributed.gradients` — ``reduce_pytree`` and ``sum_gradients``.
"""

from opaque.api.engine.distributed._shard import DatasetShard, local_shard
from opaque.api.engine.distributed._state import (
    assert_scalar_equal,
    assert_string_equal,
    gather_pytree,
    reduce_scalar,
    register_sync_type,
    sync,
    sync_object,
)
from opaque.api.engine.distributed.collectives import (
    all_reduce,
    barrier,
    gather_for_metrics,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    num_processes,
    process_index,
    wait_for_everyone,
)
from opaque.api.engine.distributed.gradients import sum_gradients

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
    "DatasetShard",
    "local_shard",
    "num_processes",
    "process_index",
    "reduce_scalar",
    "register_sync_type",
    "sync_object",
    "sum_gradients",
    "sync",
    "wait_for_everyone",
]
