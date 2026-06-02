"""Distributed training helpers for differential privacy.

Headline DP-DDP flow:

- **detection**: :func:`is_distributed`, :func:`get_rank`, :func:`get_world_size`
- **collectives**: :func:`all_reduce`
- **gradient aggregation**: :func:`sum_gradients` (out-of-place) and
  :func:`sum_gradients_` (in-place; mutates the clipped pytree leaves).
- **state sync**: :func:`sync` (type-dispatched; handles clipping + noise
  states, registered DP runtime objects).
- **dataset sharding**: :func:`local_shard`.

The two documented power-user submodules collect lower-level primitives:

- :mod:`opaque.distributed.collectives` — ``all_reduce`` (+ in-place),
  ``barrier``, and the detection helpers.
- :mod:`opaque.distributed.gradients` — ``reduce_pytree`` (+ in-place) and
  ``sum_gradients`` (+ in-place).
"""

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
from opaque.api.engine.distributed.gradients import sum_gradients, sum_gradients_
from opaque.api.engine.distributed._shard import local_shard
from opaque.api.engine.distributed._state import register_sync_type, sync

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "barrier",
    "sum_gradients",
    "sum_gradients_",
    "sync",
    "register_sync_type",
    "local_shard",
    "is_main_process",
    "num_processes",
    "process_index",
    "wait_for_everyone",
    "gather_for_metrics",
]
