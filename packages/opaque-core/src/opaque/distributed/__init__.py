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

from opaque.distributed.collectives import (
    all_reduce,
    get_rank,
    get_world_size,
    is_distributed,
)
from opaque.distributed.gradients import sum_gradients, sum_gradients_
from opaque.distributed._shard import local_shard
from opaque.distributed._state import sync

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "sum_gradients",
    "sum_gradients_",
    "sync",
    "local_shard",
]
