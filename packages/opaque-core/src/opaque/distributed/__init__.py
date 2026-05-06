"""Distributed training helpers for differential privacy.

Headline DP-DDP flow:

- **detection**: :func:`is_distributed`, :func:`get_rank`, :func:`get_world_size`
- **gradient aggregation**: :func:`sum_gradients`
- **state sync**: :func:`sync` (type-dispatched; handles clipping + noise states)
- **dataset sharding**: :func:`local_shard`

Lower-level primitives — in-place reductions, raw all-reduce, scalar /
tensor gather, assertion helpers, object-level sync, custom-type
registration — are reachable through the documented power-user submodules:

- :mod:`opaque.distributed.collectives` — ``all_reduce`` (+ in-place),
  ``barrier``, and the detection helpers.
- :mod:`opaque.distributed.gradients` — ``reduce_pytree`` (+ in-place) and
  ``sum_gradients_`` (in-place variant).
- :mod:`opaque.distributed.state` — scalar / pytree reductions, gathers,
  assertions, ``sync_object``, ``register_sync_type`` (extension hook).
- :mod:`opaque.distributed.shard` — ``local_shard``.
"""

from opaque.distributed.collectives import get_rank, get_world_size, is_distributed
from opaque.distributed.gradients import sum_gradients
from opaque.distributed.shard import local_shard
from opaque.distributed.state import sync

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "sum_gradients",
    "sync",
    "local_shard",
]
