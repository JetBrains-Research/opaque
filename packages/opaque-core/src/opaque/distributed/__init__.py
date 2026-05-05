"""Distributed training helpers for differential privacy.

The top-level ``__all__`` covers the common DP-DDP flow:

- **detection**: :func:`is_distributed`, :func:`get_rank`, :func:`get_world_size`
- **gradient aggregation**: :func:`sum_gradients`
- **state sync**: :func:`sync` (type-dispatched; handles clipping + noise states)
- **dataset sharding**: :func:`local_shard`

Everything else (in-place reductions, raw all-reduce, scalar / tensor gather,
assertion helpers, object-level sync, custom-type registration) is
importable from this module for convenience but not part of the public
``__all__``. Reach them via the submodules for documented access:

- :mod:`opaque.distributed.collectives` — ``all_reduce`` (+ in-place),
  ``barrier``, and the detection helpers.
- :mod:`opaque.distributed.gradients` — ``reduce_pytree`` (+ in-place) and
  ``sum_gradients_`` (in-place variant).
- :mod:`opaque.distributed.state` — scalar / pytree reductions, gathers,
  assertions, ``sync_object``, ``register_sync_type`` (extension hook).
- :mod:`opaque.distributed.shard` — ``local_shard``.
"""

from opaque.distributed.collectives import all_reduce as all_reduce
from opaque.distributed.collectives import all_reduce_ as all_reduce_
from opaque.distributed.collectives import barrier as barrier
from opaque.distributed.collectives import (
    get_rank,
    get_world_size,
    is_distributed,
)
from opaque.distributed.state import assert_pytree_equal as assert_pytree_equal
from opaque.distributed.state import assert_scalar_equal as assert_scalar_equal
from opaque.distributed.state import gather_pytree as gather_pytree
from opaque.distributed.state import gather_tensors as gather_tensors
from opaque.distributed.state import reduce_scalar as reduce_scalar
from opaque.distributed.state import register_sync_type as register_sync_type
from opaque.distributed.state import sync_object as sync_object
from opaque.distributed.state import sync

# `gradients` imports ClippedPytree from `opaque.clipping.types`, which triggers
# `opaque.clipping.__init__` to load `opaque.clipping.distributed`, which in turn
# imports `gather_pytree` from this module. Importing `state` first ensures
# `gather_pytree` is bound before that cycle closes.
from opaque.distributed.gradients import reduce_pytree as reduce_pytree
from opaque.distributed.gradients import reduce_pytree_ as reduce_pytree_
from opaque.distributed.gradients import sum_gradients_ as sum_gradients_
from opaque.distributed.gradients import sum_gradients
from opaque.distributed.shard import local_shard

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "sum_gradients",
    "sync",
    "local_shard",
]
