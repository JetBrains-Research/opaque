"""Distributed training helpers for differential privacy.

Submodules:

- :mod:`opaque.distributed.collectives` — thin wrappers over
  ``torch.distributed`` (``is_distributed``, ``get_rank``, ``get_world_size``,
  ``all_reduce``, ``barrier``).
- :mod:`opaque.distributed.gradients` — pytree / gradient reduction
  (``reduce_pytree``, ``sum_gradients`` and in-place variants).
- :mod:`opaque.distributed.state` — state synchronization primitives,
  gather helpers, and the type-dispatched :func:`sync` registry.
- :mod:`opaque.distributed.shard` — dataset sharding (:func:`local_shard`).

The user-facing API is flattened to this package: ``from opaque.distributed
import sync, sum_gradients, local_shard, ...``.
"""

from opaque.distributed.collectives import (
    all_reduce,
    all_reduce_,
    barrier,
    get_rank,
    get_world_size,
    is_distributed,
)
from opaque.distributed.gradients import (
    reduce_pytree,
    reduce_pytree_,
    sum_gradients,
    sum_gradients_,
)
from opaque.distributed.shard import local_shard
from opaque.distributed.state import (
    assert_pytree_equal,
    assert_scalar_equal,
    gather_pytree,
    gather_tensors,
    reduce_scalar,
    register_sync_type,
    sync,
    sync_object,
)

__all__ = [
    # Collectives
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "all_reduce_",
    "barrier",
    # Gradient / pytree reduction
    "reduce_pytree",
    "reduce_pytree_",
    "sum_gradients",
    "sum_gradients_",
    # Scalar / tensor gathering
    "reduce_scalar",
    "gather_tensors",
    "gather_pytree",
    "assert_pytree_equal",
    "assert_scalar_equal",
    # Object / state sync
    "sync_object",
    "sync",
    "register_sync_type",
    # Shard
    "local_shard",
]
