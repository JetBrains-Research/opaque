"""Distributed gradient reductions façade — re-exports from ``opaque.api.engine.distributed.gradients``."""

from opaque.api.engine.distributed.gradients import (
    reduce_pytree,
    reduce_pytree_,
    sum_gradients,
    sum_gradients_,
)

__all__ = [
    "reduce_pytree",
    "reduce_pytree_",
    "sum_gradients",
    "sum_gradients_",
]
