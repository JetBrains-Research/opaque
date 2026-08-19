"""Distributed gradient reductions — ``reduce_pytree``, ``sum_gradients``."""

from opaque.api.engine.distributed.gradients import (
    reduce_pytree,
    sum_gradients,
)

__all__ = [
    "reduce_pytree",
    "sum_gradients",
]


def __getattr__(name: str):
    # Transitional re-exports of the Torch in-place variants while
    # downstream packages migrate to ``opaque.torch.distributed``;
    # scheduled for removal once the migration completes.
    if name in ("all_reduce_", "reduce_pytree_", "sum_gradients_"):
        import opaque.api.torch.distributed as _torch_dist

        return getattr(_torch_dist, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
