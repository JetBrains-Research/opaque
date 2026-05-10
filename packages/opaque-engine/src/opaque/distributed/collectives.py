"""Distributed collectives — ``all_reduce``, ``barrier``, rank/world helpers."""

from opaque.api.engine.distributed.collectives import (
    all_reduce,
    all_reduce_,
    barrier,
    get_rank,
    get_world_size,
    is_distributed,
)

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "all_reduce",
    "all_reduce_",
    "barrier",
]
