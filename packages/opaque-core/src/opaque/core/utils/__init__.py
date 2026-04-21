"""Utility functions for opaque."""

from opaque.core.utils.functional import make_functional, with_batch_dim
from opaque.core.utils.per_group import PerGroup, per_group
from opaque.core.utils.pytree import (
    global_norm,
    merge,
    partition,
    tree_leaves,
    tree_map,
    tree_map_with_path,
)

__all__ = [
    "global_norm",
    "tree_leaves",
    "tree_map",
    "tree_map_with_path",
    "partition",
    "merge",
    "make_functional",
    "with_batch_dim",
    "PerGroup",
    "per_group",
]
