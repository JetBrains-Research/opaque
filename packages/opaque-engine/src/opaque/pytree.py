"""Torch-pytree ops — façade re-exporting from ``opaque.api.engine.pytree``."""

from opaque.api.engine.pytree import (
    global_norm,
    merge,
    partition,
    tree_leaves,
    tree_map,
    tree_map_with_path,
)

__all__ = [
    "tree_leaves",
    "tree_map",
    "tree_map_with_path",
    "partition",
    "merge",
    "global_norm",
]
