"""Backend-dispatched pytree operations and structural helpers."""

from opaque.api.engine.pytree import (
    ParamPath,
    global_norm,
    merge,
    param_path,
    param_path_display,
    partition,
    tree_flatten,
    tree_flatten_with_paths,
    tree_leaves,
    tree_map,
    tree_map_with_path,
    tree_structure,
    tree_unflatten,
)

__all__ = [
    "ParamPath",
    "global_norm",
    "merge",
    "param_path",
    "param_path_display",
    "partition",
    "tree_flatten",
    "tree_flatten_with_paths",
    "tree_leaves",
    "tree_map",
    "tree_map_with_path",
    "tree_structure",
    "tree_unflatten",
]
