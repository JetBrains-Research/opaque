"""Backend-dispatched pytree operations and structural helpers.

The ``ParamPath`` alias :func:`param_path` returns lives in
:mod:`opaque.pytree.types`.
"""

from opaque.api.engine.pytree import (
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
from opaque.pytree import types

__all__ = [
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
    "types",
]
