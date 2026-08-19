"""Portable PyTree structure and backend-dispatched array operations.

The portable structural core consists of ``dict``, ``list``, and ``tuple``
containers. Every value not traversed by those containers is a *structural
leaf*, including native arrays, numbers, strings, and ``None``. Providers may
add nodes from their native registry, but those extensions are provider-owned
and are not portable to another backend.

``tree_flatten``, ``tree_flatten_with_paths``, ``tree_map``, and
``tree_structure`` operate on all structural leaves. ``tree_leaves`` and
``global_norm`` deliberately select only native-array leaves. Portable
dictionaries have deterministic provider traversal order; callers should use
the returned paths rather than relying on insertion order.

Treedefs are opaque provider values. They may only be compared with structures
and passed to ``tree_unflatten`` while the provider that created them is active;
they are not a cross-provider serialization format. For multi-tree mapping,
all trees must have the same provider-defined structure; provider-native
mismatch errors are propagated.
"""

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.primitive import PrimitiveTier, primitive

ParamPath = tuple[str | int, ...]
"""Portable leaf path: dict keys (``str``) and sequence indices (``int``).

Flat ``named_parameters`` dicts use a one-segment path, e.g.
``("layers.0.weight",)``.  Nested trees use multi-segment paths, e.g.
``("layers", 0, "weight")``. The two never collide.

A bare structural leaf uses the empty path ``()``.
"""


def param_path(path: tuple[Any, ...] | list[Any] | str) -> ParamPath:
    """Normalize an optree path (or flat string key) to :data:`ParamPath`.

    The empty path ``()`` is valid and denotes a root leaf (a pytree that
    is itself a single tensor).
    """
    if isinstance(path, str):
        return (path,)
    out: list[str | int] = []
    for part in path:
        if isinstance(part, (str, int)):
            out.append(part)
        else:
            raise TypeError(
                f"ParamPath components must be str or int; got {type(part).__name__}"
            )
    return tuple(out)


def param_path_display(path: ParamPath) -> str:
    """Dotted display form for pattern matching and diagnostics."""
    return ".".join(str(p) for p in path)


@primitive(tier=PrimitiveTier.CORE)
def tree_flatten_with_paths(
    tree: Any,
) -> tuple[list[ParamPath], list[Any], Any]:
    """Flatten all structural leaves to ``(paths, leaves, treedef)``.

    Providers normalize native paths to :data:`ParamPath`. A root leaf has
    path ``()``; a flat dotted dict key remains one component and is distinct
    from a nested path. The returned treedef belongs to the active provider.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    """Flatten all structural leaves to ``(leaves, treedef)``.

    Prefer :func:`tree_flatten_with_paths` when callers need leaf identities.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    """Rebuild a PyTree from ``treedef`` and ``leaves``.

    The provider that created ``treedef`` must be active. The number of leaves
    must match the treedef; provider-native mismatch errors are propagated.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def tree_structure(tree: Any) -> Any:
    """Return the active provider's opaque structure for ``tree``.

    Useful for asserting that independently gathered payloads share a layout
    before unflattening.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def tree_leaves(tree: Any) -> list[Any]:
    """Extract all native-array leaves from a PyTree.

    Args:
        tree: A portable tree, optionally including provider-native nodes.

    Returns:
        List of all native-array leaves in the tree (non-array leaves are ignored).

    Example:
        After selecting a provider, construct a nested tree with two
        provider-native array leaves. ``tree_leaves(tree)`` returns those two
        leaves in traversal order.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def tree_map(fn: Callable[..., Any], *trees: Any) -> Any:
    """Apply ``fn`` to every structural leaf of one or more PyTrees.

    Args:
        fn: Function to apply to each leaf (or set of leaves from multiple trees).
        *trees: One or more PyTrees with matching provider-defined structure.

    Returns:
        PyTree with the same structure, with ``fn`` applied to all leaves.

    Raises:
        Exception: A provider-native structural error if multiple trees do not
            have matching structures.

    Example:
        After selecting a provider, construct a tree of native arrays and
        apply ``tree_map(lambda x: x * 2, tree)``. The returned tree has the
        same structure with every native-array leaf doubled.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def _squared_l2_norms(
    leaves: list[Any],
    groups: list[str] | None,
    *,
    dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    """Accumulate total and optional grouped squared L2 norms natively.

    Per-element squaring runs in ``dtype``; the cross-element and cross-leaf
    accumulation may run in a wider provider-internal dtype, so the returned
    scalars carry that internal dtype. :func:`_squared_l2_norm_roundoff`
    reports the matching error bound.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def _squared_l2_norm_roundoff(
    leaves: list[Any],
    *,
    dtype: Any,
) -> float:
    """Relative error bound on ``sqrt`` of a :func:`_squared_l2_norms` result.

    Covers per-element squaring in ``dtype``, the provider's intra-leaf
    reduction structure, and the cross-leaf accumulation, halved for the
    square root. Depends only on leaf shapes and dtypes — never on values —
    so it stays a trace-time constant under compilation.
    """
    raise NotImplementedError


def _resolve_reduction_dtype(
    leaves: list[Any],
    compute_dtype: Any | None,
) -> Any:
    """Resolve a real accumulation dtype for squared L2 reductions."""
    if compute_dtype is not None:
        if not ops.is_floating(compute_dtype) or ops.is_complex(compute_dtype):
            raise TypeError(
                f"compute_dtype must be a real floating-point dtype, got "
                f"{compute_dtype!r}. Integer/bool/complex compute dtypes can "
                f"silently corrupt the L2-norm reduction."
            )
        return compute_dtype

    acc_dtype = ops.float32()
    for leaf in leaves:
        leaf_dtype = ops.dtype(leaf)
        if ops.is_complex(leaf_dtype):
            leaf_dtype = ops.real_dtype(leaf_dtype)
        elif not ops.is_floating(leaf_dtype):
            continue
        if not ops.is_low_precision(leaf_dtype):
            acc_dtype = ops.promote_dtype(acc_dtype, leaf_dtype)
    return acc_dtype


def tree_map_with_path(
    fn: Callable[[ParamPath, Any], Any],
    tree: Any,
) -> Any:
    """Apply function to leaves with their :data:`ParamPath` in the tree.

    Paths and leaf order match :func:`tree_flatten_with_paths` (optree
    traversal, including sorted dict keys), so results align with
    :class:`~opaque.types.PerGroup` keys.

    Args:
        fn: Function that takes ``(path, leaf)`` where ``path`` is a
            :data:`ParamPath`.
        tree: PyTree to traverse

    Returns:
        PyTree with same structure, with fn applied to (path, leaf)

    Example:
        After selecting a provider, pass a tree of native arrays and use a
        callback that inspects ``path`` and ``leaf.shape``. The callback sees
        provider-native shape values together with paths such as
        ``('layer1', 'weight')`` and ``('layer2', 'bias')``.
    """
    paths, leaves, treedef = tree_flatten_with_paths(tree)
    out = [fn(path, leaf) for path, leaf in zip(paths, leaves, strict=True)]
    return tree_unflatten(treedef, out)


def partition(
    predicate: Callable[[tuple[Any, ...], Any], bool],
    tree: Any,
) -> tuple[Any, Any]:
    """Partition PyTree into two trees based on predicate.

    This is the key function for LoRA-style training: split parameters into
    trainable (e.g., LoRA adapters) and frozen (e.g., pretrained backbone).

    Args:
        predicate: Function(path, value) -> bool. Returns True for first tree.
        tree: PyTree to partition

    Returns:
        Tuple of (matching_tree, non_matching_tree) with same structure as input.
        Branches where all leaves are filtered out are omitted.

    Example:
        After selecting a provider, pass a parameter tree whose leaves are
        native arrays and define ``is_lora(path, value)`` to identify adapter
        paths. ``partition(is_lora, params)`` returns matching and
        non-matching trees, preserving the nested structure where leaves are
        present.

    References:
        Inspired by Haiku's hk.data_structures.partition():
        https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/data_structures.py
    """

    def _partition_subtree(path: tuple, subtree: Any) -> tuple[Any, Any]:
        if isinstance(subtree, dict):
            true_dict = {}
            false_dict = {}
            for k, v in subtree.items():
                true_part, false_part = _partition_subtree((*path, k), v)
                if true_part is not None:
                    true_dict[k] = true_part
                if false_part is not None:
                    false_dict[k] = false_part
            return (
                true_dict or None,
                false_dict or None,
            )
        elif isinstance(subtree, (list, tuple)):
            true_list = []
            false_list = []
            for i, v in enumerate(subtree):
                true_part, false_part = _partition_subtree((*path, i), v)
                true_list.append(true_part)
                false_list.append(false_part)
            # Keep same type (list vs tuple)
            any_true = any(x is not None for x in true_list)
            any_false = any(x is not None for x in false_list)
            return (
                type(subtree)(true_list) if any_true else None,
                type(subtree)(false_list) if any_false else None,
            )
        else:
            # Leaf node - apply predicate
            if predicate(path, subtree):
                return subtree, None
            else:
                return None, subtree

    true_tree, false_tree = _partition_subtree((), tree)
    # Return empty dicts instead of None for empty trees
    if true_tree is None:
        true_tree = {} if isinstance(tree, dict) else type(tree)()
    if false_tree is None:
        false_tree = {} if isinstance(tree, dict) else type(tree)()
    return true_tree, false_tree


def merge(*trees: Any) -> Any:
    """Merge multiple PyTrees into one.

    Later trees override earlier ones for overlapping keys.

    Args:
        *trees: PyTrees to merge (must have compatible structures)

    Returns:
        Merged PyTree

    Example:
        After selecting a provider, pass trees containing native arrays to
        ``merge``. For example, merging a frozen tree with a trainable tree
        combines their ``'weight'`` and ``'lora_a'`` entries, with later trees
        overriding earlier values at overlapping paths.

    References:
        Inspired by Haiku's hk.data_structures.merge():
        https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/data_structures.py
    """
    if not trees:
        return {}

    # Filter out None/empty trees
    trees = [t for t in trees if t is not None]
    if not trees:
        return {}

    # Base case: single tree
    if len(trees) == 1:
        return trees[0]

    # Get first tree as reference
    result = trees[0]

    # Merge remaining trees
    for tree in trees[1:]:
        result = _merge_two(result, tree)

    return result


def _merge_two(tree1: Any, tree2: Any) -> Any:
    """Merge two PyTrees (tree2 overrides tree1)."""
    # If either is None, return the other
    if tree1 is None:
        return tree2
    if tree2 is None:
        return tree1

    # If both are dicts, merge recursively
    if isinstance(tree1, dict) and isinstance(tree2, dict):
        result = dict(tree1)  # Start with tree1
        for key in tree2:
            if key in result:
                # Recursively merge
                result[key] = _merge_two(result[key], tree2[key])
            else:
                # Add new key
                result[key] = tree2[key]
        return result

    # If both are lists/tuples, merge element-wise
    if isinstance(tree1, (list, tuple)) and isinstance(tree2, (list, tuple)):
        # Must have same length
        if len(tree1) != len(tree2):
            raise ValueError(
                f"Cannot merge sequences of different lengths: {len(tree1)} vs {len(tree2)}"
            )
        merged = [_merge_two(a, b) for a, b in zip(tree1, tree2, strict=True)]
        return type(tree1)(merged)

    # Otherwise, tree2 overwrites tree1
    return tree2


def global_norm(
    tree: Any,
    *,
    compute_dtype: Any | None = None,
) -> Any:
    """Compute global L2 norm across all native arrays in a PyTree.

    The global norm is the square root of the sum of squared norms of all
    native-array leaves:
        global_norm = sqrt(sum(||leaf||^2 for leaf in tree))

    Args:
        tree: PyTree of native arrays (e.g., parameters or gradients)
        compute_dtype: Internal accumulation dtype.  ``None`` (default)
            promotes low-precision floats (fp16/bf16) to float32 for
            numerical stability and otherwise uses the input float dtype
            promoted to at least float32. Pass an explicit provider dtype to
            force a specific accumulation precision.

    Returns:
        Scalar native array containing the global L2 norm on the device of the
        first leaf (or the provider default if the tree is empty). Output dtype matches
        the resolved compute dtype.

    Example:
        After selecting a provider, pass native arrays containing
        ``[3.0, 4.0]`` and ``[0.0, 12.0]`` to ``global_norm``. The result is
        the provider-native scalar value ``13``.

    References:
        This function is commonly used in gradient clipping for deep learning.
        See: Pascanu et al. 2013, "On the difficulty of training RNNs"
    """
    leaves = tree_leaves(tree)
    real_acc_dtype = _resolve_reduction_dtype(leaves, compute_dtype)
    if not leaves:
        return ops.scalar(0.0, dtype=real_acc_dtype)

    total, _ = _squared_l2_norms(leaves, None, dtype=real_acc_dtype)
    norm = ops.sqrt(total)
    if ops.dtype(norm) != real_acc_dtype:
        norm = ops.astype(norm, real_acc_dtype)
    return norm


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
