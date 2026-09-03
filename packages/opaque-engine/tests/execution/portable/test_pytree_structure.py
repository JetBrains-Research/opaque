"""Portable public PyTree structure and parameter-partition behavior."""

from __future__ import annotations

from typing import Any

import optree

from opaque import ops
from opaque.pytree import (
    merge,
    partition,
    tree_flatten,
    tree_flatten_with_paths,
    tree_leaves,
    tree_map,
    tree_map_with_path,
    tree_structure,
    tree_unflatten,
)


class _Pair:
    def __init__(self, left: Any, right: Any) -> None:
        self.left = left
        self.right = right


def _array(backend_case, value: Any) -> Any:
    return backend_case.array(value, dtype=backend_case.dtype("float32"))


def _register_pair(backend_case) -> None:
    optree.register_pytree_node(
        _Pair,
        lambda pair: ((pair.left, pair.right), None),
        lambda _, children: _Pair(*children),
        namespace=f"opaque.{backend_case.name}",
    )


def test_tree_map_with_path_matches_flatten_paths_and_supports_root_leaves(
    backend_case,
) -> None:
    tree = {
        "layers": [
            {
                "weight": _array(backend_case, [1.0, 2.0]),
                "bias": _array(backend_case, [0.0]),
            },
            {"weight": _array(backend_case, [3.0])},
        ]
    }
    visited: list[tuple[str | int, ...]] = []

    mapped = tree_map_with_path(
        lambda path, leaf: visited.append(path) or ops.add(leaf, len(path)), tree
    )
    paths, _, _ = tree_flatten_with_paths(tree)

    assert visited == paths
    backend_case.assert_allclose(mapped["layers"][0]["weight"], [4.0, 5.0])

    root = _array(backend_case, [2.0, 3.0])
    root_paths, root_leaves, _ = tree_flatten_with_paths(root)
    mapped_root = tree_map_with_path(
        lambda path, leaf: (path, ops.multiply(leaf, 2)), root
    )
    assert root_paths == [()]
    assert len(root_leaves) == 1
    assert mapped_root[0] == ()
    backend_case.assert_allclose(mapped_root[1], [4.0, 6.0])


def test_partition_preserves_nested_dicts_lists_and_empty_branches(
    backend_case,
) -> None:
    tree = {
        "encoder": {
            "weight": _array(backend_case, [1.0, 2.0]),
            "bias": _array(backend_case, [0.0]),
        },
        "heads": [_array(backend_case, [3.0]), _array(backend_case, [4.0])],
    }

    selected, remainder = partition(lambda path, _: path[-1] in {"weight", 0}, tree)

    assert set(selected) == {"encoder", "heads"}
    assert set(selected["encoder"]) == {"weight"}
    assert set(remainder["encoder"]) == {"bias"}
    assert selected["heads"][0] is tree["heads"][0]
    assert selected["heads"][1] is None
    assert remainder["heads"][0] is None
    assert remainder["heads"][1] is tree["heads"][1]
    assert partition(lambda _path, _value: True, {}) == ({}, {})


def test_partition_and_merge_round_trip_lora_parameter_trees(backend_case) -> None:
    params = {
        "encoder": {
            "weight": _array(backend_case, [[1.0, 2.0], [3.0, 4.0]]),
            "bias": _array(backend_case, [0.0, 1.0]),
            "lora_a": _array(backend_case, [[1.0], [2.0]]),
            "lora_b": _array(backend_case, [[3.0, 4.0]]),
        },
        "decoder": {
            "weight": _array(backend_case, [5.0]),
            "lora_a": _array(backend_case, [6.0]),
        },
    }

    trainable, frozen = partition(lambda path, _: "lora" in str(path), params)
    updated_trainable = tree_map(
        lambda value: ops.add(value, 0.25) if ops.is_array(value) else value,
        trainable,
    )
    merged = merge(frozen, updated_trainable)

    assert set(merged["encoder"]) == set(params["encoder"])
    assert set(merged["decoder"]) == set(params["decoder"])
    backend_case.assert_allclose(
        merged["encoder"]["weight"], params["encoder"]["weight"]
    )
    backend_case.assert_allclose(merged["encoder"]["lora_a"], [[1.25], [2.25]])
    backend_case.assert_allclose(merged["decoder"]["lora_a"], [6.25])


def test_merge_combines_recursively_overrides_conflicts_and_keeps_leaf_identity(
    backend_case,
) -> None:
    first = {
        "encoder": {
            "weight": _array(backend_case, [1.0]),
            "bias": _array(backend_case, [2.0]),
        },
        "decoder": _array(backend_case, [3.0]),
    }
    replacement = _array(backend_case, [4.0])
    second = {"encoder": {"bias": replacement}, "head": _array(backend_case, [5.0])}

    merged = merge(first, None, second)

    assert merged["encoder"]["weight"] is first["encoder"]["weight"]
    assert merged["encoder"]["bias"] is replacement
    assert merged["decoder"] is first["decoder"]
    backend_case.assert_allclose(merged["head"], [5.0])


def test_flatten_map_and_unflatten_support_multiple_trees_and_registered_nodes(
    backend_case,
) -> None:
    _register_pair(backend_case)
    tree = _Pair(_array(backend_case, [1.0]), _array(backend_case, [2.0]))

    leaves, treedef = tree_flatten(tree)
    rebuilt = tree_unflatten(treedef, leaves)
    mapped = tree_map(lambda left, right: ops.add(left, right), tree, tree)

    assert tree_structure(tree) == treedef
    assert len(tree_leaves(tree)) == 2
    assert isinstance(rebuilt, _Pair)
    backend_case.assert_allclose(mapped.left, [2.0])
    backend_case.assert_allclose(mapped.right, [4.0])
