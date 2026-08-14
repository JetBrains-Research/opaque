"""Portable pytree conformance shared by all first-party providers."""

from __future__ import annotations

import numpy as np
import pytest
from tests.integration.backend._providers import provider_case

from opaque import pytree
from opaque.backend import clear_backend, use_backend


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_portable_structure_round_trip_and_leaf_kinds(provider_name: str) -> None:
    case = provider_case(provider_name)
    first = case.array([1.0, 2.0])
    second = case.array([3.0])
    tree = {
        "empty": (),
        "params": [first, {"label": "weight", "value": second}],
        "step": 3,
    }

    with use_backend(case.backend):
        paths, leaves, treedef = pytree.tree_flatten_with_paths(tree)
        rebuilt = pytree.tree_unflatten(treedef, leaves)
        arrays = pytree.tree_leaves(tree)

    assert paths == [
        ("params", 0),
        ("params", 1, "label"),
        ("params", 1, "value"),
        ("step",),
    ]
    assert leaves[1] == "weight"
    assert leaves[3] == 3
    assert rebuilt["empty"] == ()
    assert rebuilt["params"][1]["label"] == "weight"
    assert len(arrays) == 2
    assert all(isinstance(value, case.array_type) for value in arrays)


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_root_leaf_empty_tree_and_normalized_paths(provider_name: str) -> None:
    case = provider_case(provider_name)
    tree = {
        "flat.key": case.array([1.0]),
        "nested": [case.array([2.0])],
    }

    with use_backend(case.backend):
        root_paths, root_leaves, root_def = pytree.tree_flatten_with_paths(case.value)
        empty_leaves, empty_def = pytree.tree_flatten({})
        paths, _, _ = pytree.tree_flatten_with_paths(tree)
        rebuilt_root = pytree.tree_unflatten(root_def, root_leaves)
        rebuilt_empty = pytree.tree_unflatten(empty_def, empty_leaves)

    assert root_paths == [()]
    assert isinstance(rebuilt_root, case.array_type)
    assert empty_leaves == []
    assert rebuilt_empty == {}
    assert paths == [("flat.key",), ("nested", 0)]


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_multi_tree_map_and_structure_mismatch(provider_name: str) -> None:
    case = provider_case(provider_name)
    left = {"metadata": "kept", "params": [case.array([1.0, 2.0])]}
    right = {"metadata": "kept", "params": [case.array([3.0, 4.0])]}

    def combine(first, second):
        if isinstance(first, case.array_type):
            return first + second
        return first

    with use_backend(case.backend):
        result = pytree.tree_map(combine, left, right)
        with pytest.raises(ValueError, match=r"(?i)(mismatch|structure|keys)"):
            pytree.tree_map(combine, left, {"params": right["params"]})

    case.evaluate(result["params"][0])
    assert result["metadata"] == "kept"
    assert case.to_numpy(result["params"][0]).tolist() == [4.0, 6.0]


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_global_norm_promotes_and_uses_complex_magnitude(provider_name: str) -> None:
    case = provider_case(provider_name)
    tree = {
        "complex": case.array([3 + 4j, 1 - 2j]),
        "integer": case.array([1, 2], case.dtype("int64")),
    }

    with use_backend(case.backend):
        norm = pytree.global_norm(tree)

    case.evaluate(norm)
    assert np.issubdtype(case.to_numpy(norm).dtype, np.floating)
    assert float(case.to_numpy(norm)) == pytest.approx(35**0.5, abs=1e-5)
