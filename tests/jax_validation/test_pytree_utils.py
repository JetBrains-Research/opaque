"""JAX validation tests for pytree_utils module.

These tests validate that our PyTorch pytree utilities produce numerically
equivalent results to JAX's pytree utilities. We run both implementations
on the same input data and compare outputs for closeness.

Module under test: opaque.pytree_utils
Functions tested:
    - tree_leaves() - Extract all leaf tensors from a PyTree
    - tree_map() - Apply function to all leaves in one or more PyTrees
    - global_norm() - Compute global L2 norm of all leaves

Run only with the jax-validation group:

    uv run --group jax-validation pytest -m jax_validation tests/jax_validation/test_pytree_utils.py -v

"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from opaque.pytree_utils import global_norm, tree_leaves, tree_map

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from jax import tree_util as jtu  # type: ignore
from jax.flatten_util import ravel_pytree  # type: ignore

pytestmark = [pytest.mark.jax_validation]

ATOL = 1e-5  # Standard tolerance for JAX-PyTorch numerical comparison


def _jax_to_torch(tree: Any) -> Any:
    """Convert JAX pytree to PyTorch pytree."""
    return jax.tree.map(
        lambda x: torch.from_numpy(np.asarray(x)) if isinstance(x, jnp.ndarray) else x, tree
    )


def _torch_to_jax(tree: Any) -> Any:
    """Convert PyTorch pytree to JAX pytree."""
    return jax.tree.map(
        lambda x: jnp.asarray(x.detach().cpu().numpy()) if isinstance(x, torch.Tensor) else x, tree
    )


def _make_pytree_jax() -> dict[str, Any]:
    """Create a JAX pytree for testing."""
    return {
        "layer1": {
            "w": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "b": jnp.asarray([0.5, -0.5], dtype=jnp.float32),
        },
        "layer2": {"w": jnp.asarray([1.0], dtype=jnp.float32), "b": jnp.asarray([0.0])},
    }


# ============================================================================
# tree_leaves tests
# ============================================================================


def test_tree_leaves_matches_jax():
    """Validate tree_leaves returns same leaves as JAX (modulo order)."""
    tree_jax = _make_pytree_jax()
    tree_torch = _jax_to_torch(tree_jax)

    leaves_jax = jtu.tree_leaves(tree_jax)
    leaves_torch = tree_leaves(tree_torch)

    # Check same number of leaves
    assert len(leaves_jax) == len(leaves_torch)

    # Check shapes match (sort by shape to handle potential reordering)
    shapes_jax = sorted([tuple(x.shape) for x in leaves_jax])
    shapes_torch = sorted([tuple(t.shape) for t in leaves_torch])
    assert shapes_jax == shapes_torch

    # Check numerical values match (compare element sums as invariant)
    sum_jax = float(sum([x.sum() for x in leaves_jax]))
    sum_torch = float(sum([t.sum() for t in leaves_torch]))
    assert math.isclose(sum_jax, sum_torch, abs_tol=ATOL)


def test_tree_leaves_empty_structures():
    """Validate tree_leaves handles empty structures like JAX."""
    for empty in ({}, [], ()):
        leaves_jax = jtu.tree_leaves(empty)
        leaves_torch = tree_leaves(empty)
        assert len(leaves_jax) == len(leaves_torch) == 0


def test_tree_leaves_deeply_nested():
    """Validate tree_leaves handles deeply nested structures."""
    tree_jax = {
        "a": {"aa": {"aaa": jnp.array([1.0, 2.0])}},
        "b": {"bb": jnp.array([3.0])},
        "c": jnp.array([4.0, 5.0, 6.0]),
    }
    tree_torch = _jax_to_torch(tree_jax)

    leaves_jax = jtu.tree_leaves(tree_jax)
    leaves_torch = tree_leaves(tree_torch)

    assert len(leaves_jax) == len(leaves_torch)


# ============================================================================
# tree_map tests
# ============================================================================


def test_tree_map_matches_jax():
    """Validate tree_map produces same structure and values as JAX."""
    tree_jax = _make_pytree_jax()
    tree_torch = _jax_to_torch(tree_jax)

    # Apply same function in both frameworks
    doubled_jax = jtu.tree_map(lambda x: x * 2, tree_jax)
    doubled_torch = tree_map(lambda x: x * 2, tree_torch)

    # Convert torch result back to JAX for comparison
    doubled_torch_as_jax = _torch_to_jax(doubled_torch)

    # Compare leaf by leaf
    leaves_jax = jtu.tree_leaves(doubled_jax)
    leaves_torch = jtu.tree_leaves(doubled_torch_as_jax)

    for x_jax, x_torch in zip(
        sorted(leaves_jax, key=lambda a: (a.size, float(a.flatten()[0]))),
        sorted(leaves_torch, key=lambda a: (a.size, float(a.flatten()[0]))),
    ):
        np.testing.assert_allclose(np.asarray(x_jax), np.asarray(x_torch), atol=ATOL)


def test_tree_map_with_multiple_trees():
    """Validate tree_map with multiple input trees matches JAX."""
    tree1_jax = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
    tree2_jax = {"a": jnp.array([4.0, 5.0]), "b": jnp.array([6.0])}

    tree1_torch = _jax_to_torch(tree1_jax)
    tree2_torch = _jax_to_torch(tree2_jax)

    # Sum corresponding leaves
    result_jax = jtu.tree_map(lambda x, y: x + y, tree1_jax, tree2_jax)
    result_torch = tree_map(lambda x, y: x + y, tree1_torch, tree2_torch)

    result_torch_as_jax = _torch_to_jax(result_torch)

    leaves_jax = jtu.tree_leaves(result_jax)
    leaves_torch = jtu.tree_leaves(result_torch_as_jax)

    for x_jax, x_torch in zip(leaves_jax, leaves_torch):
        np.testing.assert_allclose(np.asarray(x_jax), np.asarray(x_torch), atol=ATOL)


def test_tree_map_empty_tree():
    """Validate tree_map handles empty trees correctly."""
    result_jax = jtu.tree_map(lambda x: x * 2, {})
    result_torch = tree_map(lambda x: x * 2, {})
    assert result_jax == result_torch == {}


# ============================================================================
# global_norm tests
# ============================================================================


@pytest.mark.parametrize(
    "tree_fn, expected_norm",
    [
        (lambda: {}, 0.0),  # empty
        (lambda: {"x": jnp.array([3.0, 4.0])}, 5.0),  # [3,4] -> norm = 5
        (lambda: {"x": jnp.array([0.0, 0.0])}, 0.0),  # zeros
        (_make_pytree_jax, None),  # compute dynamically
    ],
)
def test_global_norm_matches_jax(tree_fn, expected_norm):
    """Validate global_norm produces same result as JAX global norm."""
    tree_jax = tree_fn()
    tree_torch = _jax_to_torch(tree_jax)

    # Compute norms
    flat_jax, _ = ravel_pytree(tree_jax)
    norm_jax = float(jnp.linalg.norm(flat_jax).item())
    norm_torch = float(global_norm(tree_torch).item())

    # If expected is provided, check both against it
    if expected_norm is not None:
        assert math.isclose(norm_jax, expected_norm, abs_tol=ATOL)
        assert math.isclose(norm_torch, expected_norm, abs_tol=ATOL)

    # Always check torch matches jax
    assert math.isclose(norm_torch, norm_jax, abs_tol=ATOL)


def test_global_norm_single_large_value():
    """Validate global_norm handles large single values correctly."""
    tree_jax = {"x": jnp.array([1e6])}
    tree_torch = _jax_to_torch(tree_jax)

    flat_jax, _ = ravel_pytree(tree_jax)
    norm_jax = float(jnp.linalg.norm(flat_jax).item())
    norm_torch = float(global_norm(tree_torch).item())

    # Use relative tolerance for large values
    assert math.isclose(norm_torch, norm_jax, rel_tol=1e-6, abs_tol=1e-5)


def test_global_norm_mixed_signs():
    """Validate global_norm handles mixed positive/negative values."""
    tree_jax = {"a": jnp.array([3.0, -4.0]), "b": jnp.array([-12.0, 5.0, 0.0])}
    tree_torch = _jax_to_torch(tree_jax)

    flat_jax, _ = ravel_pytree(tree_jax)
    norm_jax = float(jnp.linalg.norm(flat_jax).item())
    norm_torch = float(global_norm(tree_torch).item())

    # Expected: sqrt(9 + 16 + 144 + 25 + 0) = sqrt(194) ≈ 13.928
    assert math.isclose(norm_torch, norm_jax, abs_tol=ATOL)
    assert math.isclose(norm_torch, math.sqrt(194), abs_tol=ATOL)


def test_global_norm_deeply_nested():
    """Validate global_norm handles deeply nested structures."""
    tree_jax = {
        "a": {"aa": {"aaa": jnp.array([1.0, 2.0])}},
        "b": {"bb": jnp.array([3.0])},
        "c": jnp.array([4.0, 5.0, 6.0]),
    }
    tree_torch = _jax_to_torch(tree_jax)

    flat_jax, _ = ravel_pytree(tree_jax)
    norm_jax = float(jnp.linalg.norm(flat_jax).item())
    norm_torch = float(global_norm(tree_torch).item())

    assert math.isclose(norm_torch, norm_jax, abs_tol=ATOL)
