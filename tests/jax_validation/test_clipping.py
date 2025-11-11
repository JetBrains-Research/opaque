"""JAX validation tests for clipping module.

These tests validate that our PyTorch clipping implementation produces numerically
equivalent results to JAX-Privacy's clipping utilities. We run both implementations
on the same input data and compare outputs for closeness, with special attention
to edge cases.

Module under test: opaque.clipping
Functions tested:
    - clip_pytree() - Clip PyTree to max L2 norm
    - clip_sum() - Wrap function to clip per-example outputs and sum

Reference: jax_privacy.experimental.clipping

Run:
    uv run --group jax-validation pytest -m jax_validation tests/jax_validation/test_clipping.py -v
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from opaque.clipping import clip_pytree, clip_sum

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.jax_validation]

ATOL = 1e-5  # Standard tolerance for JAX-PyTorch numerical comparison


def _import_jax_privacy_clipping():
    """Import JAX-Privacy clipping module with path fallbacks."""
    import importlib

    candidates = [
        "jax_privacy.experimental.clipping",
        "jax_privacy.src.experimental.clipping",
    ]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    pytest.skip("Could not import JAX-Privacy clipping (check ../jax_privacy)")


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
# clip_pytree tests
# ============================================================================


def test_clip_pytree_matches_jax_zero_clip():
    """Validate clip_pytree with clip_norm=0 matches JAX-Privacy."""
    jax_clip = _import_jax_privacy_clipping()
    tree_jax = _make_pytree_jax()
    tree_torch = _jax_to_torch(tree_jax)

    # Clip to zero in both frameworks
    clipped_jax, norm_jax = jax_clip.clip_pytree(tree_jax, clip_norm=0.0)
    clipped_torch, norm_torch = clip_pytree(tree_torch, clip_norm=0.0)

    # Convert torch result to JAX for comparison
    clipped_torch_as_jax = _torch_to_jax(clipped_torch)

    # Compare norms
    np.testing.assert_allclose(norm_torch.item(), float(norm_jax), atol=ATOL)

    # All leaves should be zero
    for leaf_jax, leaf_torch in zip(
        jax.tree.leaves(clipped_jax), jax.tree.leaves(clipped_torch_as_jax)
    ):
        np.testing.assert_allclose(np.asarray(leaf_torch), np.asarray(leaf_jax), atol=ATOL)


def test_clip_pytree_matches_jax_inf_clip():
    """Validate clip_pytree with clip_norm=inf (no clipping) matches JAX-Privacy."""
    jax_clip = _import_jax_privacy_clipping()
    tree_jax = _make_pytree_jax()
    tree_torch = _jax_to_torch(tree_jax)

    # No clipping in both frameworks
    clipped_jax, norm_jax = jax_clip.clip_pytree(tree_jax, clip_norm=float("inf"))
    clipped_torch, norm_torch = clip_pytree(tree_torch, clip_norm=float("inf"))

    clipped_torch_as_jax = _torch_to_jax(clipped_torch)

    # Compare norms
    np.testing.assert_allclose(norm_torch.item(), float(norm_jax), atol=ATOL)

    # Should match original tree exactly
    for leaf_jax, leaf_torch in zip(
        jax.tree.leaves(clipped_jax), jax.tree.leaves(clipped_torch_as_jax)
    ):
        np.testing.assert_allclose(np.asarray(leaf_torch), np.asarray(leaf_jax), atol=ATOL)


def test_clip_pytree_matches_jax_above_threshold():
    """Validate clip_pytree clips correctly when norm > clip_norm."""
    jax_clip = _import_jax_privacy_clipping()
    tree_jax = _make_pytree_jax()
    tree_torch = _jax_to_torch(tree_jax)

    clip_norm = 1.0
    clipped_jax, orig_norm_jax = jax_clip.clip_pytree(tree_jax, clip_norm=clip_norm)
    clipped_torch, orig_norm_torch = clip_pytree(tree_torch, clip_norm=clip_norm)

    # Check norms match
    np.testing.assert_allclose(orig_norm_torch.item(), float(orig_norm_jax), atol=ATOL)

    # Check clipped values match
    clipped_torch_as_jax = _torch_to_jax(clipped_torch)
    for leaf_jax, leaf_torch in zip(
        jax.tree.leaves(clipped_jax), jax.tree.leaves(clipped_torch_as_jax)
    ):
        np.testing.assert_allclose(np.asarray(leaf_torch), np.asarray(leaf_jax), atol=ATOL)


def test_clip_pytree_matches_jax_below_threshold():
    """Validate clip_pytree doesn't clip when norm < clip_norm."""
    jax_clip = _import_jax_privacy_clipping()
    tree_jax = _make_pytree_jax()
    tree_torch = _jax_to_torch(tree_jax)

    # Use large clip_norm so no clipping happens
    clip_norm = 100.0
    clipped_jax, norm_jax = jax_clip.clip_pytree(tree_jax, clip_norm=clip_norm)
    clipped_torch, norm_torch = clip_pytree(tree_torch, clip_norm=clip_norm)

    clipped_torch_as_jax = _torch_to_jax(clipped_torch)

    # Compare norms
    np.testing.assert_allclose(norm_torch.item(), float(norm_jax), atol=ATOL)

    # Should be unchanged
    for leaf_jax, leaf_torch in zip(
        jax.tree.leaves(clipped_jax), jax.tree.leaves(clipped_torch_as_jax)
    ):
        np.testing.assert_allclose(np.asarray(leaf_torch), np.asarray(leaf_jax), atol=ATOL)


def test_clip_pytree_matches_jax_rescale_to_unit_norm():
    """Validate clip_pytree with rescale_to_unit_norm=True."""
    jax_clip = _import_jax_privacy_clipping()
    tree_jax = _make_pytree_jax()
    tree_torch = _jax_to_torch(tree_jax)

    clip_norm = 2.0
    clipped_jax, norm_jax = jax_clip.clip_pytree(tree_jax, clip_norm=clip_norm, rescale_to_unit_norm=True)
    clipped_torch, norm_torch = clip_pytree(tree_torch, clip_norm=clip_norm, rescale_to_unit_norm=True)

    clipped_torch_as_jax = _torch_to_jax(clipped_torch)

    # Compare norms
    np.testing.assert_allclose(norm_torch.item(), float(norm_jax), atol=ATOL)

    # Compare clipped values
    for leaf_jax, leaf_torch in zip(
        jax.tree.leaves(clipped_jax), jax.tree.leaves(clipped_torch_as_jax)
    ):
        np.testing.assert_allclose(np.asarray(leaf_torch), np.asarray(leaf_jax), atol=ATOL)


def test_clip_pytree_matches_jax_empty_tree():
    """Validate clip_pytree handles empty trees correctly."""
    jax_clip = _import_jax_privacy_clipping()
    tree_jax = {}
    tree_torch = {}

    clipped_jax, norm_jax = jax_clip.clip_pytree(tree_jax, clip_norm=1.0)
    clipped_torch, norm_torch = clip_pytree(tree_torch, clip_norm=1.0)

    assert clipped_jax == clipped_torch == {}
    assert float(norm_jax) == norm_torch.item() == 0.0


@pytest.mark.parametrize("clip_norm", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_clip_pytree_matches_jax_various_norms(clip_norm):
    """Validate clip_pytree across various clip_norm values."""
    jax_clip = _import_jax_privacy_clipping()
    tree_jax = _make_pytree_jax()
    tree_torch = _jax_to_torch(tree_jax)

    clipped_jax, norm_jax = jax_clip.clip_pytree(tree_jax, clip_norm=clip_norm)
    clipped_torch, norm_torch = clip_pytree(tree_torch, clip_norm=clip_norm)

    clipped_torch_as_jax = _torch_to_jax(clipped_torch)

    # Compare norms
    np.testing.assert_allclose(norm_torch.item(), float(norm_jax), atol=ATOL)

    # Compare clipped values
    for leaf_jax, leaf_torch in zip(
        jax.tree.leaves(clipped_jax), jax.tree.leaves(clipped_torch_as_jax)
    ):
        np.testing.assert_allclose(np.asarray(leaf_torch), np.asarray(leaf_jax), atol=ATOL)


def test_clip_pytree_matches_jax_with_zeros():
    """Validate clip_pytree handles zero tensors correctly."""
    jax_clip = _import_jax_privacy_clipping()
    tree_jax = {"w": jnp.zeros((2, 2)), "b": jnp.zeros(3)}
    tree_torch = _jax_to_torch(tree_jax)

    clipped_jax, norm_jax = jax_clip.clip_pytree(tree_jax, clip_norm=1.0)
    clipped_torch, norm_torch = clip_pytree(tree_torch, clip_norm=1.0)

    clipped_torch_as_jax = _torch_to_jax(clipped_torch)

    # Both norms should be 0
    assert float(norm_jax) == 0.0
    assert norm_torch.item() == 0.0

    # Values should still be zero
    for leaf_jax, leaf_torch in zip(
        jax.tree.leaves(clipped_jax), jax.tree.leaves(clipped_torch_as_jax)
    ):
        np.testing.assert_allclose(np.asarray(leaf_torch), np.asarray(leaf_jax), atol=ATOL)


# ============================================================================
# clip_sum tests (low-level API)
# ============================================================================


def test_clip_sum_matches_jax_basic():
    """Validate clip_sum with grad(loss_fn) matches JAX-Privacy."""
    jax_clip = _import_jax_privacy_clipping()

    # Define loss function in both frameworks
    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    # Create clipped gradient functions
    clip_norm = 1.0
    batch_size = 3.0

    clipped_grad_jax = jax_clip.clip_sum(
        jax.grad(loss_fn_jax), l2_clip_norm=clip_norm, batch_argnums=1, normalize_by=batch_size
    )

    clipped_grad_torch = clip_sum(
        torch.func.grad(loss_fn_torch), l2_clip_norm=clip_norm, batch_argnums=1, normalize_by=batch_size
    )

    # Test data
    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])

    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    # Compute gradients
    grad_jax = clipped_grad_jax(param_jax, data_jax)
    grad_torch = clipped_grad_torch(param_torch, data_torch)

    # Compare
    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)


def test_clip_sum_matches_jax_with_return_norms():
    """Validate clip_sum with return_norms=True."""
    jax_clip = _import_jax_privacy_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    clip_norm = 1.0
    clipped_grad_jax = jax_clip.clip_sum(
        jax.grad(loss_fn_jax), l2_clip_norm=clip_norm, batch_argnums=1, return_norms=True
    )

    clipped_grad_torch = clip_sum(
        torch.func.grad(loss_fn_torch), l2_clip_norm=clip_norm, batch_argnums=1, return_norms=True
    )

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax, norms_jax = clipped_grad_jax(param_jax, data_jax)
    grad_torch, norms_torch = clipped_grad_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)
    np.testing.assert_allclose(norms_torch.detach().numpy(), np.asarray(norms_jax), atol=ATOL)


def test_clip_sum_matches_jax_with_pytree():
    """Validate clip_sum works with PyTree outputs."""
    jax_clip = _import_jax_privacy_clipping()

    # Function that returns a PyTree
    def grad_fn_jax(params, data):
        return jax.tree.map(lambda p: data * p, params)

    def grad_fn_torch(params, data):
        return {k: data * v for k, v in params.items()}

    clip_norm = 2.0
    clipped_jax = jax_clip.clip_sum(grad_fn_jax, l2_clip_norm=clip_norm, batch_argnums=1)
    clipped_torch = clip_sum(grad_fn_torch, l2_clip_norm=clip_norm, batch_argnums=1)

    params_jax = {"w": jnp.array(1.0), "b": jnp.array(0.5)}
    data_jax = jnp.array([1.0, 2.0, 3.0])

    params_torch = {"w": torch.tensor(1.0), "b": torch.tensor(0.5)}
    data_torch = torch.tensor([1.0, 2.0, 3.0])

    result_jax = clipped_jax(params_jax, data_jax)
    result_torch = clipped_torch(params_torch, data_torch)

    # Compare each key
    for key in ["w", "b"]:
        np.testing.assert_allclose(
            result_torch[key].detach().numpy(), np.asarray(result_jax[key]), atol=ATOL
        )


def test_clip_sum_matches_jax_keep_batch_dim_false():
    """Validate clip_sum with keep_batch_dim=False."""
    jax_clip = _import_jax_privacy_clipping()

    # Function expects single example (no batch dim)
    def fn_jax(param, data):
        return param * data

    def fn_torch(param, data):
        return param * data

    clip_norm = 1.0
    clipped_jax = jax_clip.clip_sum(
        fn_jax, l2_clip_norm=clip_norm, batch_argnums=1, keep_batch_dim=False
    )
    clipped_torch = clip_sum(
        fn_torch, l2_clip_norm=clip_norm, batch_argnums=1, keep_batch_dim=False
    )

    param_jax = jnp.array(2.0)
    data_jax = jnp.array([1.0, 2.0, 3.0])

    param_torch = torch.tensor(2.0)
    data_torch = torch.tensor([1.0, 2.0, 3.0])

    result_jax = clipped_jax(param_jax, data_jax)
    result_torch = clipped_torch(param_torch, data_torch)

    np.testing.assert_allclose(result_torch.detach().numpy(), np.asarray(result_jax), atol=ATOL)


def test_clip_sum_matches_jax_l2_norm_bound_property():
    """Validate that clip_sum returns callable with l2_norm_bound property."""
    jax_clip = _import_jax_privacy_clipping()

    def fn_jax(x):
        return x * 2

    def fn_torch(x):
        return x * 2

    clip_norm = 5.0
    clipped_jax = jax_clip.clip_sum(fn_jax, l2_clip_norm=clip_norm, batch_argnums=0)
    clipped_torch = clip_sum(fn_torch, l2_clip_norm=clip_norm, batch_argnums=0)

    # Both should have l2_norm_bound property
    assert hasattr(clipped_jax, "l2_norm_bound")
    assert hasattr(clipped_torch, "l2_norm_bound")

    # Values should match
    assert float(clipped_jax.l2_norm_bound) == clip_norm
    assert float(clipped_torch.l2_norm_bound) == clip_norm
