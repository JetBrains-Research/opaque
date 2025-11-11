"""JAX validation tests for gradient_clipping module.

These tests validate that our high-level gradient clipping API produces numerically
equivalent results to JAX-Privacy's gradient_clipping module. This is the main
user-facing API for DP-SGD gradient clipping.

Module under test: opaque.gradient_clipping
Functions tested:
    - clipped_grad() - High-level API for per-example gradient clipping

Reference: jax_privacy.experimental.gradient_clipping

Run:
    uv run --group jax-validation pytest -m jax_validation tests/jax_validation/test_gradient_clipping.py -v
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from opaque.gradient_clipping import clipped_grad

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.jax_validation]

ATOL = 1e-5  # Standard tolerance for JAX-PyTorch numerical comparison


def _import_jax_privacy_gradient_clipping():
    """Import JAX-Privacy gradient_clipping module with path fallbacks."""
    import importlib

    candidates = [
        "jax_privacy.experimental.gradient_clipping",
        "jax_privacy.src.experimental.gradient_clipping",
    ]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    pytest.skip("Could not import JAX-Privacy gradient_clipping (check ../jax_privacy)")


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


# ============================================================================
# clipped_grad tests (high-level API)
# ============================================================================


def test_clipped_grad_matches_jax_basic():
    """Validate clipped_grad matches JAX-Privacy gradient_clipping."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    clip_norm = 1.0
    batch_size = 3.0

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=clip_norm, normalize_by=batch_size)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=clip_norm, normalize_by=batch_size)

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax = cg_jax(param_jax, data_jax)
    grad_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)


def test_clipped_grad_matches_jax_scalar_params():
    """Validate clipped_grad with scalar parameters."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=1.0)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=1.0)

    param_jax = jnp.array(5.0)
    data_jax = jnp.array([1.0, 2.0, 3.0])
    param_torch = torch.tensor(5.0, requires_grad=True)
    data_torch = torch.tensor([1.0, 2.0, 3.0])

    grad_jax = cg_jax(param_jax, data_jax)
    grad_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)


def test_clipped_grad_matches_jax_with_pytree_params():
    """Validate clipped_grad with PyTree parameters."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(params, data):
        pred = params["w"] * data + params["b"]
        return 0.5 * jnp.mean((pred - data) ** 2)

    def loss_fn_torch(params, data):
        pred = params["w"] * data + params["b"]
        return 0.5 * torch.mean((pred - data) ** 2)

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, argnums=0, l2_clip_norm=1.0, batch_argnums=1)
    cg_torch = clipped_grad(loss_fn_torch, argnums=0, l2_clip_norm=1.0, batch_argnums=1)

    params_jax = {"w": jnp.array(1.0), "b": jnp.array(0.5)}
    data_jax = jnp.array([0.0, 1.0, 2.0])

    params_torch = {"w": torch.tensor(1.0, requires_grad=True), "b": torch.tensor(0.5, requires_grad=True)}
    data_torch = torch.tensor([0.0, 1.0, 2.0])

    grad_jax = cg_jax(params_jax, data_jax)
    grad_torch = cg_torch(params_torch, data_torch)

    # Compare leaf by leaf
    for key in ["w", "b"]:
        np.testing.assert_allclose(grad_torch[key].detach().numpy(), np.asarray(grad_jax[key]), atol=ATOL)


def test_clipped_grad_matches_jax_nested_pytree():
    """Validate clipped_grad with deeply nested PyTree parameters."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(params, data):
        pred = params["layer1"]["w"] * data + params["layer1"]["b"]
        pred = params["layer2"]["w"] * pred + params["layer2"]["b"]
        return jnp.mean(pred**2)

    def loss_fn_torch(params, data):
        pred = params["layer1"]["w"] * data + params["layer1"]["b"]
        pred = params["layer2"]["w"] * pred + params["layer2"]["b"]
        return torch.mean(pred**2)

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, argnums=0, l2_clip_norm=2.0, batch_argnums=1)
    cg_torch = clipped_grad(loss_fn_torch, argnums=0, l2_clip_norm=2.0, batch_argnums=1)

    params_jax = {
        "layer1": {"w": jnp.array(1.0), "b": jnp.array(0.5)},
        "layer2": {"w": jnp.array(2.0), "b": jnp.array(-0.5)},
    }
    data_jax = jnp.array([1.0, 2.0, 3.0])

    params_torch = {
        "layer1": {"w": torch.tensor(1.0, requires_grad=True), "b": torch.tensor(0.5, requires_grad=True)},
        "layer2": {"w": torch.tensor(2.0, requires_grad=True), "b": torch.tensor(-0.5, requires_grad=True)},
    }
    data_torch = torch.tensor([1.0, 2.0, 3.0])

    grad_jax = cg_jax(params_jax, data_jax)
    grad_torch = cg_torch(params_torch, data_torch)

    # Compare nested structure
    for layer in ["layer1", "layer2"]:
        for key in ["w", "b"]:
            np.testing.assert_allclose(
                grad_torch[layer][key].detach().numpy(),
                np.asarray(grad_jax[layer][key]),
                atol=ATOL,
            )


def test_clipped_grad_matches_jax_with_return_grad_norms():
    """Validate clipped_grad with return_grad_norms=True."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=1.0, return_grad_norms=True)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=1.0, return_grad_norms=True)

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax, aux_jax = cg_jax(param_jax, data_jax)
    grad_torch, aux_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)
    np.testing.assert_allclose(
        aux_torch.grad_norms.detach().numpy(), np.asarray(aux_jax.grad_norms), atol=ATOL
    )


def test_clipped_grad_matches_jax_with_return_values():
    """Validate clipped_grad with return_values=True."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=1.0, return_values=True)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=1.0, return_values=True)

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax, aux_jax = cg_jax(param_jax, data_jax)
    grad_torch, aux_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)
    np.testing.assert_allclose(
        aux_torch.values.detach().numpy(), np.asarray(aux_jax.values), atol=ATOL
    )


def test_clipped_grad_matches_jax_with_all_auxiliary_outputs():
    """Validate clipped_grad with all auxiliary outputs enabled."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    cg_jax = jax_gc.clipped_grad(
        loss_fn_jax, l2_clip_norm=1.0, return_values=True, return_grad_norms=True
    )
    cg_torch = clipped_grad(
        loss_fn_torch, l2_clip_norm=1.0, return_values=True, return_grad_norms=True
    )

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax, aux_jax = cg_jax(param_jax, data_jax)
    grad_torch, aux_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)
    np.testing.assert_allclose(
        aux_torch.values.detach().numpy(), np.asarray(aux_jax.values), atol=ATOL
    )
    np.testing.assert_allclose(
        aux_torch.grad_norms.detach().numpy(), np.asarray(aux_jax.grad_norms), atol=ATOL
    )


def test_clipped_grad_matches_jax_rescale_to_unit_norm():
    """Validate clipped_grad with rescale_to_unit_norm=True."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=2.0, rescale_to_unit_norm=True)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=2.0, rescale_to_unit_norm=True)

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax = cg_jax(param_jax, data_jax)
    grad_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)


def test_clipped_grad_matches_jax_keep_batch_dim_false():
    """Validate clipped_grad with keep_batch_dim=False."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    # Loss function expects single example (no batch dimension)
    def loss_fn_jax(param, data):
        return 0.5 * (data - param) ** 2

    def loss_fn_torch(param, data):
        return 0.5 * (data - param) ** 2

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=1.0, keep_batch_dim=False)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=1.0, keep_batch_dim=False)

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax = cg_jax(param_jax, data_jax)
    grad_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)


@pytest.mark.parametrize("clip_norm", [0.5, 1.0, 2.0, 5.0])
def test_clipped_grad_matches_jax_various_clip_norms(clip_norm):
    """Validate clipped_grad across various clip_norm values."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=clip_norm)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=clip_norm)

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax = cg_jax(param_jax, data_jax)
    grad_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)


def test_clipped_grad_matches_jax_has_aux():
    """Validate clipped_grad with has_aux=True for user auxiliary outputs."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        loss = 0.5 * jnp.mean((data - param) ** 2)
        aux = {"debug_info": jnp.array(42.0)}
        return loss, aux

    def loss_fn_torch(param, data):
        loss = 0.5 * torch.mean((data - param) ** 2)
        aux = {"debug_info": torch.tensor(42.0)}
        return loss, aux

    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=1.0, has_aux=True)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=1.0, has_aux=True)

    param_jax = jnp.array(3.0)
    data_jax = jnp.array([0.0, 7.0, -2.0])
    param_torch = torch.tensor(3.0, requires_grad=True)
    data_torch = torch.tensor([0.0, 7.0, -2.0])

    grad_jax, aux_output_jax = cg_jax(param_jax, data_jax)
    grad_torch, aux_output_torch = cg_torch(param_torch, data_torch)

    np.testing.assert_allclose(grad_torch.detach().numpy(), np.asarray(grad_jax), atol=ATOL)
    # Check user auxiliary output is in the .aux field
    np.testing.assert_allclose(
        aux_output_torch.aux["debug_info"].detach().numpy(),
        np.asarray(aux_output_jax.aux["debug_info"]),
        atol=ATOL,
    )


def test_clipped_grad_matches_jax_l2_norm_bound_property():
    """Validate that clipped_grad returns callable with l2_norm_bound property."""
    jax_gc = _import_jax_privacy_gradient_clipping()

    def loss_fn_jax(param, data):
        return 0.5 * jnp.mean((data - param) ** 2)

    def loss_fn_torch(param, data):
        return 0.5 * torch.mean((data - param) ** 2)

    clip_norm = 5.0
    cg_jax = jax_gc.clipped_grad(loss_fn_jax, l2_clip_norm=clip_norm)
    cg_torch = clipped_grad(loss_fn_torch, l2_clip_norm=clip_norm)

    # Both should have l2_norm_bound property
    assert hasattr(cg_jax, "l2_norm_bound")
    assert hasattr(cg_torch, "l2_norm_bound")

    # Values should match
    assert float(cg_jax.l2_norm_bound) == clip_norm
    assert float(cg_torch.l2_norm_bound) == clip_norm
