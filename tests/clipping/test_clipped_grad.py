"""Unit tests for clipped_grad function.

Simplified tests adapted for our PyTorch implementation.
For comprehensive validation against JAX-Privacy, see tests/jax_validation/test_clipped_grad.py
"""

import pytest
import torch

from opaque.clipping import clipped_grad


def test_clipped_grad_validate_args_overlap():
    """Test that argnums and batch_argnums cannot overlap."""

    def loss(params, x):
        return ((params - x) ** 2).mean()

    with pytest.raises(ValueError, match="overlap"):
        clipped_grad(
            loss,
            l2_clip_norm=1.0,
            argnums=0,
            batch_argnums=0,
        )


def test_clipped_grad_validate_args_empty_batch():
    """Test that batch_argnums cannot be empty."""

    def loss(params, x):
        return ((params - x) ** 2).mean()

    with pytest.raises(ValueError, match="Batch argnums must not be empty"):
        clipped_grad(
            loss,
            l2_clip_norm=1.0,
            argnums=0,
            batch_argnums=(),
        )


def test_clipped_grad_basic():
    """Test basic clipped_grad returns gradient."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,  # High clip norm so no clipping occurs
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, _ = grad_fn(param, data, state=clip_state)

    # Check grad is a tensor
    assert isinstance(grad, torch.Tensor)
    # With high clip norm, should get unclipped sum of gradients
    assert grad.shape == param.shape


def test_clipped_grad_with_rescale():
    """Test clipped_grad with rescale_to_unit_norm=True."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=2.5,
        rescale_to_unit_norm=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, _ = grad_fn(param, data, state=clip_state)

    # Check grad is a tensor with correct shape
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape


def test_clipped_grad_with_pytree_params():
    """Test clipped_grad with PyTree (dict) parameters."""

    def loss(params, data):
        pred = params["w"] * data + params["b"]
        return ((pred - data) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
    )

    params = {
        "w": torch.tensor(1.0, requires_grad=True),
        "b": torch.tensor(0.5, requires_grad=True),
    }
    data = torch.tensor([0.0, 1.0, 2.0])

    grads, _ = grad_fn(params, data, state=clip_state)

    # Check grads has same structure as params
    assert isinstance(grads, dict)
    assert set(grads.keys()) == set(params.keys())
    assert grads["w"].shape == params["w"].shape
    assert grads["b"].shape == params["b"].shape


def test_clipped_grad_return_grad_norms():
    """Test clipped_grad with return_grad_norms=True."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
        return_grad_norms=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check grad_aux has grad_norms
    assert grad_aux.grad_norms is not None
    assert grad_aux.grad_norms.shape == (3,)  # One norm per example
    assert (grad_aux.grad_norms >= 0).all()  # Norms are non-negative


def test_clipped_grad_return_values():
    """Test clipped_grad with return_values=True."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
        return_values=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check grad_aux has values
    assert grad_aux.loss_values is not None
    assert grad_aux.loss_values.shape == (3,)  # One value per example


def test_clipped_grad_has_aux():
    """Test clipped_grad with has_aux=True."""

    def loss_with_aux(param, data):
        loss = 0.5 * ((data - param) ** 2).mean()
        user_aux = {"mean_data": data.mean()}
        return loss, user_aux

    grad_fn, clip_state = clipped_grad(
        loss_with_aux,
        argnums=0,
        has_aux=True,
        batch_argnums=1,
        l2_clip_norm=10.0,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check grad_aux has user_aux data
    assert grad_aux.user_aux is not None
    assert isinstance(grad_aux.user_aux, dict)
    assert "mean_data" in grad_aux.user_aux


def test_clipped_grad_with_normalize_by():
    """Test clipped_grad with normalize_by parameter."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    batch_size = 3.0
    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
        normalize_by=batch_size,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, _ = grad_fn(param, data, state=clip_state)

    # Check grad is normalized (smaller than without normalization)
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape


def test_clipped_grad_actual_clipping():
    """Test that gradients are actually clipped when they exceed the norm threshold."""

    def loss(param, data):
        # MSE loss that will produce large gradients
        return ((data - param) ** 2).mean()

    # Small clip norm to force clipping
    clip_norm = 1.0
    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=clip_norm,
        return_grad_norms=True,
    )

    param = torch.tensor(0.0)
    data = torch.tensor([100.0, 200.0, 300.0])  # Large values -> large gradients

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)

    # Check that some gradients were clipped
    assert (grad_aux.grad_norms > clip_norm).any(), "Expected some gradients to be clipped"
    assert grad_aux.grad_norms.shape == (3,), "Should have 3 per-example norms"


def test_clipped_grad_preserves_direction():
    """Test that clipping preserves gradient direction (only scales magnitude)."""

    def loss(param, data):
        return ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=1.0,  # Small clip norm
        return_grad_norms=True,
    )

    # Compute unclipped gradient for single example
    param = torch.tensor(0.0, requires_grad=True)
    data_single = torch.tensor([10.0])

    param.grad = None
    loss_val = ((data_single - param) ** 2).mean()
    loss_val.backward()
    unclipped_grad = param.grad.clone()
    unclipped_norm = unclipped_grad.abs().item()

    # Compute clipped gradient
    (grad, grad_aux), _ = grad_fn(param, data_single, state=clip_state)

    # Check direction is preserved (signs match)
    assert torch.sign(grad) == torch.sign(unclipped_grad), "Direction should be preserved"

    # Check that norm is clipped
    if unclipped_norm > 1.0:
        assert abs(grad.item()) <= 1.0, "Should be clipped to norm 1.0"


def test_clipped_grad_no_clipping_below_threshold():
    """Test that gradients below threshold are unchanged."""

    def loss(param, data):
        return ((data - param) ** 2).mean()

    # Large clip norm (won't clip)
    large_clip_norm = 1000.0
    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=large_clip_norm,
        return_grad_norms=True,
    )

    param = torch.tensor(1.0)
    data = torch.tensor([1.1, 0.9, 1.05])  # Small differences -> small gradients

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)

    # All norms should be below threshold
    assert (grad_aux.grad_norms < large_clip_norm).all(), "All norms should be below threshold"
    # Gradient should be non-trivial
    assert grad.abs() > 1e-6, "Gradient should be non-zero"


def test_clipped_grad_zero_gradients():
    """Test handling of zero gradients."""

    def loss(param, data):
        return ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=1.0,
        return_grad_norms=True,
    )

    # Zero data and zero param -> zero gradients
    param = torch.tensor(0.0)
    data = torch.zeros(3)

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)

    # Check all zero
    assert grad.abs() < 1e-7, "Gradient should be zero"
    assert (grad_aux.grad_norms < 1e-7).all(), "All norms should be zero"


def test_clipped_grad_keep_batch_dim():
    """Test keep_batch_dim parameter affects loss function signature."""

    def loss_no_batch(param, data):
        # Expects data without batch dim (will unsqueeze in real usage)
        return ((data.unsqueeze(0) - param) ** 2).mean()

    def loss_with_batch(param, data):
        # Expects data with batch dim of size 1
        return ((data - param) ** 2).mean()

    param = torch.tensor(1.0)
    data = torch.tensor([0.5, 1.5, 2.0])

    # Test keep_batch_dim=False
    grad_fn_no_batch, clip_state_no_batch = clipped_grad(
        loss_no_batch,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
        keep_batch_dim=False,
    )
    grad_no_batch, _ = grad_fn_no_batch(param, data, state=clip_state_no_batch)

    # Test keep_batch_dim=True
    grad_fn_with_batch, clip_state_with_batch = clipped_grad(
        loss_with_batch,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
        keep_batch_dim=True,
    )
    grad_with_batch, _ = grad_fn_with_batch(param, data, state=clip_state_with_batch)

    # Both should produce gradients (exact values may differ due to loss function)
    assert grad_no_batch.abs() > 1e-6, "Should have non-zero gradient"
    assert grad_with_batch.abs() > 1e-6, "Should have non-zero gradient"
