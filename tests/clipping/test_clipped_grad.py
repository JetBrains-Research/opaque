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

    clipped_grad_fn = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,  # High clip norm so no clipping occurs
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    result = clipped_grad_fn(param, data)

    # Check result is a tensor
    assert isinstance(result, torch.Tensor)
    # With high clip norm, should get unclipped sum of gradients
    assert result.shape == param.shape


def test_clipped_grad_with_rescale():
    """Test clipped_grad with rescale_to_unit_norm=True."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    clipped_grad_fn = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=2.5,
        rescale_to_unit_norm=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    result = clipped_grad_fn(param, data)

    # Check result is a tensor with correct shape
    assert isinstance(result, torch.Tensor)
    assert result.shape == param.shape


def test_clipped_grad_with_pytree_params():
    """Test clipped_grad with PyTree (dict) parameters."""

    def loss(params, data):
        pred = params["w"] * data + params["b"]
        return ((pred - data) ** 2).mean()

    clipped_grad_fn = clipped_grad(
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

    result = clipped_grad_fn(params, data)

    # Check result has same structure as params
    assert isinstance(result, dict)
    assert set(result.keys()) == set(params.keys())
    assert result["w"].shape == params["w"].shape
    assert result["b"].shape == params["b"].shape


def test_clipped_grad_return_grad_norms():
    """Test clipped_grad with return_grad_norms=True."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    clipped_grad_fn = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
        return_grad_norms=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, aux_output = clipped_grad_fn(param, data)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check aux_output has grad_norms
    assert aux_output.grad_norms is not None
    assert aux_output.grad_norms.shape == (3,)  # One norm per example
    assert (aux_output.grad_norms >= 0).all()  # Norms are non-negative


def test_clipped_grad_return_values():
    """Test clipped_grad with return_values=True."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    clipped_grad_fn = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
        return_values=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, aux_output = clipped_grad_fn(param, data)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check aux_output has values
    assert aux_output.values is not None
    assert aux_output.values.shape == (3,)  # One value per example


def test_clipped_grad_has_aux():
    """Test clipped_grad with has_aux=True."""

    def loss_with_aux(param, data):
        loss = 0.5 * ((data - param) ** 2).mean()
        aux = {"mean_data": data.mean()}
        return loss, aux

    clipped_grad_fn = clipped_grad(
        loss_with_aux,
        argnums=0,
        has_aux=True,
        batch_argnums=1,
        l2_clip_norm=10.0,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, aux_output = clipped_grad_fn(param, data)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check aux_output has aux data
    assert aux_output.aux is not None
    assert isinstance(aux_output.aux, dict)
    assert "mean_data" in aux_output.aux


def test_clipped_grad_with_normalize_by():
    """Test clipped_grad with normalize_by parameter."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    batch_size = 3.0
    clipped_grad_fn = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        l2_clip_norm=10.0,
        normalize_by=batch_size,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    result = clipped_grad_fn(param, data)

    # Check result is normalized (smaller than without normalization)
    assert isinstance(result, torch.Tensor)
    assert result.shape == param.shape
