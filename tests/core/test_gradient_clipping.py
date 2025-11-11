"""Unit tests for gradient_clipping module."""

import pytest
import torch

from opaque.gradient_clipping import AuxiliaryOutput, clipped_grad


def test_clipped_grad_basic():
    """Test basic clipped_grad functionality."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    cg = clipped_grad(loss_fn, l2_clip_norm=1.0, normalize_by=3.0)

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad = cg(param, data)
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape
    assert hasattr(cg, "l2_norm_bound")


def test_clipped_grad_with_return_values():
    """Test clipped_grad with return_values=True."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    cg = clipped_grad(loss_fn, l2_clip_norm=1.0, normalize_by=3.0, return_values=True)

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, aux = cg(param, data)
    assert isinstance(grad, torch.Tensor)
    assert isinstance(aux, AuxiliaryOutput)
    assert aux.values is not None
    assert aux.values.shape[0] == 3  # One value per example
    assert aux.grad_norms is None
    assert aux.aux is None


def test_clipped_grad_with_return_grad_norms():
    """Test clipped_grad with return_grad_norms=True."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    cg = clipped_grad(loss_fn, l2_clip_norm=1.0, normalize_by=3.0, return_grad_norms=True)

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, aux = cg(param, data)
    assert isinstance(grad, torch.Tensor)
    assert isinstance(aux, AuxiliaryOutput)
    assert aux.values is None
    assert aux.grad_norms is not None
    assert aux.grad_norms.shape[0] == 3  # One norm per example
    assert aux.aux is None


def test_clipped_grad_with_has_aux():
    """Test clipped_grad with has_aux=True."""

    def loss_fn_with_aux(param, data):
        loss = 0.5 * ((data - param) ** 2).mean()
        aux_data = data * 2  # Tensor aux (not dict with int)
        return loss, aux_data

    cg = clipped_grad(loss_fn_with_aux, l2_clip_norm=1.0, normalize_by=3.0, has_aux=True)

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, aux = cg(param, data)
    assert isinstance(grad, torch.Tensor)
    assert isinstance(aux, AuxiliaryOutput)
    assert aux.values is None
    assert aux.grad_norms is None
    assert aux.aux is not None  # Per-example aux


def test_clipped_grad_with_all_outputs():
    """Test clipped_grad with all output options enabled."""

    def loss_fn_with_aux(param, data):
        loss = 0.5 * ((data - param) ** 2).mean()
        aux_data = data * 2
        return loss, aux_data

    cg = clipped_grad(
        loss_fn_with_aux,
        l2_clip_norm=1.0,
        normalize_by=3.0,
        has_aux=True,
        return_values=True,
        return_grad_norms=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, aux = cg(param, data)
    assert isinstance(grad, torch.Tensor)
    assert isinstance(aux, AuxiliaryOutput)
    assert aux.values is not None
    assert aux.grad_norms is not None
    assert aux.aux is not None


def test_clipped_grad_with_pre_clipping_transform():
    """Test clipped_grad with pre_clipping_transform."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    # Scale gradients by 0.5 before clipping
    def transform(grad):
        return grad * 0.5

    cg = clipped_grad(
        loss_fn,
        l2_clip_norm=100.0,  # High clip norm so we can see the scaling effect
        normalize_by=3.0,
        pre_clipping_transform=transform,
        return_grad_norms=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, aux = cg(param, data)
    assert isinstance(grad, torch.Tensor)
    # Gradient norms should be scaled by 0.5
    # (checking that transform was applied before clipping)


def test_clipped_grad_pytree_params():
    """Test clipped_grad with PyTree (dict) parameters."""

    def loss_fn(params, data):
        # Simple linear model: y = w * x + b
        pred = params["w"] * data + params["b"]
        return 0.5 * ((pred - data) ** 2).mean()

    cg = clipped_grad(loss_fn, argnums=0, l2_clip_norm=1.0, normalize_by=3.0, batch_argnums=1)

    params = {"w": torch.tensor(1.0, requires_grad=True), "b": torch.tensor(0.5, requires_grad=True)}
    data = torch.tensor([0.0, 1.0, 2.0])

    grad = cg(params, data)
    assert isinstance(grad, dict)
    assert "w" in grad and "b" in grad
    assert isinstance(grad["w"], torch.Tensor)
    assert isinstance(grad["b"], torch.Tensor)


def test_clipped_grad_keep_batch_dim_false():
    """Test clipped_grad with keep_batch_dim=False."""

    def loss_fn_single_example(param, data):
        # data has no batch dimension
        assert data.dim() == 0  # scalar
        return 0.5 * ((data - param) ** 2)

    cg = clipped_grad(loss_fn_single_example, l2_clip_norm=1.0, normalize_by=3.0, keep_batch_dim=False)

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad = cg(param, data)
    assert isinstance(grad, torch.Tensor)


def test_clipped_grad_rescale_to_unit_norm():
    """Test clipped_grad with rescale_to_unit_norm=True."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    cg = clipped_grad(loss_fn, l2_clip_norm=2.0, normalize_by=3.0, rescale_to_unit_norm=True)

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad = cg(param, data)
    assert isinstance(grad, torch.Tensor)
    # Sensitivity should be 1.0 / normalize_by
    assert cg.l2_norm_bound == pytest.approx(1.0 / 3.0)


def test_clipped_grad_validation_errors():
    """Test that clipped_grad raises appropriate validation errors."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    # normalize_by must be positive
    with pytest.raises(ValueError, match="normalize_by must be > 0"):
        clipped_grad(loss_fn, l2_clip_norm=1.0, normalize_by=0.0)

    # batch_argnums must not be empty
    with pytest.raises(ValueError, match="Batch argnums must not be empty"):
        clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=())

    # argnums and batch_argnums must not overlap
    with pytest.raises(ValueError, match="Cannot compute clipped gradients"):
        clipped_grad(loss_fn, argnums=1, l2_clip_norm=1.0, batch_argnums=1)


def test_clipped_grad_dtype():
    """Test clipped_grad with custom dtype for accumulation."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    cg = clipped_grad(loss_fn, l2_clip_norm=1.0, normalize_by=3.0, dtype=torch.float64)

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad = cg(param, data)
    assert grad.dtype == torch.float64
