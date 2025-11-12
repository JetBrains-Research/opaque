"""Tests for functional_utils module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap

from opaque.functional_utils import make_functional


def test_make_functional_basic():
    """Test basic functionality of make_functional."""
    # Create a simple linear model
    model = nn.Linear(10, 1)

    # Convert to functional form
    fmodel, params = make_functional(model)

    # Check params is a tuple
    assert isinstance(params, tuple)
    assert len(params) == 2  # weight and bias

    # Check params have correct shapes
    assert params[0].shape == (1, 10)  # weight
    assert params[1].shape == (1,)  # bias

    # Test forward pass
    x = torch.randn(5, 10)
    output = fmodel(params, x)

    assert output.shape == (5, 1)
    assert output.dtype == torch.float32


def test_make_functional_matches_original():
    """Test that functional model matches original model output."""
    # Create model
    model = nn.Linear(10, 5)

    # Convert to functional
    fmodel, params = make_functional(model)

    # Test data
    x = torch.randn(3, 10)

    # Compare outputs
    with torch.no_grad():
        output_original = model(x)
        output_functional = fmodel(params, x)

    assert torch.allclose(output_original, output_functional, atol=1e-6)


def test_make_functional_with_mlp():
    """Test make_functional with a multi-layer network."""
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x.squeeze(-1)

    model = SimpleMLP()
    fmodel, params = make_functional(model)

    # Should have 6 parameters (3 weights + 3 biases)
    assert len(params) == 6

    # Test forward pass
    x = torch.randn(4, 10)
    output = fmodel(params, x)
    assert output.shape == (4,)


def test_make_functional_with_grad():
    """Test that make_functional works with torch.func.grad."""
    # Simple model
    model = nn.Linear(5, 1)
    fmodel, params = make_functional(model)

    # Define loss function
    def loss_fn(p, x, y):
        pred = fmodel(p, x)
        return ((pred - y) ** 2).mean()

    # Test data
    x = torch.randn(3, 5)
    y = torch.randn(3, 1)

    # Compute gradients
    grads = grad(loss_fn)(params, x, y)

    # Check gradients
    assert isinstance(grads, tuple)
    assert len(grads) == 2
    assert grads[0].shape == params[0].shape  # weight grad
    assert grads[1].shape == params[1].shape  # bias grad


def test_make_functional_with_vmap():
    """Test that make_functional works with torch.func.vmap."""
    # Simple model
    model = nn.Linear(5, 1)
    fmodel, params = make_functional(model)

    # Per-example loss function
    def loss_single(p, x, y):
        pred = fmodel(p, x.unsqueeze(0))
        return ((pred - y) ** 2).mean()

    # Test data (batch)
    x_batch = torch.randn(8, 5)
    y_batch = torch.randn(8, 1)

    # Compute per-example gradients
    per_example_grads = vmap(grad(loss_single), in_dims=(None, 0, 0))(
        params, x_batch, y_batch
    )

    # Check structure: tuple of (batch_size, *param_shape)
    assert isinstance(per_example_grads, tuple)
    assert len(per_example_grads) == 2
    assert per_example_grads[0].shape == (8, 1, 5)  # weight grads
    assert per_example_grads[1].shape == (8, 1)  # bias grads


def test_make_functional_disable_autograd_tracking():
    """Test disable_autograd_tracking parameter."""
    model = nn.Linear(5, 1)

    # Without disable_autograd_tracking
    fmodel1, params1 = make_functional(model, disable_autograd_tracking=False)
    assert all(p.requires_grad for p in params1)

    # With disable_autograd_tracking
    fmodel2, params2 = make_functional(model, disable_autograd_tracking=True)
    assert not any(p.requires_grad for p in params2)


def test_make_functional_preserves_device():
    """Test that make_functional preserves device."""
    if not torch.cuda.is_available():
        return  # Skip if no CUDA

    model = nn.Linear(5, 1).cuda()
    fmodel, params = make_functional(model)

    # Params should be on cuda
    assert all(p.device.type == "cuda" for p in params)

    # Forward pass should work on cuda
    x = torch.randn(3, 5, device="cuda")
    output = fmodel(params, x)
    assert output.device.type == "cuda"


def test_make_functional_with_kwargs():
    """Test that fmodel works with keyword arguments."""
    class ModelWithKwargs(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 1)

        def forward(self, x, scale=1.0):
            return self.fc(x) * scale

    model = ModelWithKwargs()
    fmodel, params = make_functional(model)

    x = torch.randn(3, 5)

    # Test with default kwargs
    output1 = fmodel(params, x)
    assert output1.shape == (3, 1)

    # Test with custom kwargs
    output2 = fmodel(params, x, scale=2.0)
    assert torch.allclose(output2, output1 * 2.0, atol=1e-6)


def test_make_functional_parameter_independence():
    """Test that modifying params doesn't affect original model."""
    model = nn.Linear(5, 1)
    original_weight = model.weight.data.clone()

    fmodel, params = make_functional(model)

    # Modify params
    params = tuple(p * 2 for p in params)

    # Original model should be unchanged
    assert torch.allclose(model.weight.data, original_weight)


def test_make_functional_with_sequential():
    """Test make_functional with nn.Sequential."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )

    fmodel, params = make_functional(model)

    # Should have 4 parameters (2 weights + 2 biases)
    assert len(params) == 4

    # Test forward pass
    x = torch.randn(4, 10)
    output = fmodel(params, x)
    assert output.shape == (4, 5)
