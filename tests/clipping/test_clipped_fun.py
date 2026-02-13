"""Unit tests for clipping module."""

import pytest
import torch
from torch.func import grad

from opaque.clipping import clip_pytree, clipped_fun


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request):
    """Parametrize tests over all available devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device(request.param)


# ============================================================================
# clip_pytree tests
# ============================================================================


def test_clip_zero_returns_zeros(device):
    """clip_norm=0 should return all zeros."""
    pytree = {"w": torch.tensor([3.0, 4.0], device=device)}
    clipped, clip_aux = clip_pytree(pytree, clip_norm=0.0)
    assert torch.allclose(clipped["w"], torch.zeros_like(pytree["w"]))
    assert clip_aux.norm.item() == pytest.approx(5.0)


def test_clip_inf_is_passthrough(device):
    """clip_norm=inf should not modify the pytree."""
    pytree = {"w": torch.tensor([3.0, 4.0], device=device)}
    clipped, _ = clip_pytree(pytree, clip_norm=float("inf"))
    assert torch.allclose(clipped["w"], pytree["w"])


def test_clip_rescales_to_threshold_when_above(device):
    """When norm > clip_norm, should scale down to clip_norm."""
    pytree = {"w": torch.tensor([3.0, 4.0], device=device)}  # norm=5
    clipped, _ = clip_pytree(pytree, clip_norm=1.0)
    from opaque.utils.pytree import global_norm

    assert global_norm(clipped).item() == pytest.approx(1.0)


def test_clip_preserves_structure_and_device(device):
    """Clipping should preserve PyTree structure and tensor devices."""
    pytree = {
        "layer1": {
            "w": torch.tensor([1.0, 2.0], device=device),
            "b": torch.tensor([0.5], device=device),
        },
        "layer2": {"w": torch.tensor([3.0, 4.0], device=device)},
    }
    clipped, _ = clip_pytree(pytree, clip_norm=1.0)
    assert set(clipped.keys()) == set(pytree.keys())
    assert clipped["layer1"]["w"].device.type == device.type


def test_clip_no_change_when_below_threshold(device):
    """When norm < clip_norm, pytree should be unchanged."""
    pytree = {"w": torch.tensor([0.3, 0.4], device=device)}  # norm=0.5
    clipped, _ = clip_pytree(pytree, clip_norm=1.0)
    assert torch.allclose(clipped["w"], pytree["w"])


def test_clip_nan_safe_replaces_nans_before_scaling(device):
    """nan_safe=True should replace NaNs/Infs with zeros."""
    pytree = {"w": torch.tensor([float("nan"), float("inf"), 1.0], device=device)}
    clipped, _ = clip_pytree(pytree, clip_norm=1.0, nan_safe=True)
    # After nan_to_num, only the 1.0 remains, so norm=1.0, no scaling needed
    assert torch.allclose(clipped["w"], torch.tensor([0.0, 0.0, 1.0], device=device))


def test_clip_handles_empty_tree(device):
    """Empty pytree should return empty pytree with norm=0."""
    pytree = {}
    clipped, clip_aux = clip_pytree(pytree, clip_norm=1.0)
    assert clipped == {}
    assert clip_aux.norm.item() == 0.0


# ============================================================================
# clipped_fun tests (with gradients)
# ============================================================================


def test_clipped_fun_scalar_basic():
    """Basic test: clip_sum with scalar param and batch data."""

    def loss_fn(param, data):
        # With keep_batch_dim=True (default), data has shape (1,)
        return 0.5 * ((data - param) ** 2).mean()

    clipped_grad_fn, clip_state = clipped_fun(
        grad(loss_fn), batch_argnums=1, l2_clip_norm=1.0, normalize_by=3.0
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    clipped_grad, _ = clipped_grad_fn(param, data, state=clip_state)
    assert isinstance(clipped_grad, torch.Tensor)
    assert clipped_grad.shape == param.shape


def test_clipped_fun_return_norms():
    """Test return_norms returns per-example norms before clipping."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    clipped_grad_fn, clip_state = clipped_fun(
        grad(loss_fn), batch_argnums=1, l2_clip_norm=1.0, normalize_by=3.0, return_norms=True
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    (clipped_grad, (user_aux, clip_aux)), _ = clipped_grad_fn(param, data, state=clip_state)
    assert isinstance(clipped_grad, torch.Tensor)
    assert user_aux == ()  # user_aux should be empty tuple for has_aux=False
    # clip_aux is ClipPytreeAux with .norm attribute containing per-example norms
    assert isinstance(clip_aux.norm, torch.Tensor)
    assert clip_aux.norm.shape == (3,)  # One norm per example
    assert all(clip_aux.norm >= 0)


def test_clipped_fun_keep_batch_dim_true():
    """Test keep_batch_dim=True passes size-1 batch dim to loss."""

    def loss_fn(param, data):
        # Expect data to have shape (1,)
        assert data.shape == (1,)
        return 0.5 * ((data - param) ** 2).mean()

    clipped_grad_fn, clip_state = clipped_fun(
        grad(loss_fn), batch_argnums=1, l2_clip_norm=1.0, normalize_by=3.0, keep_batch_dim=True
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    clipped_grad, _ = clipped_grad_fn(param, data, state=clip_state)
    assert isinstance(clipped_grad, torch.Tensor)


def test_clipped_fun_has_aux_false():
    """Test has_aux=False with function returning only value."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    clipped_grad_fn, clip_state = clipped_fun(grad(loss_fn), batch_argnums=1, l2_clip_norm=1.0, has_aux=False)

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    clipped_grad, _ = clipped_grad_fn(param, data, state=clip_state)
    assert isinstance(clipped_grad, torch.Tensor)
    # Should only return the clipped grad, no aux


def test_clipped_fun_has_aux_true():
    """Test has_aux=True with function returning (value, user_aux)."""

    # Test on a simple function (not grad) that returns aux
    def fn_with_aux(x, data):
        # data has shape (1,) due to keep_batch_dim
        value = x + data  # Return a tensor
        user_aux = data * 2  # Some auxiliary value (tensor)
        return value, user_aux

    clipped_fn, clip_state = clipped_fun(fn_with_aux, batch_argnums=1, l2_clip_norm=1.0, has_aux=True)

    x = torch.tensor([1.0])
    data = torch.tensor([0.0, 1.0, 2.0])

    (clipped_value, user_aux_list), _ = clipped_fn(x, data, state=clip_state)
    assert isinstance(clipped_value, torch.Tensor)
    # user_aux_list contains per-example aux values


def test_clipped_fun_has_aux_with_return_norms():
    """Test has_aux=True and return_norms=True together."""

    def fn_with_aux(x, data):
        value = x + data
        user_aux = data
        return value, user_aux

    clipped_fn, clip_state = clipped_fun(
        fn_with_aux, batch_argnums=1, l2_clip_norm=1.0, has_aux=True, return_norms=True
    )

    x = torch.tensor([1.0])
    data = torch.tensor([0.0, 1.0, 2.0])

    (clipped_value, (user_aux_list, clip_aux)), _ = clipped_fn(x, data, state=clip_state)
    assert isinstance(clipped_value, torch.Tensor)
    assert isinstance(clip_aux.norm, torch.Tensor)
    assert clip_aux.norm.shape == (3,)


def test_clipped_fun_nan_safe_replaces_nans():
    """Test nan_safe=True replaces NaN/Inf in gradients."""

    def loss_fn(param, data):
        # Create a scenario that produces NaN gradient
        # Use mean() to return scalar
        return torch.sqrt(param - data).mean()  # Gradient is undefined for param < data

    clipped_grad_fn, clip_state = clipped_fun(grad(loss_fn), batch_argnums=1, l2_clip_norm=1.0, nan_safe=True)

    param = torch.tensor(1.0, requires_grad=True)
    data = torch.tensor([0.5, 2.0, 0.3])  # data[1]=2.0 will cause NaN

    clipped_grad, _ = clipped_grad_fn(param, data, state=clip_state)
    # Should complete without error and clipped_grad should be finite
    assert torch.isfinite(clipped_grad).all()


def test_clipped_fun_dtype_controls_accumulation():
    """Test dtype parameter controls accumulation precision."""

    def loss_fn(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    # Use float64 for higher precision accumulation
    clipped_grad_fn, clip_state = clipped_fun(
        grad(loss_fn), batch_argnums=1, l2_clip_norm=1.0, normalize_by=3.0, dtype=torch.float64
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    clipped_grad, _ = clipped_grad_fn(param, data, state=clip_state)
    assert clipped_grad.dtype == torch.float64


def test_clipped_fun_pytree_params():
    """Test clip_sum with PyTree (dict) parameters."""

    def loss_fn(params, data):
        # params = {'w': tensor, 'b': tensor}
        return 0.5 * ((data - (params["w"] * data + params["b"])) ** 2).mean()

    clipped_grad_fn, clip_state = clipped_fun(
        grad(loss_fn), batch_argnums=1, l2_clip_norm=1.0, normalize_by=3.0
    )

    params = {
        "w": torch.tensor(1.0, requires_grad=True),
        "b": torch.tensor(0.5, requires_grad=True),
    }
    data = torch.tensor([0.0, 1.0, 2.0])

    clipped_grads, _ = clipped_grad_fn(params, data, state=clip_state)
    assert isinstance(clipped_grads, dict)
    assert "w" in clipped_grads and "b" in clipped_grads
    assert isinstance(clipped_grads["w"], torch.Tensor)
    assert isinstance(clipped_grads["b"], torch.Tensor)


def test_clipped_fun_nested_pytree_params():
    """Test clip_sum with deeply nested PyTree parameters."""

    def loss_fn(params, data):
        # params = {'layer1': {'w': ..., 'b': ...}, 'layer2': {'w': ..., 'b': ...}}
        pred = params["layer1"]["w"] * data + params["layer1"]["b"]
        pred = params["layer2"]["w"] * pred + params["layer2"]["b"]
        return (pred**2).mean()

    clipped_grad_fn, clip_state = clipped_fun(
        grad(loss_fn), batch_argnums=1, l2_clip_norm=2.0, normalize_by=3.0
    )

    params = {
        "layer1": {
            "w": torch.tensor(1.0, requires_grad=True),
            "b": torch.tensor(0.5, requires_grad=True),
        },
        "layer2": {
            "w": torch.tensor(2.0, requires_grad=True),
            "b": torch.tensor(-0.5, requires_grad=True),
        },
    }
    data = torch.tensor([1.0, 2.0, 3.0])

    clipped_grads, _ = clipped_grad_fn(params, data, state=clip_state)
    assert isinstance(clipped_grads, dict)
    assert "layer1" in clipped_grads and "layer2" in clipped_grads
    assert isinstance(clipped_grads["layer1"], dict)
    assert isinstance(clipped_grads["layer2"], dict)
    assert "w" in clipped_grads["layer1"] and "b" in clipped_grads["layer1"]
    assert "w" in clipped_grads["layer2"] and "b" in clipped_grads["layer2"]
    assert all(
        isinstance(clipped_grads[layer][key], torch.Tensor)
        for layer in ["layer1", "layer2"]
        for key in ["w", "b"]
    )


# ============================================================================
# Microbatching tests
# ============================================================================


def test_clipped_fun_microbatching_identical_results():
    """Test that microbatching produces identical results to non-microbatched."""

    # Simple function to clip
    def square_fn(x):
        return x ** 2

    batch_size = 100
    data = torch.randn(batch_size, 10)

    # Without microbatching
    clipped_fn, clip_state = clipped_fun(square_fn, l2_clip_norm=1.0, microbatch_size=None)
    clipped_no_mb, _ = clipped_fn(data, state=clip_state)

    # With microbatching (chunk_size=10)
    clipped_fn_mb, clip_state_mb = clipped_fun(square_fn, l2_clip_norm=1.0, microbatch_size=10)
    clipped_mb, _ = clipped_fn_mb(data, state=clip_state_mb)

    # Results should be identical
    assert torch.allclose(clipped_no_mb, clipped_mb, atol=1e-6)


def test_clipped_fun_microbatching_different_sizes():
    """Test microbatching with various chunk sizes."""

    def square_fn(x):
        return x ** 2

    batch_size = 64
    data = torch.randn(batch_size, 5)

    # Reference without microbatching
    clipped_fn_ref, clip_state_ref = clipped_fun(square_fn, l2_clip_norm=1.0)
    clipped_ref, _ = clipped_fn_ref(data, state=clip_state_ref)

    # Test different microbatch sizes
    for microbatch_size in [1, 4, 16, 32, 64]:
        clipped_fn_mb, clip_state_mb = clipped_fun(square_fn, l2_clip_norm=1.0, microbatch_size=microbatch_size)
        clipped_mb, _ = clipped_fn_mb(data, state=clip_state_mb)
        assert torch.allclose(clipped_ref, clipped_mb, atol=1e-6), (
            f"Failed for microbatch_size={microbatch_size}"
        )


def test_clipped_fun_microbatching_with_pytree():
    """Test microbatching with PyTree parameters."""

    def loss_fn(params, x):
        return torch.sum((params["w"] @ x - params["b"]) ** 2)

    batch_size = 50
    params = {
        "w": torch.randn(5, 10, requires_grad=True),
        "b": torch.randn(5, 1, requires_grad=True),
    }
    x = torch.randn(batch_size, 10, 1)

    # Create clipped gradient functions
    from opaque.clipping import clipped_grad

    clipped_grad_fn_no_mb, clip_state_no_mb = clipped_grad(
        loss_fn, argnums=0, batch_argnums=1, l2_clip_norm=1.0, microbatch_size=None
    )
    clipped_grad_fn_mb, clip_state_mb = clipped_grad(
        loss_fn, argnums=0, batch_argnums=1, l2_clip_norm=1.0, microbatch_size=10
    )

    # Compute gradients
    grads_no_mb, _ = clipped_grad_fn_no_mb(params, x, state=clip_state_no_mb)
    grads_mb, _ = clipped_grad_fn_mb(params, x, state=clip_state_mb)

    # Results should be identical
    assert torch.allclose(grads_no_mb["w"], grads_mb["w"], atol=1e-6)
    assert torch.allclose(grads_no_mb["b"], grads_mb["b"], atol=1e-6)


def test_clipped_fun_microbatching_larger_than_batch():
    """Test microbatch_size larger than batch size."""

    def square_fn(x):
        return x ** 2

    batch_size = 10
    data = torch.randn(batch_size, 5)

    # Microbatch size larger than batch
    clipped_fn_mb, clip_state_mb = clipped_fun(square_fn, l2_clip_norm=1.0, microbatch_size=100)
    clipped_mb, _ = clipped_fn_mb(data, state=clip_state_mb)

    # Reference without microbatching
    clipped_fn_ref, clip_state_ref = clipped_fun(square_fn, l2_clip_norm=1.0)
    clipped_ref, _ = clipped_fn_ref(data, state=clip_state_ref)

    # Should still work correctly
    assert torch.allclose(clipped_ref, clipped_mb, atol=1e-6)


def test_clipped_fun_microbatching_single_example():
    """Test microbatch_size=1 (process one example at a time)."""

    def square_fn(x):
        return x ** 2

    batch_size = 20
    data = torch.randn(batch_size, 3)

    # Single example microbatches
    clipped_fn_mb, clip_state_mb = clipped_fun(square_fn, l2_clip_norm=1.0, microbatch_size=1)
    clipped_mb, _ = clipped_fn_mb(data, state=clip_state_mb)

    # Reference without microbatching
    clipped_fn_ref, clip_state_ref = clipped_fun(square_fn, l2_clip_norm=1.0)
    clipped_ref, _ = clipped_fn_ref(data, state=clip_state_ref)

    # Results should be identical
    assert torch.allclose(clipped_ref, clipped_mb, atol=1e-6)


def test_clipped_fun_microbatching_with_aux():
    """Test microbatching preserves auxiliary outputs correctly."""

    def fn_with_aux(x):
        value = x ** 2
        user_aux = torch.mean(x, dim=-1)  # Per-example auxiliary
        return value, user_aux

    batch_size = 30
    data = torch.randn(batch_size, 5)

    # Without microbatching
    clipped_fn_no_mb, clip_state_no_mb = clipped_fun(
        fn_with_aux, has_aux=True, l2_clip_norm=1.0, microbatch_size=None
    )
    (clipped_no_mb, user_aux_no_mb), _ = clipped_fn_no_mb(data, state=clip_state_no_mb)

    # With microbatching
    clipped_fn_mb, clip_state_mb = clipped_fun(fn_with_aux, has_aux=True, l2_clip_norm=1.0, microbatch_size=6)
    (clipped_mb, user_aux_mb), _ = clipped_fn_mb(data, state=clip_state_mb)

    # Primary results should be identical
    assert torch.allclose(clipped_no_mb, clipped_mb, atol=1e-6)

    # Auxiliary outputs should be identical (per-example)
    assert torch.allclose(user_aux_no_mb, user_aux_mb, atol=1e-6)


def test_clipped_fun_microbatching_with_return_norms():
    """Test microbatching with return_norms=True."""

    def square_fn(x):
        return x ** 2

    batch_size = 40
    data = torch.randn(batch_size, 8)

    # Without microbatching
    clipped_fn_no_mb, clip_state_no_mb = clipped_fun(
        square_fn, l2_clip_norm=1.0, return_norms=True, microbatch_size=None
    )
    (clipped_no_mb, (_, clip_aux_no_mb)), _ = clipped_fn_no_mb(data, state=clip_state_no_mb)

    # With microbatching
    clipped_fn_mb, clip_state_mb = clipped_fun(square_fn, l2_clip_norm=1.0, return_norms=True, microbatch_size=8)
    (clipped_mb, (_, clip_aux_mb)), _ = clipped_fn_mb(data, state=clip_state_mb)

    # Primary results should be identical
    assert torch.allclose(clipped_no_mb, clipped_mb, atol=1e-6)

    # Norms should be identical (per-example)
    assert torch.allclose(clip_aux_no_mb.norm, clip_aux_mb.norm, atol=1e-6)
