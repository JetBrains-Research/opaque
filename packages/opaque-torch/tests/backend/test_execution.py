"""Tests for Torch execution transforms."""

from __future__ import annotations

import pytest
import torch

from opaque.api.engine.autodiff import grad_and_value, vmap
from opaque.api.engine.backend import active_backend, clear_backend
from opaque.api.torch.backend import torch_backend
from opaque.execution import (
    ExecutionProfile,
    checkpoint,
    compile,
    optimize_saved_activations,
)


@pytest.fixture(autouse=True)
def _unselected_backend():
    clear_backend()
    yield
    clear_backend()


def _square_sum(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum()


def test_compile_preserves_eager_values() -> None:
    compiled = compile(_square_sum)
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    expected = _square_sum(x)
    got = compiled(x)
    torch.testing.assert_close(got, expected)
    assert active_backend() is not None
    assert active_backend().name == "torch"


def test_compile_preserves_gradients() -> None:
    compiled = compile(_square_sum)
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (expected,) = torch.autograd.grad(_square_sum(x), x)
    (got,) = torch.autograd.grad(compiled(x), x)
    torch.testing.assert_close(got, expected)


def test_checkpoint_preserves_eager_values_and_gradients() -> None:
    def block(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ weight)

    def loss_fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return checkpoint(block)(x, weight).sum()

    x = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(8, 8, requires_grad=True)

    ref_out = torch.relu(x @ weight).sum()
    (ref_gx, ref_gw) = torch.autograd.grad(ref_out, (x, weight))

    out = loss_fn(x, weight)
    (gx, gw) = torch.autograd.grad(out, (x, weight))

    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(gx, ref_gx)
    torch.testing.assert_close(gw, ref_gw)


def test_vmap_grad_checkpoint_matches_eager() -> None:
    def block(x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum()

    batched = torch.randn(4, 3, requires_grad=True)

    grad_ckpt = grad_and_value(checkpoint(block))
    grad_ref = grad_and_value(block)

    got_grads, got_values = vmap(grad_ckpt)(batched)
    ref_grads, ref_values = vmap(grad_ref)(batched)

    torch.testing.assert_close(got_values, ref_values)
    torch.testing.assert_close(got_grads, ref_grads)


def test_optimize_saved_activations_preserves_values_and_gradients() -> None:
    def fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (x @ weight).sum()

    x = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(8, 8, requires_grad=True)

    ref_out = fn(x, weight)
    (ref_gx, ref_gw) = torch.autograd.grad(ref_out, (x, weight))

    optimized = optimize_saved_activations(fn)
    out = optimized(x, weight)
    (gx, gw) = torch.autograd.grad(out, (x, weight))

    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(gx, ref_gx)
    torch.testing.assert_close(gw, ref_gw)


def test_execution_profiles_report_torch_supported() -> None:
    backend = torch_backend()
    assert ExecutionProfile.COMPILATION.supports(backend)
    assert ExecutionProfile.CHECKPOINTING.supports(backend)
    assert ExecutionProfile.SAVED_ACTIVATIONS.supports(backend)
