"""Torch-autograd invariants for clipped gradient execution."""

from __future__ import annotations

import pytest
import torch

from opaque.api.engine.clipping import clipped_grad


def _sum_of_clipped_grads(grad_fn, params, batch, state):
    result, _ = grad_fn(params, batch, state=state)
    return result.pytree.sum()


def test_clipped_grad_returns_values_not_a_torch_autograd_graph() -> None:
    def loss(params, values):
        return ((values - params) ** 2).sum()

    param = torch.nn.Parameter(torch.tensor(3.0))
    grad_fn, state = clipped_grad(loss, clipping_norm=100.0)

    total = _sum_of_clipped_grads(grad_fn, param, torch.tensor([0.0, 5.0]), state)

    assert not total.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(total, param)


def test_clipped_grad_composes_under_torch_func_grad() -> None:
    def loss(params, values):
        return ((values - params) ** 2).sum()

    grad_fn, state = clipped_grad(loss, clipping_norm=100.0)
    batch = torch.tensor([0.0, 5.0])

    second = torch.func.grad(
        lambda params: _sum_of_clipped_grads(grad_fn, params, batch, state)
    )(torch.tensor(3.0))

    assert second.item() == pytest.approx(2 * len(batch))
