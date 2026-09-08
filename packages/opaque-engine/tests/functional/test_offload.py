# Copyright (c) 2026 Opaque Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref

import pytest
import torch
from torch.func import grad, vmap

from opaque.functional import SaveOnCpuStats, save_on_cpu


def test_first_order_torch_func_works_without_patch_package():
    x = torch.randn(8)
    with save_on_cpu(min_bytes=0):
        actual = grad(lambda value: value.square().sum())(x)

    torch.testing.assert_close(actual, 2 * x)


def test_cpu_tensors_are_left_in_place():
    x = torch.randn(8, requires_grad=True)
    ctx = save_on_cpu(min_bytes=0)

    with ctx:
        x.square().sum().backward()

    assert ctx.stats.selected_tensors == 0
    assert ctx.stats.skipped_device_tensors > 0


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"pin_memory": 1}, TypeError, "pin_memory"),
        ({"min_bytes": -1}, ValueError, "min_bytes"),
        ({"min_bytes": False}, ValueError, "min_bytes"),
        ({"max_pinned_bytes": -1}, ValueError, "max_pinned_bytes"),
        ({"stats": object()}, TypeError, "stats"),
    ],
)
def test_configuration_validation(kwargs, error, match):
    with pytest.raises(error, match=match):
        save_on_cpu(**kwargs)


def test_skipped_graph_output_does_not_retain_input_tensor():
    observed = None
    x = torch.randn(8, requires_grad=True)

    def loss_fn():
        nonlocal observed
        activation = x * 2
        observed = weakref.ref(activation)
        return activation.square().sum()

    with save_on_cpu(min_bytes=1 << 30):
        loss = loss_fn()
    loss.backward()

    assert observed is not None
    assert observed() is None


def test_skipped_tensor_preserves_version_check():
    x = torch.ones(3, requires_grad=True)
    with save_on_cpu(min_bytes=1 << 30):
        loss = x.square().sum()
    with torch.no_grad():
        x.add_(1)

    with pytest.raises(RuntimeError, match="modified after it was saved"):
        loss.backward()


def test_stats_are_reusable_and_flattenable():
    stats = SaveOnCpuStats()
    ctx = save_on_cpu(min_bytes=0, stats=stats)
    x = torch.randn(8, requires_grad=True)

    with ctx:
        x.square().sum().backward()
    with ctx:
        x.cos().sum().backward()

    assert ctx.stats is stats
    assert stats.skipped_device_tensors >= 2
    assert stats.to_dict()["activation_offload_skipped_device_tensors"] >= 2


@pytest.mark.mps
def test_mps_vmap_grad_parity_and_selection():
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="mps")
    w1 = torch.randn(64, 64, device="mps")
    w2 = torch.randn(64, 64, device="mps")

    def loss(a, b, row):
        return ((row @ a) @ b).square().sum()

    expected = vmap(grad(loss, argnums=(0, 1)), in_dims=(None, None, 0))(w1, w2, x)
    ctx = save_on_cpu(min_bytes=0, protected_tensors=(w1, w2))
    with ctx:
        actual = vmap(grad(loss, argnums=(0, 1)), in_dims=(None, None, 0))(w1, w2, x)

    torch.testing.assert_close(actual, expected)
    assert ctx.stats.selected_tensors > 0
    assert ctx.stats.pageable_bytes > 0
    assert ctx.stats.skipped_protected_tensors > 0


@pytest.mark.mps
def test_offloaded_tensor_supports_repeated_backward():
    x = torch.randn(1024, device="mps", requires_grad=True)
    ctx = save_on_cpu(min_bytes=0)
    with ctx:
        loss = x.square().sum()

    loss.backward(retain_graph=True)
    first = x.grad.detach().clone()
    x.grad = None
    loss.backward()

    torch.testing.assert_close(x.grad, first)


@pytest.mark.mps
def test_threshold_skips_physical_vmap_batch_bytes():
    x = torch.randn(4, 64, device="mps")
    w = torch.randn(64, 64, device="mps")

    def loss(weight, row):
        return (row @ weight).square().sum()

    ctx = save_on_cpu(min_bytes=1 << 30)
    with ctx:
        vmap(grad(loss), in_dims=(None, 0))(w, x)

    assert ctx.stats.selected_tensors == 0
    assert ctx.stats.skipped_small_tensors > 0
    assert ctx.stats.skipped_small_bytes > 0


@pytest.mark.mps
def test_protected_storage_covers_views():
    weight = torch.randn(64, 64, device="mps", requires_grad=True)
    x = torch.randn(64, device="mps", requires_grad=True)
    ctx = save_on_cpu(min_bytes=0, protected_tensors=weight)

    with ctx:
        (x @ weight.t()).square().sum().backward()

    assert ctx.stats.skipped_protected_tensors > 0


@pytest.mark.cuda
def test_cuda_overlap_snapshots_before_later_in_place_write():
    x = torch.randn(1 << 18, device="cuda", requires_grad=True)
    original = x.detach().clone()
    ctx = save_on_cpu(pin_memory=True, min_bytes=0)

    with ctx:
        activation = x * 2
        loss = activation.square().sum()
        activation.add_(100)
    loss.backward()

    torch.testing.assert_close(x.grad, 8 * original)


@pytest.mark.cuda
def test_cuda_overlap_uses_each_source_device_stream():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    first = torch.randn(1 << 18, device="cuda:0", requires_grad=True)
    second = torch.randn(1 << 18, device="cuda:1", requires_grad=True)
    ctx = save_on_cpu(pin_memory=True, min_bytes=0)

    with ctx:
        first_loss = first.square().sum()
        second_loss = second.square().sum()
    first_loss.backward()
    second_loss.backward()

    torch.testing.assert_close(first.grad, 2 * first.detach())
    torch.testing.assert_close(second.grad, 2 * second.detach())


@pytest.mark.cuda
def test_cuda_pinned_budget_falls_back_to_pageable():
    x = torch.randn(8, 512, device="cuda")
    w1 = torch.randn(512, 512, device="cuda")
    w2 = torch.randn(512, 512, device="cuda")

    def loss(a, b, row):
        return ((row @ a).relu() @ b).square().sum()

    expected = vmap(grad(loss, argnums=(0, 1)), in_dims=(None, None, 0))(w1, w2, x)
    budget = 64 << 10
    ctx = save_on_cpu(
        pin_memory=True,
        min_bytes=0,
        protected_tensors=(w1, w2),
        max_pinned_bytes=budget,
    )
    with ctx:
        actual = vmap(grad(loss, argnums=(0, 1)), in_dims=(None, None, 0))(w1, w2, x)

    torch.testing.assert_close(actual, expected)
    assert 0 < ctx.stats.peak_pinned_bytes <= budget
    assert ctx.stats.max_pending_transfers <= 2
    assert ctx.stats.pinned_bytes > 0
    assert ctx.stats.pageable_bytes > 0
