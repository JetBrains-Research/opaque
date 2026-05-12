"""Distributed adaptive clipping (NCCL)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from opaque.api.engine.distributed._state import reduce_scalar
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.distributed import sync
from opaque.random import key

from ._ddp_helpers import _cleanup_ddp, _setup_ddp, _spawn


pytestmark = pytest.mark.cuda


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _worker_sync_adaptive_clip_state(rank: int, world_size: int, port: int) -> None:
    from opaque.api.dpsgd.clipping._adaptive import AdaptiveClipState
    from opaque.distributed import sync

    _setup_ddp(rank, world_size, port)
    try:
        state = AdaptiveClipState(
            _current_clipping_norm=float(rank + 1),
            _next_clipping_norm=float(rank + 1),
            _step=100,
            _rng_key=key(42),
            _fraction_noise_std=0.05,
            _learning_rate=0.2,
            _target_quantile=0.5,
            _clipping_norm_min=0.01,
            _clipping_norm_max=100.0,
            _num_clipped=float(3 * (rank + 1)),
            _batch_size=8 * (rank + 1),
        )
        synced = sync(state)
        expected_bs = sum(8 * (r + 1) for r in range(world_size))
        assert synced._batch_size == expected_bs
    finally:
        _cleanup_ddp()


def _worker_adaptive_clipping(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        from opaque.functional import make_functional

        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.1,
            key=key(0),
        )

        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        grads, new_state = grad_fn(params, x, y, state=clip_state)
        new_state = sync(new_state)

        assert new_state._current_clipping_norm > 0
        assert new_state._step == 1
        assert grads is not None
    finally:
        _cleanup_ddp()


def _worker_adaptive_clipping_uneven_batches(
    rank: int, world_size: int, port: int
) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        from opaque.functional import make_functional

        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.1,
            key=key(0),
        )

        local_batch_size = 4 if rank == 0 else 7
        x = torch.randn(local_batch_size, 10, device=device)
        y = torch.randn(local_batch_size, 1, device=device)

        _grads, new_state = grad_fn(params, x, y, state=clip_state)
        synced = sync(new_state)

        assert synced._batch_size == 11
        assert synced._next_clipping_norm > 0
    finally:
        _cleanup_ddp()


def _worker_sync_aux_adaptive_clipping(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        from opaque.functional import make_functional

        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.1,
            key=key(0),
            return_aux=True,
        )

        local_batch_size = 3 if rank == 0 else 5
        x = torch.randn(local_batch_size, 10, device=device)
        y = torch.randn(local_batch_size, 1, device=device)

        (_grads, aux), new_state = grad_fn(params, x, y, state=clip_state)
        synced_aux = sync(aux)

        expected_n = sum(3 if r == 0 else 5 for r in range(world_size))
        assert synced_aux.loss_values.shape[0] == expected_n
        assert synced_aux.grad_norms.shape[0] == expected_n
        assert synced_aux.clipped_grad_norms.shape[0] == expected_n

        local_clipped = float(
            (aux.grad_norms > new_state._current_clipping_norm).sum().item()
        )
        local_total = float(aux.grad_norms.numel())
        global_clipped = reduce_scalar(local_clipped, op="sum", device=device)
        global_total = reduce_scalar(local_total, op="sum", device=device)
        expected_rate = global_clipped / max(1.0, global_total)
        assert abs(synced_aux.clipping_rate - expected_rate) < 1e-6
    finally:
        _cleanup_ddp()


class TestAdaptiveClippingDistributed:
    def test_sync_adaptive_clip_state(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_adaptive_clip_state)

    def test_adaptive_clipping_with_sync(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_adaptive_clipping)

    def test_adaptive_clipping_with_uneven_batches(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_adaptive_clipping_uneven_batches)

    def test_sync_aux_adaptive_clipping(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_aux_adaptive_clipping)
