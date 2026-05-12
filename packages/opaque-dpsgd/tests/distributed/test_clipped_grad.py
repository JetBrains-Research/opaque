"""Distributed clipped-grad + Gaussian noise (NCCL)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.distributed import sum_gradients
from opaque.functional import make_functional
from opaque.pytree import tree_leaves
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


class CheckpointedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def _block(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = checkpoint(self._block, x, use_reentrant=False)
        return self.fc2(x)


def _worker_dp_training_step(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )
        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))

        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        grads, clip_state = grad_fn(params, x, y, state=clip_state)
        summed_grads = sum_gradients(grads)
        for grad, summed in zip(
            tree_leaves(grads), tree_leaves(summed_grads), strict=False
        ):
            assert grad is not summed
        noisy_grads, noise_state = noise_fn(summed_grads, noise_state)

        for grad in tree_leaves(noisy_grads.pytree):
            assert grad.device == device
            assert not torch.isnan(grad).any()
            assert not torch.isinf(grad).any()
    finally:
        _cleanup_ddp()


def _worker_checkpointed_dp_training_step(
    rank: int, world_size: int, port: int
) -> None:
    from opaque.distributed.gradients import sum_gradients_

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = CheckpointedModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )

        x = torch.randn(8, 10, device=device)
        y = torch.randn(8, 1, device=device)

        grads, _ = grad_fn(params, x, y, state=clip_state)
        result = sum_gradients_(grads)
        assert result is None

        for grad in tree_leaves(grads):
            assert grad is not None
            assert torch.isfinite(grad).all()
    finally:
        _cleanup_ddp()


def _fixed_sd() -> dict[str, torch.Tensor]:
    torch.manual_seed(4242)
    m = SimpleModel()
    return m.state_dict()


def _baseline_noised_fc1() -> torch.Tensor:
    device = torch.device("cuda:0")
    m = SimpleModel().to(device)
    m.load_state_dict(_fixed_sd())
    func_model, params = make_functional(m)

    def loss_fn(params, x, y):
        pred = func_model(params, x)
        return ((pred - y) ** 2).mean()

    grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))
    x = torch.arange(80, dtype=torch.float32, device=device).reshape(8, 10)
    y = torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.1
    grads, _ = grad_fn(params, x, y, state=clip_state)
    summed = sum_gradients(grads)
    noised, _ = noise_fn(summed, noise_state)
    return noised.pytree["fc1.weight"].detach().cpu()


def _worker_dp_parity(rank: int, world_size: int, port: int, out_path: str) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        model.load_state_dict(_fixed_sd())
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )
        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))

        x_full = torch.arange(80, dtype=torch.float32, device=device).reshape(8, 10)
        y_full = torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.1
        sl = slice(rank * 4, (rank + 1) * 4)
        x = x_full[sl]
        y = y_full[sl]

        grads, _ = grad_fn(params, x, y, state=clip_state)
        summed = sum_gradients(grads)
        noised, _ = noise_fn(summed, noise_state)
        if rank == 0:
            torch.save(noised.pytree["fc1.weight"].cpu(), out_path)
    finally:
        _cleanup_ddp()


class TestClippedGradDistributed:
    def test_dp_training_step(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_dp_training_step)

    def test_checkpointed_dp_training_step(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_checkpointed_dp_training_step)

    def test_parity_single_vs_two_ranks(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        ref = _baseline_noised_fc1()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "fc1.pt")
            _spawn(2, _worker_dp_parity, out)
            got = torch.load(out, map_location="cpu")
        assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5)
