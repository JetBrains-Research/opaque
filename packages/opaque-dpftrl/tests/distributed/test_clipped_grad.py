"""DP-FTRL clipped grad + MF Gaussian noise under NCCL."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from opaque.distributed import sum_gradients, sync
from opaque.dpftrl.clipping import clipped_grad
from opaque.dpftrl.noise import identity_strategy, mf_gaussian_noise
from opaque.functional import make_functional
from opaque.pytree import tree_map
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


def _fixed_sd() -> dict[str, torch.Tensor]:
    torch.manual_seed(1313)
    return SimpleModel().state_dict()


def _worker_mf_clip_three_steps(rank: int, world_size: int, port: int) -> None:
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
        tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
        noise_fn, noise_state = mf_gaussian_noise(
            tmpl,
            identity_strategy(),
            n_steps=4,
            noise_multiplier=1.1,
            key=key(0),
        )
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 1, device=device)
        for _ in range(3):
            grads, clip_state = grad_fn(params, x, y, state=clip_state)
            summed = sum_gradients(grads)
            _noised, noise_state = noise_fn(summed, noise_state)
            noise_state = sync(noise_state)
            assert torch.isfinite(_noised.pytree["fc1.weight"]).all()
    finally:
        _cleanup_ddp()


def _baseline_identity_mf_fc1() -> torch.Tensor:
    device = torch.device("cuda:0")
    m = SimpleModel().to(device)
    m.load_state_dict(_fixed_sd())
    func_model, params = make_functional(m)

    def loss_fn(params, x, y):
        pred = func_model(params, x)
        return ((pred - y) ** 2).mean()

    grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
    tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
    noise_fn, noise_state = mf_gaussian_noise(
        tmpl,
        identity_strategy(),
        n_steps=4,
        noise_multiplier=1.1,
        key=key(0),
    )
    x = torch.arange(80, dtype=torch.float32, device=device).reshape(8, 10)
    y = torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.1
    grads, _ = grad_fn(params, x, y, state=clip_state)
    summed = sum_gradients(grads)
    noised, _ = noise_fn(summed, noise_state)
    return noised.pytree["fc1.weight"].detach().cpu()


def _worker_mf_parity(rank: int, world_size: int, port: int, out_path: str) -> None:
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
        tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
        noise_fn, noise_state = mf_gaussian_noise(
            tmpl,
            identity_strategy(),
            n_steps=4,
            noise_multiplier=1.1,
            key=key(0),
        )
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


class TestClippedGradMFDistributed:
    def test_clipped_mf_three_steps(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_mf_clip_three_steps)

    def test_parity_identity_mf_single_vs_two_ranks(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        ref = _baseline_identity_mf_fc1()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "fc1.pt")
            _spawn(2, _worker_mf_parity, out)
            got = torch.load(out, map_location="cpu")
        assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5)
