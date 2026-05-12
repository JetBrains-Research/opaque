"""Band MF noise parity under NCCL vs single-process."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from opaque.distributed import sum_gradients
from opaque.dpftrl.clipping import clipped_grad
from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opaque.functional import make_functional
from opaque.pytree import tree_map
from opaque.random import key

from ._ddp_helpers import _cleanup_ddp, _setup_ddp, _spawn


pytestmark = pytest.mark.cuda


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _fixed_sd() -> dict[str, torch.Tensor]:
    torch.manual_seed(9191)
    return TinyModel().state_dict()


def _baseline_band_fc1() -> torch.Tensor:
    device = torch.device("cuda:0")
    m = TinyModel().to(device)
    m.load_state_dict(_fixed_sd())
    func_model, params = make_functional(m)

    def loss_fn(params, x, y):
        pred = func_model(params, x)
        return ((pred - y) ** 2).mean()

    grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
    tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
    noise_fn, noise_state = mf_gaussian_noise(
        tmpl,
        band_mf_strategy(bands=2, momentum=0.9),
        n_steps=4,
        noise_multiplier=0.85,
        key=key(5),
    )
    x = torch.arange(48, dtype=torch.float32, device=device).reshape(8, 6)
    y = torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.05
    grads, _ = grad_fn(params, x, y, state=clip_state)
    summed = sum_gradients(grads)
    noised, _ = noise_fn(summed, noise_state)
    return noised.pytree["fc1.weight"].detach().cpu()


def _worker_band_parity(rank: int, world_size: int, port: int, out_path: str) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = TinyModel().to(device)
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
            band_mf_strategy(bands=2, momentum=0.9),
            n_steps=4,
            noise_multiplier=0.85,
            key=key(5),
        )
        x_full = torch.arange(48, dtype=torch.float32, device=device).reshape(8, 6)
        y_full = (
            torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.05
        )
        sl = slice(rank * 4, (rank + 1) * 4)
        grads, _ = grad_fn(params, x_full[sl], y_full[sl], state=clip_state)
        summed = sum_gradients(grads)
        noised, _ = noise_fn(summed, noise_state)
        if rank == 0:
            torch.save(noised.pytree["fc1.weight"].cpu(), out_path)
    finally:
        _cleanup_ddp()


class TestBandMfNoiseDistributed:
    def test_parity_band_mf_single_vs_two_ranks(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        ref = _baseline_band_fc1()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "fc1.pt")
            _spawn(2, _worker_band_parity, out)
            got = torch.load(out, map_location="cpu")
        assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5)
