"""AUTO-S (second moment) + MF Gaussian noise under NCCL."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from opaque.distributed import sum_gradients, sync
from opaque.dpftrl.clipping import auto_clipped_grad
from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opaque.functional import make_functional
from opaque.pytree import tree_map
from opaque.random import key
from opaque.types import SecondMomentNoiseOutput

from ._ddp_helpers import _cleanup_ddp, _setup_ddp, _spawn


pytestmark = pytest.mark.cuda


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _worker_auto_mf(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(777 + rank)
        model = SimpleModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = auto_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            R=1.0,
            normalize_by=4.0,
            key=key(11),
            second_moment=True,
        )
        tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
        noise_fn, noise_state = mf_gaussian_noise(
            tmpl,
            band_mf_strategy(bands=2, momentum=0.9),
            n_steps=4,
            noise_multiplier=0.6,
            key=key(22),
        )
        x = torch.randn(4, 8, device=device)
        y = torch.randn(4, 1, device=device)
        for _ in range(2):
            grads, clip_state = grad_fn(params, x, y, state=clip_state)
            summed = sum_gradients(grads)
            noised, noise_state = noise_fn(summed, noise_state)
            assert isinstance(noised, SecondMomentNoiseOutput)
            assert torch.isfinite(noised.noisy_grads.pytree["fc1.weight"]).all()
            assert torch.isfinite(noised.noisy_squared_grads.pytree["fc1.weight"]).all()
            noise_state = sync(noise_state)
            clip_state = sync(clip_state)
    finally:
        _cleanup_ddp()


class TestAutoClippedMFDistributed:
    def test_auto_clipped_band_mf_multi_step(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_auto_mf)
