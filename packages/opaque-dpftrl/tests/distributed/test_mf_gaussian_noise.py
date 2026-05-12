"""MF Gaussian noise under NCCL (identity + cross-rank determinism)."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from opaque.types import clipped

from opaque.dpftrl.noise import mf_gaussian_noise, identity_strategy
from opaque.random import key

from ._ddp_helpers import _cleanup_ddp, _setup_ddp, _spawn


pytestmark = pytest.mark.cuda


def _worker_identity_mf_three_steps(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        batch_size = 32
        param_dim = 64
        grad_template = {"weight": torch.zeros(batch_size, param_dim, device=device)}
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            identity_strategy(),
            n_steps=3,
            noise_multiplier=1.0,
            key=key(0),
        )
        step_noise_values = []
        step_stds = []
        for _step in range(3):
            grads = clipped(
                {"weight": torch.zeros(batch_size, param_dim, device=device)},
                max_norm=1.0,
            )
            noised, state = noise_fn(grads, state)
            step_noise_values.append(noised.pytree["weight"].clone())
            step_stds.append(noised.pytree["weight"].std().item())

        assert not torch.allclose(step_noise_values[0], step_noise_values[1])
        assert not torch.allclose(step_noise_values[1], step_noise_values[2])

        for step_idx, std in enumerate(step_stds):
            assert 0.8 < std < 1.2, f"Step {step_idx}: std {std} out of range"

        for step_idx, noise_val in enumerate(step_noise_values):
            assert torch.isfinite(noise_val).all(), f"Step {step_idx}: non-finite noise"

        for step_idx, noise_val in enumerate(step_noise_values):
            gathered = [torch.zeros_like(noise_val) for _ in range(world_size)]
            dist.all_gather(gathered, noise_val)
            if rank == 0:
                for other in gathered[1:]:
                    assert torch.allclose(gathered[0], other, atol=1e-6), (
                        f"Step {step_idx}: rank 0 and other ranks disagree"
                    )

    finally:
        _cleanup_ddp()


def _worker_mf_shared_noise(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grad_template = {"weight": torch.zeros(4, device=device)}
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            identity_strategy(),
            n_steps=1,
            noise_multiplier=1.0,
            key=key(0),
        )
        grads = {"weight": torch.zeros(4, device=device)}
        noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)

        w = noised.pytree["weight"]
        gathered = [torch.zeros_like(w) for _ in range(world_size)]
        dist.all_gather(gathered, w)
        if rank == 0:
            for other in gathered[1:]:
                assert torch.equal(gathered[0], other)
    finally:
        _cleanup_ddp()


class TestIdentityMFMultiStepDistributed:
    def test_identity_mf_three_steps_cross_rank(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_identity_mf_three_steps)


class TestDistributedMFNoiseSpawn:
    def test_mf_noise_shared_seed_byte_identical(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_mf_shared_noise)
