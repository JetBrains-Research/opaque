"""Integration tests for DDP (DistributedDataParallel) training.

These tests validate that Opaque's distributed primitives work correctly
with PyTorch DDP on multiple GPUs. Tests spawn workers internally so
`torchrun` is not required.

Run with:
    uv run pytest tests/distributed/test_ddp_integration.py -v
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from opaque.clipping import adaptive_clipped_grad, clipped_grad
from opaque.distributed import get_rank, get_world_size, reduce_scalar, sum_gradients
from opaque.noise import gaussian_noise
from opaque.random import key
from opaque.utils import make_functional
from opaque.utils.pytree import tree_leaves

# Mark all tests in this file as requiring GPU
pytestmark = pytest.mark.gpu


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _is_ddp_available() -> bool:
    return dist.is_available() and torch.cuda.is_available()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup_ddp(rank: int, world_size: int, port: int) -> None:
    if not _is_ddp_available():
        raise RuntimeError("DDP requires CUDA and torch.distributed support")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn(world_size: int, fn, *args) -> None:
    port = _find_free_port()
    mp.spawn(fn, args=(world_size, port, *args), nprocs=world_size, join=True)


def _worker_distributed_detection(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed import is_distributed as opaque_is_distributed

    _setup_ddp(rank, world_size, port)
    try:
        assert opaque_is_distributed() is True
        assert dist.is_initialized() is True
    finally:
        _cleanup_ddp()


def _worker_rank_world_size(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        assert get_rank() == rank
        assert get_world_size() == world_size
        assert 0 <= rank < world_size
    finally:
        _cleanup_ddp()


def _worker_reduce_scalar(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        value = float(rank + 1)
        synced = reduce_scalar(value, op="mean", device=device)
        expected_avg = sum(range(1, world_size + 1)) / world_size
        assert abs(synced - expected_avg) < 1e-5
    finally:
        _cleanup_ddp()


def _worker_sync_adaptive_clip_state(rank: int, world_size: int, port: int) -> None:
    from opaque.clipping.adaptive import AdaptiveClipState
    from opaque.distributed import sync_state

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        state = AdaptiveClipState(
            clip_norm=float(rank + 1),
            step=100,
            clipping_rate=0.5 + 0.1 * rank,
            batch_size=8 * (rank + 1),
        )
        synced = sync_state(
            state,
            field_ops={"clip_norm": "mean", "clipping_rate": "mean"},
            device=device,
        )
        expected_clip_norm = sum(range(1, world_size + 1)) / world_size
        expected_rate = sum(0.5 + 0.1 * r for r in range(world_size)) / world_size
        assert abs(synced.clip_norm - expected_clip_norm) < 1e-5
        assert abs(synced.clipping_rate - expected_rate) < 1e-5
        assert synced.step == 100
    finally:
        _cleanup_ddp()


def _worker_shared_noise_is_deterministic(
    rank: int, world_size: int, port: int
) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grads = {
            "weight": torch.zeros(10, 5, device=device),
            "bias": torch.zeros(5, device=device),
        }
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(0))
        noisy, _ = noise_fn(grads, state)

        gathered = [torch.zeros_like(noisy["weight"]) for _ in range(world_size)]
        dist.all_gather(gathered, noisy["weight"])
        if rank == 0:
            for other in gathered[1:]:
                assert torch.allclose(gathered[0], other)
    finally:
        _cleanup_ddp()


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
            loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2)
        )
        noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(0))

        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        grads, clip_state = grad_fn(params, x, y, state=clip_state)
        grads = sum_gradients(grads)
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        for grad, param in zip(
            tree_leaves(noisy_grads), tree_leaves(params), strict=True
        ):
            assert grad.shape == param.shape
            assert grad.device == device
            assert not torch.isnan(grad).any()
            assert not torch.isinf(grad).any()
    finally:
        _cleanup_ddp()


def _worker_adaptive_clipping(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clip_norm=0.1,
            key=key(0),
        )

        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        grads, new_state = grad_fn(params, x, y, state=clip_state)
        from opaque.distributed import sync_state

        new_state = sync_state(
            new_state,
            field_ops={"clip_norm": "mean", "clipping_rate": "mean"},
            device=device,
        )

        assert new_state.clip_norm > 0
        assert new_state.step == 1
        assert 0 <= new_state.clipping_rate <= 1
        assert grads is not None
    finally:
        _cleanup_ddp()


class TestDistributedUtilities:
    """Test basic distributed utilities."""

    def test_distributed_detection(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_distributed_detection)

    def test_rank_and_world_size(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_rank_world_size)


class TestStateSynchronization:
    """Test state synchronization across devices."""

    def test_reduce_scalar(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_scalar)

    def test_sync_adaptive_clip_state(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_adaptive_clip_state)


class TestDeterministicNoise:
    """Test deterministic noise generation in distributed mode."""

    def test_shared_noise_is_deterministic(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_shared_noise_is_deterministic)


class TestEndToEndDPTraining:
    """End-to-end DP training with DDP."""

    def test_dp_training_step(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_dp_training_step)

    def test_adaptive_clipping_with_sync(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_adaptive_clipping)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
