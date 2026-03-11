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
from torch.utils.checkpoint import checkpoint

from opaque.clipping import adaptive_clipped_grad, clipped_grad
from opaque.distributed import get_rank, get_world_size, reduce_scalar, sum_gradients, sync
from opaque.noise import gaussian_noise, identity_mf_noise
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


class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def _block(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use non-reentrant checkpoint path, which is required for functorch transforms.
        x = checkpoint(self._block, x, use_reentrant=False)
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


def _worker_all_reduce_values(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed import all_reduce

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        base = torch.tensor([float(rank + 1), float(2 * (rank + 1))], device=device)

        summed = base.clone()
        result = all_reduce(summed, op="sum")
        assert result is None
        assert torch.allclose(summed, torch.tensor([3.0, 6.0], device=device))

        averaged = base.clone()
        all_reduce(averaged, op="mean")
        assert torch.allclose(averaged, torch.tensor([1.5, 3.0], device=device))

        maximum = base.clone()
        all_reduce(maximum, op="max")
        assert torch.allclose(maximum, torch.tensor([2.0, 4.0], device=device))

        minimum = base.clone()
        all_reduce(minimum, op="min")
        assert torch.allclose(minimum, torch.tensor([1.0, 2.0], device=device))

        product = base.clone()
        all_reduce(product, op="product")
        assert torch.allclose(product, torch.tensor([2.0, 8.0], device=device))
    finally:
        _cleanup_ddp()


def _worker_reduce_pytree(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed import reduce_pytree

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grads = {
            "w": torch.tensor([1.0, 2.0], device=device),
            "b": torch.tensor([0.5], device=device),
        }

        result = reduce_pytree(grads, op="sum")
        assert result is None
        assert torch.allclose(grads["w"], torch.tensor([2.0, 4.0], device=device))
        assert torch.allclose(grads["b"], torch.tensor([1.0], device=device))
    finally:
        _cleanup_ddp()


def _worker_reduce_pytree_nested(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed import reduce_pytree

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        pytree = {
            "encoder": {
                "w": torch.tensor([[float(rank + 1), 1.0]], device=device),
                "b": torch.tensor([float(rank)], device=device),
            },
            "head": [torch.tensor([2.0 * (rank + 1)], device=device)],
        }

        result = reduce_pytree(pytree, op="sum")
        assert result is None

        assert torch.allclose(
            pytree["encoder"]["w"],
            torch.tensor([[3.0, 2.0]], device=device),
        )
        assert torch.allclose(
            pytree["encoder"]["b"],
            torch.tensor([1.0], device=device),
        )
        assert torch.allclose(
            pytree["head"][0],
            torch.tensor([6.0], device=device),
        )
    finally:
        _cleanup_ddp()


def _worker_sync_adaptive_clip_state(rank: int, world_size: int, port: int) -> None:
    from opaque.clipping.adaptive import AdaptiveClipState
    from opaque.distributed import sync
    from opaque.random import key as rng_key

    _setup_ddp(rank, world_size, port)
    try:
        state = AdaptiveClipState(
            clip_norm=float(rank + 1),
            clipping_rate=0.5 + 0.1 * rank,
            key=rng_key(42),
            step=100,
            quantile_noise_multiplier=0.05,
            learning_rate=0.2,
            target_quantile=0.5,
            clip_norm_min=0.01,
            clip_norm_max=100.0,
            base_clip_norm=float(rank + 1),
            num_clipped=float(3 * (rank + 1)),
            total=float(10 * (rank + 1)),
            batch_size=8 * (rank + 1),
        )
        synced = sync(state)
        # num_clipped and total are summed, then global rate recomputed
        expected_total_clipped = sum(3.0 * (r + 1) for r in range(world_size))
        expected_total = sum(10.0 * (r + 1) for r in range(world_size))
        expected_rate = expected_total_clipped / expected_total
        assert abs(synced.clipping_rate - expected_rate) < 1e-5
        # batch_size is summed
        expected_batch_size = sum(8 * (r + 1) for r in range(world_size))
        assert synced.batch_size == expected_batch_size
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


def _worker_sync_noise_states(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grads = {
            "weight": torch.zeros(4, 3, device=device),
            "bias": torch.zeros(3, device=device),
        }

        gaussian_fn, gaussian_state = gaussian_noise(stddev=1.0, key=key(42))
        _noisy_gauss, gaussian_state = gaussian_fn(grads, gaussian_state)
        synced_gaussian_state = sync(gaussian_state)
        assert synced_gaussian_state.step_counter == 1

        mf_fn, mf_state = identity_mf_noise(grads, stddev=1.0, key=key(42))
        _noisy_mf, mf_state = mf_fn(grads, mf_state)
        synced_mf_state = sync(mf_state)
        assert synced_mf_state.step_counter == 1
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
        result = sum_gradients(grads)
        assert result is None
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        for grad in tree_leaves(noisy_grads):
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
        from opaque.distributed import sync

        new_state = sync(new_state)

        assert new_state.clip_norm > 0
        assert new_state.step == 1
        assert 0 <= new_state.clipping_rate <= 1
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

        local_batch_size = 4 if rank == 0 else 7
        x = torch.randn(local_batch_size, 10, device=device)
        y = torch.randn(local_batch_size, 1, device=device)

        _grads, new_state = grad_fn(params, x, y, state=clip_state)
        from opaque.distributed import sync

        synced = sync(new_state)

        assert synced.batch_size == 11
        assert synced.total == 11.0
        assert 0.0 <= synced.clipping_rate <= 1.0
    finally:
        _cleanup_ddp()


def _worker_sync_aux_adaptive_clipping(
    rank: int, world_size: int, port: int
) -> None:
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

        local_clipped = float((aux.grad_norms > new_state.clip_norm).sum().item())
        local_total = float(aux.grad_norms.numel())
        global_clipped = reduce_scalar(local_clipped, op="sum", device=device)
        global_total = reduce_scalar(local_total, op="sum", device=device)
        expected_rate = global_clipped / max(1.0, global_total)
        assert abs(synced_aux.clipping_rate - expected_rate) < 1e-6
    finally:
        _cleanup_ddp()


def _worker_checkpointed_dp_training_step(
    rank: int, world_size: int, port: int
) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = CheckpointedModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2)
        )

        x = torch.randn(8, 10, device=device)
        y = torch.randn(8, 1, device=device)

        grads, _ = grad_fn(params, x, y, state=clip_state)
        result = sum_gradients(grads)
        assert result is None

        for grad in tree_leaves(grads):
            assert grad is not None
            assert torch.isfinite(grad).all()
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

    def test_all_reduce_values(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_all_reduce_values)

    def test_reduce_pytree(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_pytree)

    def test_reduce_pytree_nested(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_pytree_nested)

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

    def test_sync_noise_states(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_noise_states)


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

    def test_adaptive_clipping_with_uneven_batches(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_adaptive_clipping_uneven_batches)

    def test_sync_aux_adaptive_clipping(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_aux_adaptive_clipping)

    def test_checkpointed_dp_training_step(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_checkpointed_dp_training_step)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
