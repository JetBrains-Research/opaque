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

from opaque.types import clipped

from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.distributed import get_rank, get_world_size, sum_gradients, sync
from opaque.distributed.gradients import sum_gradients_
from opaque.api.engine.distributed._state import reduce_scalar
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpftrl.noise import mf_noise, identity_strategy
from opaque.profiling import StepTimer, TrainingProfiler
from opaque.random import key
from opaque.functional import make_functional
from opaque.pytree import tree_leaves

# Mark all tests in this file as requiring GPU
pytestmark = pytest.mark.cuda


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
    from opaque.distributed.collectives import all_reduce, all_reduce_

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        base = torch.tensor([float(rank + 1), float(2 * (rank + 1))], device=device)

        result = all_reduce(base, op="sum")
        assert torch.allclose(
            base, torch.tensor([float(rank + 1), float(2 * (rank + 1))], device=device)
        )
        assert torch.allclose(result, torch.tensor([3.0, 6.0], device=device))

        averaged = base.clone()
        inplace_result = all_reduce_(averaged, op="mean")
        assert inplace_result is None
        assert torch.allclose(averaged, torch.tensor([1.5, 3.0], device=device))

        maximum = all_reduce(base, op="max")
        assert torch.allclose(maximum, torch.tensor([2.0, 4.0], device=device))

        minimum = all_reduce(base, op="min")
        assert torch.allclose(minimum, torch.tensor([1.0, 2.0], device=device))

        product = all_reduce(base, op="product")
        assert torch.allclose(product, torch.tensor([2.0, 8.0], device=device))
    finally:
        _cleanup_ddp()


def _worker_reduce_pytree(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.gradients import reduce_pytree, reduce_pytree_

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grads = {
            "w": torch.tensor([1.0, 2.0], device=device),
            "b": torch.tensor([0.5], device=device),
        }

        local_grads = {
            "w": grads["w"].clone(),
            "b": grads["b"].clone(),
        }

        result = reduce_pytree(grads, op="sum")
        assert torch.allclose(grads["w"], local_grads["w"])
        assert torch.allclose(grads["b"], local_grads["b"])
        assert torch.allclose(result["w"], torch.tensor([2.0, 4.0], device=device))
        assert torch.allclose(result["b"], torch.tensor([1.0], device=device))

        inplace_result = reduce_pytree_(grads, op="sum")
        assert inplace_result is None
        assert torch.allclose(grads["w"], torch.tensor([2.0, 4.0], device=device))
        assert torch.allclose(grads["b"], torch.tensor([1.0], device=device))
    finally:
        _cleanup_ddp()


def _worker_reduce_pytree_nested(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.gradients import reduce_pytree

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

        original_w = pytree["encoder"]["w"].clone()
        original_b = pytree["encoder"]["b"].clone()
        original_head = pytree["head"][0].clone()

        result = reduce_pytree(pytree, op="sum")

        assert torch.allclose(pytree["encoder"]["w"], original_w)
        assert torch.allclose(pytree["encoder"]["b"], original_b)
        assert torch.allclose(pytree["head"][0], original_head)

        assert torch.allclose(
            result["encoder"]["w"],
            torch.tensor([[3.0, 2.0]], device=device),
        )
        assert torch.allclose(
            result["encoder"]["b"],
            torch.tensor([1.0], device=device),
        )
        assert torch.allclose(
            result["head"][0],
            torch.tensor([6.0], device=device),
        )
    finally:
        _cleanup_ddp()


def _worker_sync_adaptive_clip_state(rank: int, world_size: int, port: int) -> None:
    from opaque.api.dpsgd.clipping._adaptive import AdaptiveClipState
    from opaque.distributed import sync
    from opaque.random import key as rng_key

    _setup_ddp(rank, world_size, port)
    try:
        state = AdaptiveClipState(
            _current_clipping_norm=float(rank + 1),
            _next_clipping_norm=float(rank + 1),
            _step=100,
            _rng_key=rng_key(42),
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
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)

        gathered = [
            torch.zeros_like(noised.pytree["weight"]) for _ in range(world_size)
        ]
        dist.all_gather(gathered, noised.pytree["weight"])
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

        gaussian_fn, gaussian_state = gaussian_noise(noise_multiplier=1.0, key=key(42))
        _noisy_gauss, gaussian_state = gaussian_fn(
            clipped(grads, max_norm=1.0), gaussian_state
        )
        synced_gaussian_state = sync(gaussian_state)
        assert synced_gaussian_state._step_counter == 1

        mf_fn, mf_state = mf_noise(
            grads,
            identity_strategy(),
            n_steps=1,
            noise_multiplier=1.0,
            key=key(42),
        )
        _noisy_mf, mf_state = mf_fn(clipped(grads, max_norm=1.0), mf_state)
        synced_mf_state = sync(mf_state)
        assert synced_mf_state._step_counter == 1
    finally:
        _cleanup_ddp()


def _worker_sync_profiler(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        profiler = TrainingProfiler(device)

        timer = StepTimer(device, batch_size=4)
        with timer:
            x = torch.randn(1024 + 512 * rank, device=device)
            _ = (x * x).sum()
        profiler = profiler.add_step(timer)

        # --- first sync ---
        synced_profiler = sync(profiler)
        assert synced_profiler is not profiler  # must be a new object
        assert synced_profiler.num_steps == 1
        # batch_size is summed across ranks (4 per rank × world_size)
        assert synced_profiler._step_batch_sizes[-1] == world_size * 4
        # original profiler must be unchanged
        assert profiler._step_batch_sizes[-1] == 4
        # pending records moved to synced; nothing left to flush
        assert synced_profiler.is_fully_synced
        # step time uses max across ranks (rank 1 does more work → rank 1 ≥ rank 0)
        assert synced_profiler._step_metrics[-1]._step_time >= 0.0

        # peak is consistent across ranks after sync
        local_peak = float(synced_profiler._observed_peak_gb)
        peak_min = reduce_scalar(local_peak, op="min", device=device)
        peak_max = reduce_scalar(local_peak, op="max", device=device)
        assert abs(peak_max - peak_min) < 1e-6

        # --- second sync is idempotent: returns the same object ---
        twice_synced = sync(synced_profiler)
        assert twice_synced is synced_profiler

        # --- checkpoint sync ---
        profiler_with_mark, _ = synced_profiler.mark("after_sync")
        assert not profiler_with_mark.is_fully_synced  # mark adds a pending checkpoint
        synced_with_mark = sync(profiler_with_mark)
        assert synced_with_mark.is_fully_synced
        assert len(synced_with_mark.checkpoints) == 1
        assert synced_with_mark.checkpoints[0].name == "after_sync"
        assert synced_with_mark._synced_checkpoints == len(synced_with_mark.checkpoints)
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
            initial_clipping_norm=0.1,
            key=key(0),
        )

        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        grads, new_state = grad_fn(params, x, y, state=clip_state)
        from opaque.distributed import sync

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
        from opaque.distributed import sync

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

    def test_sync_profiler(self):
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_sync_profiler)


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
