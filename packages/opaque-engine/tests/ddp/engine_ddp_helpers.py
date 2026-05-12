"""Minimal NCCL DDP helpers + mp.spawn entrypoints (must live in this module for pickle)."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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


def _worker_reduce_scalar(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import reduce_scalar

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


def _worker_sync_profiler(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import reduce_scalar
    from opaque.distributed import sync
    from opaque.profiling import StepTimer, TrainingProfiler

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        profiler = TrainingProfiler(device)

        timer = StepTimer(device, batch_size=4)
        with timer:
            x = torch.randn(1024 + 512 * rank, device=device)
            _ = (x * x).sum()
        profiler = profiler.add_step(timer)

        synced_profiler = sync(profiler)
        assert synced_profiler is not profiler
        assert synced_profiler.num_steps == 1
        assert synced_profiler.step_batch_sizes[-1] == world_size * 4
        assert profiler.step_batch_sizes[-1] == 4
        assert synced_profiler.is_fully_synced
        assert synced_profiler.step_metrics[-1].step_time >= 0.0

        local_peak = float(synced_profiler._observed_peak_gb)
        peak_min = reduce_scalar(local_peak, op="min", device=device)
        peak_max = reduce_scalar(local_peak, op="max", device=device)
        assert abs(peak_max - peak_min) < 1e-6

        twice_synced = sync(synced_profiler)
        assert twice_synced is synced_profiler

        profiler_with_mark, _ = synced_profiler.mark("after_sync")
        assert not profiler_with_mark.is_fully_synced
        synced_with_mark = sync(profiler_with_mark)
        assert synced_with_mark.is_fully_synced
        assert len(synced_with_mark.checkpoints) == 1
        assert synced_with_mark.checkpoints[0].name == "after_sync"
        assert synced_with_mark._synced_checkpoints == len(synced_with_mark.checkpoints)
    finally:
        _cleanup_ddp()
