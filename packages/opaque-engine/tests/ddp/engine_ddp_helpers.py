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
    from opaque.profiling import PerfState, step_perf

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        state = PerfState(device=device)

        with step_perf(device, batch_size=4) as perf:
            x = torch.randn(1024 + 512 * rank, device=device)
            _ = (x * x).sum()
        state = state.add(perf.result)

        synced_state = sync(state)
        assert synced_state is not state
        assert synced_state.num_steps == 1
        assert synced_state.last_step.batch_size == world_size * 4
        assert state.last_step.batch_size == 4
        assert synced_state.last_step.step_time_sec >= 0.0

        local_peak = float(synced_state.max_peak_memory_gb)
        peak_min = reduce_scalar(local_peak, op="min", device=device)
        peak_max = reduce_scalar(local_peak, op="max", device=device)
        assert abs(peak_max - peak_min) < 1e-6
    finally:
        _cleanup_ddp()


def _setup_gloo(rank: int, world_size: int, port: int) -> None:
    """CPU process-group init for empty-batch collective-parity tests."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _worker_second_moment_clip_gloo(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.gradients import reduce_pytree
    from opaque.types import ClippedPytree, SecondMomentClippingOutput

    _setup_gloo(rank, world_size, port)
    try:
        scale = 1.0 if rank == 0 else 10.0
        out = SecondMomentClippingOutput(
            grads=ClippedPytree({"w": torch.tensor([1.0, 2.0]) * scale}, max_norm=1.0),
            squared_grads=ClippedPytree(
                {"w": torch.tensor([3.0]) * scale}, max_norm=2.0
            ),
        )
        reduced = reduce_pytree(out, op="sum")
        assert isinstance(reduced, SecondMomentClippingOutput)
        assert torch.allclose(reduced.grads.pytree["w"], torch.tensor([11.0, 22.0]))
        assert torch.allclose(reduced.squared_grads.pytree["w"], torch.tensor([33.0]))
        assert abs(reduced.grads.max_norm - 1.0) < 1e-6
        assert abs(reduced.squared_grads.max_norm - 2.0) < 1e-6
    finally:
        _cleanup_ddp()


def _worker_second_moment_noise_gloo(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.gradients import reduce_pytree
    from opaque.types import NoisedPytree, SecondMomentNoiseOutput

    _setup_gloo(rank, world_size, port)
    try:
        scale = 1.0 if rank == 0 else 10.0
        out = SecondMomentNoiseOutput(
            noisy_grads=NoisedPytree(
                {"w": torch.tensor([1.0]) * scale},
                max_norm=1.0,
                noise_stddev=0.5,
            ),
            noisy_squared_grads=NoisedPytree(
                {"w": torch.tensor([2.0]) * scale},
                max_norm=2.0,
                noise_stddev=0.25,
            ),
        )
        reduced = reduce_pytree(out, op="sum")
        assert isinstance(reduced, SecondMomentNoiseOutput)
        assert torch.allclose(reduced.noisy_grads.pytree["w"], torch.tensor([11.0]))
        assert torch.allclose(
            reduced.noisy_squared_grads.pytree["w"], torch.tensor([22.0])
        )
        assert abs(reduced.noisy_grads.noise_stddev - 0.5 * (2.0**0.5)) < 1e-6
        assert abs(reduced.noisy_squared_grads.noise_stddev - 0.25 * (2.0**0.5)) < 1e-6
    finally:
        _cleanup_ddp()


def _paired_clipping_fixture(
    device: torch.device | str,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    params = {
        "linear": {
            "weight": torch.tensor([0.25, -0.5, 0.75], device=device),
        },
        "bias": torch.tensor(0.1, device=device),
    }
    x = torch.arange(24, dtype=torch.float32, device=device).reshape(8, 3) / 10.0
    y = torch.linspace(-0.4, 0.6, 8, device=device)
    return params, x, y


def _paired_clipping_loss(
    params: dict, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    prediction = x @ params["linear"]["weight"] + params["bias"]
    return (prediction - y).square()


def _worker_second_moment_clipping_parity_gloo(
    rank: int,
    world_size: int,
    port: int,
    out_path: str,
) -> None:
    from opaque.api.engine.clipping import clipped_grad
    from opaque.distributed import sum_gradients
    from opaque.pytree import tree_map
    from opaque.types import SecondMomentClippingOutput

    _setup_gloo(rank, world_size, port)
    try:
        params, x, y = _paired_clipping_fixture("cpu")
        grad_fn, clip_state = clipped_grad(
            _paired_clipping_loss,
            clipping_norm=0.7,
            batch_argnums=(1, 2),
            normalize_by=len(x),
            second_moment=True,
        )
        shard_size = len(x) // world_size
        shard = slice(rank * shard_size, (rank + 1) * shard_size)
        local, _ = grad_fn(params, x[shard], y[shard], state=clip_state)
        reduced = sum_gradients(local)

        assert isinstance(reduced, SecondMomentClippingOutput)
        if rank == 0:
            torch.save(
                {
                    "grads": tree_map(
                        lambda tensor: tensor.cpu(), reduced.grads.pytree
                    ),
                    "squared_grads": tree_map(
                        lambda tensor: tensor.cpu(), reduced.squared_grads.pytree
                    ),
                    "max_norm": reduced.grads.max_norm,
                    "squared_max_norm": reduced.squared_grads.max_norm,
                },
                out_path,
            )
    finally:
        _cleanup_ddp()


def _spawn_gloo(world_size: int, fn, *args) -> None:
    port = _find_free_port()
    mp.spawn(fn, args=(world_size, port, *args), nprocs=world_size, join=True)


def _worker_sync_aux_empty_batch(rank: int, world_size: int, port: int) -> None:
    """Rank 0 draws an empty batch; rank 1 draws examples. Must not hang."""
    from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
    from opaque.api.engine.clipping._distributed import sync_clipped_grad_aux
    from opaque.distributed import sync

    _setup_gloo(rank, world_size, port)
    try:
        if rank == 0:
            aux = ClippedGradAux(
                loss_values=torch.empty(0),
                grad_norms=torch.empty(0),
                clipped_grad_norms=torch.empty(0),
                loss_aux=None,
                clipping_rate=0.0,
                batch_size=0,
                group_norms=None,
            )
        else:
            aux = ClippedGradAux(
                loss_values=torch.tensor([1.0, 2.0, 3.0]),
                grad_norms=torch.tensor([0.4, 1.2, 0.8]),
                clipped_grad_norms=torch.tensor([0.4, 1.0, 0.8]),
                loss_aux=None,
                clipping_rate=1.0 / 3.0,
                batch_size=3,
                group_norms=None,
            )

        synced = sync_clipped_grad_aux(aux)
        # Also exercise the type-dispatched sync path used by trainers.
        synced2 = sync(aux)

        assert synced.batch_size == 3
        assert synced2.batch_size == 3
        assert synced.grad_norms.shape[0] == 3
        assert abs(synced.clipping_rate - (1.0 / 3.0)) < 1e-5
        # A follow-up collective must still succeed (proves no desync).
        token = torch.tensor([float(rank + 1)])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert abs(token.item() - sum(range(1, world_size + 1))) < 1e-5
    finally:
        _cleanup_ddp()


def _worker_sync_aux_empty_vs_per_group(rank: int, world_size: int, port: int) -> None:
    """Empty rank has group_norms=None; nonempty has per-group dict.

    After ParamPath-keyed PerGroup, aux ``group_norms`` are still keyed by
    group name, but empty batches still omit the dict. Sync must not hang.
    """
    from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
    from opaque.api.engine.clipping._distributed import sync_clipped_grad_aux

    _setup_gloo(rank, world_size, port)
    try:
        if rank == 0:
            aux = ClippedGradAux(
                loss_values=torch.empty(0),
                grad_norms=torch.empty(0),
                clipped_grad_norms=torch.empty(0),
                loss_aux=None,
                clipping_rate=0.0,
                batch_size=0,
                group_norms=None,
            )
        else:
            aux = ClippedGradAux(
                loss_values=torch.tensor([1.0, 2.0]),
                grad_norms=torch.tensor([0.5, 1.5]),
                clipped_grad_norms=torch.tensor([0.5, 1.0]),
                loss_aux=None,
                clipping_rate=0.5,
                batch_size=2,
                group_norms={
                    "attn": torch.tensor([0.5, 1.5]),
                    "mlp": torch.tensor([0.2, 0.3]),
                },
            )

        synced = sync_clipped_grad_aux(aux)
        assert synced.batch_size == 2
        assert synced.group_norms is not None
        assert set(synced.group_norms) == {"attn", "mlp"}
        assert synced.group_norms["attn"].shape[0] == 2
        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert abs(token.item() - float(world_size)) < 1e-5
    finally:
        _cleanup_ddp()
