"""Minimal NCCL DDP helpers + mp.spawn entrypoints (must live in this module for pickle)."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass(frozen=True)
class _ScalarExactnessState:
    value: int


@dataclass(frozen=True)
class _CoreGlooState:
    total: int
    average: float
    minimum: int
    maximum: int
    product: int
    local_rank: int


@dataclass(frozen=True)
class _DuckTypedPerGroup:
    groups: dict[str, str]
    values: dict[str, float]


def _is_ddp_available() -> bool:
    return dist.is_available() and torch.cuda.is_available()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup_ddp(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.backend import ensure_backend

    if not _is_ddp_available():
        raise RuntimeError("DDP requires CUDA and torch.distributed support")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    ensure_backend(torch.empty(0, device=f"cuda:{rank}"))


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
        value = float(rank + 1)
        synced = reduce_scalar(value, op="mean")
        expected_avg = sum(range(1, world_size + 1)) / world_size
        assert abs(synced - expected_avg) < 1e-5
    finally:
        _cleanup_ddp()


def _worker_all_reduce_values(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.collectives import all_reduce

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        base = torch.tensor([float(rank + 1), float(2 * (rank + 1))], device=device)

        result = all_reduce(base, op="sum")
        assert torch.allclose(
            base,
            torch.tensor([float(rank + 1), float(2 * (rank + 1))], device=device),
        )
        assert torch.allclose(result, torch.tensor([3.0, 6.0], device=device))

        averaged = all_reduce(base, op="mean")
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
    from opaque.distributed.gradients import reduce_pytree

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
    from opaque.profiling import step_perf
    from opaque.profiling.types import PerfState

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
        peak_min = reduce_scalar(local_peak, op="min")
        peak_max = reduce_scalar(local_peak, op="max")
        assert abs(peak_max - peak_min) < 1e-6
    finally:
        _cleanup_ddp()


def _setup_gloo(rank: int, world_size: int, port: int) -> None:
    """CPU process-group init for empty-batch collective-parity tests."""
    from opaque.api.engine.backend import ensure_backend

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    ensure_backend(torch.empty(0))


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


def _worker_in_place_wrapper_reduction_gloo(
    rank: int, world_size: int, port: int
) -> None:
    from opaque.torch.distributed import all_reduce_, reduce_pytree_, sum_gradients_
    from opaque.types import ClippedPytree, PerGroup, SecondMomentClippingOutput

    _setup_gloo(rank, world_size, port)
    try:
        tensor = torch.tensor([1.0, 2.0]) * (rank + 1)
        all_reduce_(tensor, op="sum")
        assert torch.allclose(tensor, torch.tensor([3.0, 6.0]))

        scale = 1.0 if rank == 0 else 10.0
        gradients = ClippedPytree({"w": torch.tensor([1.0, 2.0]) * scale}, max_norm=0.5)
        assert sum_gradients_(gradients) is None
        assert torch.allclose(gradients.pytree["w"], torch.tensor([11.0, 22.0]))
        assert abs(gradients.max_norm - 0.5) < 1e-6

        out = SecondMomentClippingOutput(
            grads=ClippedPytree({"w": torch.tensor([1.0]) * scale}, max_norm=0.5),
            squared_grads=ClippedPytree(
                {"w": torch.tensor([2.0]) * scale}, max_norm=1.0
            ),
        )
        assert sum_gradients_(out) is None
        assert torch.allclose(out.grads.pytree["w"], torch.tensor([11.0]))
        assert torch.allclose(out.squared_grads.pytree["w"], torch.tensor([22.0]))

        # An in-place mean would rescale ClippedPytree.max_norm; the
        # in-place API refuses metadata-changing reductions.
        with pytest.raises(TypeError, match="would change metadata"):
            reduce_pytree_(gradients, op="mean")

        # A PerGroup bound must survive the in-place path.  ``PerGroup``
        # stores its mappings as ``MappingProxyType``, which no duck-typed
        # ``isinstance(..., dict)`` check matches and no object-gather can
        # pickle, so a per-group bound that reaches the scalar branch aborts
        # the step with ``TypeError: cannot pickle 'mappingproxy' object``.
        per_group_norm = PerGroup(groups={"w": "weights"}, values={"weights": 1.0})
        per_group_grads = ClippedPytree(
            {"w": torch.tensor([1.0, 2.0]) * scale}, max_norm=per_group_norm
        )
        assert sum_gradients_(per_group_grads) is None
        assert torch.allclose(per_group_grads.pytree["w"], torch.tensor([11.0, 22.0]))
        assert per_group_grads.max_norm == per_group_norm

        # ...and the equality check must still fire on a real divergence,
        # per group and on the metadata kind itself.
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.values\['weights'\] mismatch",
        ):
            sum_gradients_(
                ClippedPytree(
                    {"w": torch.tensor([1.0])},
                    max_norm=PerGroup(
                        groups={"w": "weights"},
                        values={"weights": float(rank + 1)},
                    ),
                )
            )

        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.kind mismatch",
        ):
            sum_gradients_(
                ClippedPytree(
                    {"w": torch.tensor([1.0])},
                    max_norm=per_group_norm if rank == 0 else 1.0,
                )
            )
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
        assert len(x) % world_size == 0
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


def _worker_core_collectives_gloo(rank: int, world_size: int, port: int) -> None:
    """Exercise backend-neutral engine primitives through a live Gloo group."""
    from opaque.api.engine.distributed._state import gather_tensors, sync_object
    from opaque.distributed import (
        all_reduce,
        barrier,
        gather_for_metrics,
        get_rank,
        get_world_size,
        is_distributed,
        is_main_process,
        num_processes,
        process_index,
        sync,
    )
    from opaque.distributed.gradients import reduce_pytree
    from opaque.profiling import step_perf
    from opaque.profiling.types import PerfState
    from opaque.torch.distributed import all_reduce_
    from opaque.types import ClippedPytree, NoisedPytree, PerGroup

    _setup_gloo(rank, world_size, port)
    try:
        assert is_distributed()
        assert get_rank() == process_index() == rank
        assert get_world_size() == num_processes() == world_size
        assert is_main_process() is (rank == 0)

        base = torch.tensor([float(rank + 1), float(2 * (rank + 1))])
        expected = {
            "sum": torch.tensor([3.0, 6.0]),
            "mean": torch.tensor([1.5, 3.0]),
            "max": torch.tensor([2.0, 4.0]),
            "min": torch.tensor([1.0, 2.0]),
            "product": torch.tensor([2.0, 8.0]),
        }
        for op, value in expected.items():
            torch.testing.assert_close(all_reduce(base, op=op), value)
        torch.testing.assert_close(
            base, torch.tensor([float(rank + 1), float(2 * (rank + 1))])
        )

        inplace = base.clone()
        assert all_reduce_(inplace, op="mean") is None
        torch.testing.assert_close(inplace, expected["mean"])

        gathered_scalars = gather_for_metrics(torch.tensor(float(rank)))
        torch.testing.assert_close(gathered_scalars, torch.tensor([0.0, 1.0]))
        gathered_ragged = gather_tensors(torch.full((rank + 1, 1), float(rank)))
        torch.testing.assert_close(
            gathered_ragged,
            torch.tensor([[0.0], [1.0], [1.0]]),
        )

        local_tree = {
            "weight": torch.tensor([float(rank + 1)]),
            "bias": torch.tensor([float(rank)]),
        }
        reduced_tree = reduce_pytree(local_tree, op="sum")
        torch.testing.assert_close(reduced_tree["weight"], torch.tensor([3.0]))
        torch.testing.assert_close(reduced_tree["bias"], torch.tensor([1.0]))
        torch.testing.assert_close(
            local_tree["weight"], torch.tensor([float(rank + 1)])
        )

        clipped = ClippedPytree({"w": torch.tensor([float(rank + 1)])}, max_norm=2.0)
        averaged = reduce_pytree(clipped, op="mean")
        assert isinstance(averaged, ClippedPytree)
        assert averaged.max_norm == 1.0
        torch.testing.assert_close(averaged.pytree["w"], torch.tensor([1.5]))

        noised = NoisedPytree(
            {"w": torch.tensor([float(rank + 1)])},
            max_norm=2.0,
            noise_stddev=0.5,
        )
        summed_noised = reduce_pytree(noised, op="sum")
        assert isinstance(summed_noised, NoisedPytree)
        assert summed_noised.max_norm == 2.0
        assert summed_noised.noise_stddev == pytest.approx(0.5 * 2**0.5)
        torch.testing.assert_close(summed_noised.pytree["w"], torch.tensor([3.0]))

        per_group_norm = PerGroup(
            groups={"w": "weights"},
            values={"weights": 1.0},
        )
        per_group_reduced = reduce_pytree(
            ClippedPytree(
                {"w": torch.tensor([float(rank + 1)])},
                max_norm=per_group_norm,
            ),
            op="sum",
        )
        assert per_group_reduced.max_norm == per_group_norm
        torch.testing.assert_close(per_group_reduced.pytree["w"], torch.tensor([3.0]))

        mismatched_per_group_norm = PerGroup(
            groups=per_group_norm.groups,
            values={"weights": float(rank + 1)},
        )
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.values\['weights'\] mismatch",
        ):
            reduce_pytree(
                ClippedPytree(
                    {"w": torch.tensor([float(rank + 1)])},
                    max_norm=mismatched_per_group_norm,
                ),
                op="sum",
            )

        reordered_mismatch = PerGroup(
            groups={"w": "weights", "b": "biases"},
            values=(
                {"weights": 1.0, "biases": 2.0}
                if rank == 0
                else {"biases": 1.0, "weights": 2.0}
            ),
        )
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.values\['(?:biases|weights)'\] mismatch",
        ):
            reduce_pytree(
                ClippedPytree(
                    {
                        "w": torch.tensor([float(rank + 1)]),
                        "b": torch.tensor([float(rank + 1)]),
                    },
                    max_norm=reordered_mismatch,
                ),
                op="sum",
            )

        reordered_duck_typed_metadata = _DuckTypedPerGroup(
            groups={"w": "weights", "b": "biases"},
            values=(
                {"weights": 1.0, "biases": 2.0}
                if rank == 0
                else {"biases": 1.0, "weights": 2.0}
            ),
        )
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.values\['(?:biases|weights)'\] mismatch",
        ):
            reduce_pytree(
                ClippedPytree(
                    {
                        "w": torch.tensor([float(rank + 1)]),
                        "b": torch.tensor([float(rank + 1)]),
                    },
                    max_norm=reordered_duck_typed_metadata,
                ),
                op="sum",
            )

        mixed_metadata = per_group_norm if rank == 0 else 1.0
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.kind mismatch",
        ):
            reduce_pytree(
                ClippedPytree(
                    {"w": torch.tensor([float(rank + 1)])},
                    max_norm=mixed_metadata,
                ),
                op="sum",
            )

        synced = sync_object(
            _CoreGlooState(
                total=rank + 1,
                average=float(rank + 1),
                minimum=rank + 1,
                maximum=rank + 1,
                product=rank + 1,
                local_rank=rank,
            ),
            {
                "total": "sum",
                "average": "mean",
                "minimum": "min",
                "maximum": "max",
                "product": "product",
                "local_rank": "local",
            },
        )
        assert synced == _CoreGlooState(
            total=3,
            average=1.5,
            minimum=1,
            maximum=2,
            product=2,
            local_rank=rank,
        )

        perf_state = PerfState(torch.device("cpu"))
        with step_perf(torch.device("cpu"), batch_size=rank + 1) as step:
            _ = torch.arange(64, dtype=torch.float32).square().sum()
        perf_state = perf_state.add(step.perf)
        synced_perf_state = sync(perf_state)
        assert synced_perf_state is not perf_state
        assert synced_perf_state.num_steps == 1
        assert synced_perf_state.total_samples == 3
        assert synced_perf_state.last_step is not None
        assert synced_perf_state.last_step.batch_size == 3
        assert synced_perf_state.last_step.step_time_sec >= 0.0

        barrier()
        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_ddp()


def _worker_gather_optional_ragged(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import gather_pytree, gather_tensors

    _setup_gloo(rank, world_size, port)
    try:
        optional = (
            None if rank == 0 else {"value": torch.tensor([[1.0, 2.0]]), "aux": None}
        )
        gathered_optional = gather_pytree(optional)
        assert torch.equal(
            gathered_optional["value"],
            torch.tensor([[1.0, 2.0]]),
        )
        assert gathered_optional["aux"] is None

        ragged = torch.full((rank * 2, 2), float(rank))
        gathered_ragged = gather_tensors(ragged)
        assert torch.equal(
            gathered_ragged,
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        )

        incompatible_dtype = (
            torch.ones((1, 2), dtype=torch.float32)
            if rank == 0
            else torch.ones((1, 2), dtype=torch.int64)
        )
        with pytest.raises(TypeError, match="matching dtypes"):
            gather_tensors(incompatible_dtype)

        incompatible_shape = (
            torch.ones((1, 2), dtype=torch.float32)
            if rank == 0
            else torch.ones((1, 3), dtype=torch.float32)
        )
        with pytest.raises(ValueError, match="non-concatenated dimensions"):
            gather_tensors(incompatible_shape)

        incompatible_rank = (
            torch.ones((1, 2), dtype=torch.float32)
            if rank == 0
            else torch.ones((1, 2, 1), dtype=torch.float32)
        )
        with pytest.raises(ValueError, match="matching tensor ranks"):
            gather_tensors(incompatible_rank)

        incompatible_structure = (
            {"value": torch.ones((1, 2))} if rank == 0 else [torch.ones((1, 2))]
        )
        with pytest.raises(TypeError, match="matching pytree structures"):
            gather_pytree(incompatible_structure)
    finally:
        _cleanup_ddp()


@dataclass(frozen=True)
class _RareEventState:
    """A mechanism author's own state, shaped like the documented example."""

    hits: int
    examples: int
    threshold: float
    seed: int


def _worker_custom_sync_seam_gloo(rank: int, world_size: int, port: int) -> None:
    """The public seam a new mechanism uses to cross ranks.

    Mirrors ``docs/user-guide/distributed.md``: describe each field, register
    the type, then ``sync()`` finds it — including the fail-closed behaviour
    that keeps state from crossing ranks unsynchronized by omission.
    """
    from opaque.distributed import register_sync_type, sync, sync_object

    _setup_gloo(rank, world_size, port)
    try:

        def _sync_rare_event(state: _RareEventState) -> _RareEventState:
            return sync_object(
                state,
                field_ops={
                    "hits": "sum",
                    "examples": "sum",
                    "threshold": "assert_equal",
                    "seed": "local",
                },
            )

        # Unregistered types are refused rather than passed through.
        with pytest.raises(TypeError, match="No sync function registered"):
            sync(_RareEventState(hits=1, examples=2, threshold=0.5, seed=rank))

        register_sync_type(_RareEventState, _sync_rare_event)

        synced = sync(
            _RareEventState(hits=rank + 1, examples=10, threshold=0.5, seed=rank)
        )
        assert synced.hits == 3
        assert synced.examples == 2 * 10
        assert synced.threshold == 0.5
        # ``local`` keeps a rank-local field rank-local, on purpose.
        assert synced.seed == rank

        # A field that must agree and does not is an error, not a reduction.
        with pytest.raises(RuntimeError):
            sync(_RareEventState(hits=1, examples=1, threshold=float(rank), seed=rank))
    finally:
        _cleanup_ddp()


def _worker_scalar_exactness_gloo(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import (
        assert_scalar_equal,
        reduce_scalar,
        sync_object,
    )

    _setup_gloo(rank, world_size, port)
    try:
        integer_value = 2**24 + rank
        assert reduce_scalar(integer_value, op="min") == 2**24
        assert reduce_scalar(integer_value, op="max") == 2**24 + 1
        assert reduce_scalar(integer_value, op="sum") == 2 * 2**24 + 1
        # Integer means come from the exact int64 sum divided in Python
        # (float64); float32 tensor division would round the .5 away.
        mean_value = reduce_scalar(integer_value, op="mean")
        assert isinstance(mean_value, float)
        assert mean_value == (2 * 2**24 + 1) / 2

        float_value = float(rank)
        assert reduce_scalar(float_value, op="max") == 1.0

        # A Python float is a float64 and must reach the wire as one.  Both
        # of these fail if the reduction narrows to the framework's default
        # dtype: the mean rounds to 3.1415927410125732 at float32 and to
        # 3.140625 at bfloat16, and the sum cannot represent 2e17 at float32.
        precise = 3.14159265358979
        assert reduce_scalar(precise, op="mean") == precise
        assert reduce_scalar(1e17 + 1.0, op="sum") == (1e17 + 1.0) * world_size

        # ...including when a process-global default says otherwise.  This is
        # a legitimate setting in low-precision training, and it must not
        # reach a DP-relevant scalar.
        previous_default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            assert reduce_scalar(precise, op="mean") == precise
        finally:
            torch.set_default_dtype(previous_default)

        with pytest.raises(RuntimeError, match="integer"):
            assert_scalar_equal(integer_value, name="integer")
        with pytest.raises(RuntimeError, match="float"):
            assert_scalar_equal(
                float_value,
                name="float",
                atol=0.0,
                rtol=0.0,
            )
        with pytest.raises(RuntimeError, match=r"_ScalarExactnessState\.value"):
            sync_object(
                _ScalarExactnessState(integer_value),
                field_ops={"value": "assert_equal"},
            )
    finally:
        _cleanup_ddp()


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


def _worker_sync_schema_contracts_gloo(rank: int, world_size: int, port: int) -> None:
    """Exercise schema mismatches without leaving Gloo ranks desynchronized."""
    from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
    from opaque.api.engine.clipping._distributed import sync_clipped_grad_aux
    from opaque.api.engine.distributed._state import reduce_scalar, sync_object
    from opaque.api.engine.profiling._distributed import (
        sync_perf_state,
        sync_perf_tracker,
    )
    from opaque.profiling.types import PerfState, PerfTracker, StepPerf

    @dataclass(frozen=True)
    class _CompatibleAux(ClippedGradAux):
        pass

    @dataclass(frozen=True)
    class _UnsupportedAux(ClippedGradAux):
        extra: torch.Tensor | None = None

    _setup_gloo(rank, world_size, port)
    try:
        aux = _CompatibleAux(
            loss_values=torch.tensor([float(rank)]),
            grad_norms=torch.tensor([1.0]),
            clipped_grad_norms=torch.tensor([1.0]),
            clipping_rate=0.0,
            batch_size=1,
        )
        synced_aux = sync_clipped_grad_aux(aux)
        assert type(synced_aux) is _CompatibleAux
        assert synced_aux.batch_size == world_size

        with pytest.raises(TypeError, match="synchronization schema"):
            sync_clipped_grad_aux(_UnsupportedAux(extra=torch.tensor([float(rank)])))

        with pytest.raises(RuntimeError, match="clipping_rate presence mismatch"):
            sync_clipped_grad_aux(
                ClippedGradAux(
                    grad_norms=torch.empty(0),
                    clipped_grad_norms=torch.empty(0),
                    clipping_rate=None if rank == 0 else 0.0,
                    batch_size=0,
                )
            )

        ordered_tracker = PerfTracker(torch.device("cpu"), warmup_steps=0)
        for name in ("train", "eval") if rank == 0 else ("eval", "train"):
            stage = ordered_tracker[name]
            stage.num_steps = 1
        synced_tracker = sync_perf_tracker(ordered_tracker)
        assert tuple(synced_tracker.stages) == ("eval", "train")

        mismatched_tracker = PerfTracker(torch.device("cpu"), warmup_steps=0)
        mismatched_tracker["train" if rank == 0 else "eval"].num_steps = 1
        with pytest.raises(RuntimeError, match="stage schema mismatch"):
            sync_perf_tracker(mismatched_tracker)

        state = PerfState(
            device=torch.device("cpu"),
            last_step=(
                None if rank == 0 else StepPerf(step_time_sec=1.0, batch_size=1)
            ),
        )
        with pytest.raises(RuntimeError, match="StepPerf presence mismatch"):
            sync_perf_state(state)

        # A field callable that fails after reducing must reduce exactly once.
        # The TypeError below is raised by the callback body, not by a signature
        # mismatch, so retrying the call would issue the collective twice and
        # desynchronise the ranks.
        reductions: list[int] = []

        def _reduce_then_fail(value: int) -> int:
            reductions.append(reduce_scalar(value, op="sum"))
            raise TypeError("field callable failed after a collective")

        with pytest.raises(TypeError, match="failed after a collective"):
            sync_object(
                _ScalarExactnessState(1), field_ops={"value": _reduce_then_fail}
            )
        assert reductions == [world_size]

        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_ddp()


def _worker_cold_process_group_query_gloo(
    rank: int, world_size: int, port: int
) -> None:
    """Query a live group from a process that never imported the provider.

    Backend selection is value-driven, so a rank query can be the first
    Opaque call a process makes. Nothing here touches a tensor before the
    query, and nothing imports ``opaque.torch``: the engine has to find the
    group on its own, and hand the scalar collectives that follow a
    positive answer a backend to dispatch on.
    """
    import sys

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from opaque.api.engine.backend import active_backend
        from opaque.api.engine.distributed._state import (
            assert_scalar_equal,
            reduce_scalar,
        )
        from opaque.distributed import (
            barrier,
            get_rank,
            get_world_size,
            is_distributed,
            is_main_process,
        )

        assert "opaque.api.torch" not in sys.modules
        assert active_backend() is None

        assert is_distributed() is True
        assert get_rank() == rank
        assert get_world_size() == world_size
        assert is_main_process() is (rank == 0)

        assert active_backend() is not None
        assert active_backend().name == "torch"

        assert reduce_scalar(1, op="sum") == world_size
        assert_scalar_equal(7, name="agreed")
        if world_size > 1:
            with pytest.raises(RuntimeError, match="mismatch across ranks"):
                assert_scalar_equal(rank, name="disagreed")
        barrier()
    finally:
        dist.destroy_process_group()
