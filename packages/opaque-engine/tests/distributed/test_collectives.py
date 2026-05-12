"""Distributed collectives on NCCL (engine substrate)."""

from __future__ import annotations

import pytest
import torch

from opaque.api.engine.distributed._state import reduce_scalar

from ._ddp_helpers import _cleanup_ddp, _setup_ddp, _spawn


pytestmark = pytest.mark.cuda


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


class TestDistributedCollectives:
    def test_reduce_scalar(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_scalar)

    def test_all_reduce_values(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_all_reduce_values)

    def test_reduce_pytree(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_pytree)

    def test_reduce_pytree_nested(self) -> None:
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        _spawn(2, _worker_reduce_pytree_nested)
