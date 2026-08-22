"""CPU/Gloo workers for functional optimizer distributed-state tests."""

from __future__ import annotations

import os
import socket
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _cleanup_gloo() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn_gloo(world_size: int, fn, *args) -> None:
    mp.spawn(
        fn,
        args=(world_size, _find_free_port(), *args),
        nprocs=world_size,
        join=True,
    )


def _worker_optimizer_state_audit_gloo(rank: int, world_size: int, port: int) -> None:
    from opaque.api.optimizers._distributed import sync_optimizer_state
    from opaque.distributed import sync
    from opaque.optimizers.types import AdamState, LionState

    _setup_gloo(rank, world_size, port)
    try:
        adam = AdamState(
            mu={"weight": torch.tensor([1.0, -2.0])},
            nu={"weight": torch.tensor([3.0, 4.0])},
            phi={"weight": 0.25},
            step=7,
        )
        lion = LionState(m={"weight": torch.tensor([0.5, -0.5])}, step=7)

        assert sync(adam) is adam
        audited_chain = sync_optimizer_state((adam, lion))
        assert audited_chain == (adam, lion)

        mismatched = replace(adam, phi={"weight": float(rank)})
        with pytest.raises(RuntimeError, match=r"AdamState\.phi"):
            sync(mismatched)

        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_gloo()
