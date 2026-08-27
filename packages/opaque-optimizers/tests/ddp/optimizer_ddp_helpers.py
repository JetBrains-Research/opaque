"""CPU/Gloo workers for functional optimizer distributed-state tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from opaque_test_support import (
    cleanup_process_group as _cleanup_gloo,
)
from opaque_test_support import (
    setup_gloo as _setup_gloo,
)
from opaque_test_support import (
    spawn as _spawn,
)

_spawn_gloo = _spawn


def _worker_optimizer_state_audit_gloo(rank: int, world_size: int, port: int) -> None:
    from opaque.api.optimizers._adam import AdamState
    from opaque.api.optimizers._lion import LionState
    from opaque.api.optimizers.distributed import sync_optimizer_state
    from opaque.distributed import sync

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
