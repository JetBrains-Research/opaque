"""DDP runtime support for :class:`opaque.transformers.trainer.DPTrainer`.

Owns the rank/world resolution, the per-process gating for I/O sites
(logging, saving, hub push), and the small collection of cross-rank
collectives the trainer's training loop needs at the points described in
Phase 10 of ``docs/development/dp_training_arguments_plan.md``.

The trainer never calls ``torch.distributed.init_process_group``; the
launcher (``torchrun`` or test-side ``mp.spawn``) owns that. We just read
``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` from the environment for the
``__init__`` device pick, and use :mod:`opaque.distributed` once the
process group is up.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import torch

from opaque.distributed import get_rank, get_world_size, is_distributed
from opaque.api.engine.distributed._state import reduce_scalar
from opaque.distributed.collectives import barrier as _opaque_barrier

__all__ = [
    "DDPState",
    "resolve_ddp_state",
    "validate_ddp_backend",
    "should_log",
    "should_save",
    "barrier",
]

_BACKEND_FIRST_CLASS = {"nccl", "gloo", "mpi"}
_BACKEND_ENV_DEPENDENT_HINTS = {
    "xccl": "Intel XPU/XCCL runtime",
    "hccl": "Habana Gaudi/HCCL runtime",
    "cncl": "Cambricon MLU/CNCL runtime",
    "mccl": "Moore Threads MUSA/MCCL runtime",
}


@dataclasses.dataclass(frozen=True)
class DDPState:
    """Snapshot of the rank/world topology for one DPTrainer instance.

    ``is_distributed`` reflects ``torch.distributed.is_initialized()`` at
    construction time. When it's ``False``, the trainer behaves
    identically to the pre-DDP single-process path.

    The ``device`` field is the device this rank's model parameters live
    on. For NCCL/CUDA, that's ``cuda:{local_rank}``; for the
    single-process / CPU path it's whatever
    :class:`TrainingArguments._setup_devices` resolved.
    """

    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: torch.device

    @property
    def is_world_zero(self) -> bool:
        return self.rank == 0

    @property
    def is_local_zero(self) -> bool:
        return self.local_rank == 0


def resolve_ddp_state(device: torch.device) -> DDPState:
    """Build a :class:`DDPState` for the current process.

    If ``torch.distributed`` has been initialised, the rank/world fields
    come from the live process group. Otherwise we fall back to env vars
    (so a 1-rank ``torchrun`` smoke test still reports
    ``world_size==1`` rather than guessing from ``LOCAL_RANK``).
    """
    if is_distributed():
        rank = get_rank()
        world_size = get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = torch.distributed.get_backend()
        return DDPState(
            is_distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            device=device,
        )
    # No process group — single-process. Honour LOCAL_RANK if set so a
    # 1-rank torchrun reports the same shape as 1-rank no-launcher.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return DDPState(
        is_distributed=False,
        rank=0,
        local_rank=local_rank if local_rank >= 0 else 0,
        world_size=1,
        backend=None,
        device=device,
    )


def validate_ddp_backend(args: Any, ddp: DDPState) -> None:
    """Validate configured backend against active process-group backend.

    ``args.ddp_backend`` mirrors HF surface values.  If a process group is
    already initialized, the configured value (when set) must match the live
    backend to avoid silent misconfiguration.
    """
    configured = getattr(args, "ddp_backend", None)
    if configured is None:
        return
    configured_backend = str(configured).lower()
    if not ddp.is_distributed:
        if configured_backend in _BACKEND_ENV_DEPENDENT_HINTS:
            raise ValueError(
                f"ddp_backend={configured_backend!r} requires a distributed process "
                f"group initialized with vendor runtime "
                f"({_BACKEND_ENV_DEPENDENT_HINTS[configured_backend]}), but no "
                "process group is currently initialized."
            )
        return
    live_backend = (ddp.backend or "").lower()
    if live_backend and configured_backend != live_backend:
        raise ValueError(
            "Configured ddp_backend does not match initialized process group: "
            f"ddp_backend={configured_backend!r}, live_backend={live_backend!r}."
        )
    if (
        configured_backend not in _BACKEND_FIRST_CLASS
        and configured_backend in _BACKEND_ENV_DEPENDENT_HINTS
        and live_backend != configured_backend
    ):
        raise ValueError(
            f"ddp_backend={configured_backend!r} requires "
            f"{_BACKEND_ENV_DEPENDENT_HINTS[configured_backend]}, but the active "
            f"backend is {live_backend!r}."
        )


def should_log(args: Any, ddp: DDPState) -> bool:
    """Return ``True`` if this rank should emit log payloads.

    Mirrors HF's ``TrainingArguments.should_log`` (training_args.py:1992-2002):
    if ``log_on_each_node`` is set the gate fires on each node's local zero
    (typical multi-node default); otherwise only world-rank-0 logs.
    """
    if not ddp.is_distributed:
        return True
    if getattr(args, "log_on_each_node", True):
        return ddp.is_local_zero
    return ddp.is_world_zero


def should_save(args: Any, ddp: DDPState) -> bool:
    """Return ``True`` if this rank should write checkpoint / artefact files.

    Mirrors HF's ``TrainingArguments.should_save`` (training_args.py:2009-2015).
    Default ``save_on_each_node=False`` => only world-rank-0 writes; under
    ``save_on_each_node=True`` each node's local zero writes.
    """
    if not ddp.is_distributed:
        return True
    if getattr(args, "save_on_each_node", False):
        return ddp.is_local_zero
    return ddp.is_world_zero


def barrier(ddp: DDPState) -> None:
    """Block until every rank reaches this point.

    No-op outside a process group.
    """
    if ddp.is_distributed:
        _opaque_barrier()


def reduce_step_finite(grads_finite: bool, ddp: DDPState) -> bool:
    """Cluster-wide AND on ``grads_finite``.

    fp16 overflow on **any** rank must trip every rank to skip the
    optimizer update — otherwise rank A applies an update with sane grads
    while rank B no-ops, and the parameter trees diverge instantly.
    Implemented as ``min`` reduction on a 0/1 int (``min`` of any zero is
    zero — i.e. "any rank saw an inf" wins).
    """
    if not ddp.is_distributed:
        return grads_finite
    return bool(reduce_scalar(int(grads_finite), op="min", device=ddp.device))
