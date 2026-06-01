"""CPU (non-distributed) behavior of the rank/gather helper extensions.

These wrap ``torch.distributed`` but must degrade gracefully when no
process group is initialized: rank 0, world size 1, no-op barrier, and
``gather_for_metrics`` returning its input unchanged. The multi-rank
collective behavior is exercised by the CUDA tests in
``test_collectives.py``.
"""

from __future__ import annotations

import torch

from opaque.distributed import (
    gather_for_metrics,
    is_main_process,
    num_processes,
    process_index,
    wait_for_everyone,
)


def test_is_main_process_non_distributed() -> None:
    assert is_main_process() is True


def test_num_processes_non_distributed() -> None:
    assert num_processes() == 1


def test_process_index_non_distributed() -> None:
    assert process_index() == 0


def test_wait_for_everyone_is_noop_non_distributed() -> None:
    # Must not raise when there is no process group.
    assert wait_for_everyone() is None


def test_gather_for_metrics_returns_input_non_distributed() -> None:
    x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    out = gather_for_metrics(x)
    assert out is x
    assert torch.equal(out, x)
