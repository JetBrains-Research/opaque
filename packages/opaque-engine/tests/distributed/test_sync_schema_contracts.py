"""Two-rank Gloo regressions for schema-derived synchronization schedules."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from tests._support.torch_distributed import (
    _spawn_gloo,
    _worker_sync_schema_contracts_gloo,
)


@pytest.mark.slow
@pytest.mark.distributed
def test_sync_schema_contracts_preserve_collective_parity() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_sync_schema_contracts_gloo)
