"""CPU gloo regression: empty-batch sync(aux) must not desync ranks."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from engine_ddp_helpers import (
    _spawn_gloo,
    _worker_sync_aux_empty_batch,
    _worker_sync_aux_empty_vs_per_group,
)


def _require_gloo() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")


@pytest.mark.slow
def test_sync_aux_empty_batch_does_not_desync() -> None:
    _require_gloo()
    _spawn_gloo(2, _worker_sync_aux_empty_batch)


@pytest.mark.slow
def test_sync_aux_empty_vs_per_group_group_norms() -> None:
    """Empty rank omits group_norms; nonempty has a per-group dict."""
    _require_gloo()
    _spawn_gloo(2, _worker_sync_aux_empty_vs_per_group)
