"""CPU gloo regression: empty-batch sync(aux) must not desync ranks."""

from __future__ import annotations

import torch.distributed as dist
from engine_ddp_helpers import _spawn_gloo, _worker_sync_aux_empty_batch


def test_sync_aux_empty_batch_does_not_desync() -> None:
    if not dist.is_available():
        import pytest

        pytest.skip("torch.distributed is not available")
    _spawn_gloo(2, _worker_sync_aux_empty_batch)
