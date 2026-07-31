"""Multi-rank reduce for SecondMoment* wrappers (gloo/CPU)."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from engine_ddp_helpers import (
    _spawn,
    _worker_second_moment_clip_gloo,
    _worker_second_moment_noise_gloo,
)


class TestSecondMomentReduceGloo:
    def test_second_moment_clipping_sum(self) -> None:
        if not dist.is_available():
            pytest.skip("torch.distributed unavailable")
        _spawn(2, _worker_second_moment_clip_gloo)

    def test_second_moment_noise_sum(self) -> None:
        if not dist.is_available():
            pytest.skip("torch.distributed unavailable")
        _spawn(2, _worker_second_moment_noise_gloo)
