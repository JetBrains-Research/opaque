"""CPU/Gloo contract for DP-FTRL's backend-neutral distributed path."""

from __future__ import annotations

import pytest
import torch.distributed as dist
from dpftrl_ddp_helpers import _spawn_gloo, _worker_cpu_gloo_training_contract

pytestmark = pytest.mark.distributed


def test_gradient_reduction_and_mf_noise_handle_empty_ranks() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend is not available")
    _spawn_gloo(2, _worker_cpu_gloo_training_contract)
