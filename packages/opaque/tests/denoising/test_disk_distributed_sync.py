"""Tests for DiskDenoiserState distributed sync registration."""

import torch

from opaque.denoising import disk_denoiser
from opaque.distributed import sync


def test_sync_disk_denoiser_state_single_process_no_op():
    """Without torch.distributed, sync() returns state unchanged."""
    template = torch.zeros(3)
    denoise, st = disk_denoiser(
        template, noise_stddev=1.0, process_stddev=0.1, dtype=torch.float64
    )
    _, st = denoise(torch.ones(3), st)
    out = sync(st)
    assert out._step_counter == st._step_counter
    assert torch.equal(out._estimate, st._estimate)
