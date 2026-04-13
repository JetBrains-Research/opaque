"""Multi-step distributed matrix factorization noise tests.

Tests verify that MF noise mechanisms produce:
1. Deterministic output across ranks when using shared seed
2. Proper noise correlation over multiple training steps
3. Cross-rank agreement on noise values

Note: BandMF and BLT have device placement limitations with streaming
matrices (coefficients kept on CPU). These tests focus on identity noise which
doesn't use streaming matrices and is fully functional on CUDA.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from opaque.noise.mf import mf_noise, identity_strategy


def _find_free_port() -> int:
    """Find a free port to avoid EADDRINUSE errors."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup_ddp(rank: int, world_size: int, port: int) -> None:
    """Initialize torch.distributed for testing."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl")


def _cleanup_ddp() -> None:
    """Clean up torch.distributed after testing."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Multi-step Identity MF tests (standard Gaussian, DP-SGD equivalent)
# ============================================================================


def _worker_identity_mf_three_steps(rank: int, world_size: int, port: int) -> None:
    """Worker for identity MF noise over 3 training steps.

    Identity MF is equivalent to standard DP-SGD (independent noise at each step).

    Verifies:
    1. Each step produces different noise
    2. All ranks produce same noise with shared seed
    3. Noise is standard Gaussian (mean ~0, std ~1)
    """
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        batch_size = 32  # Larger for better statistics
        param_dim = 64

        # Create template gradient
        grad_template = {"weight": torch.zeros(batch_size, param_dim, device=device)}

        # Initialize identity MF (standard Gaussian noise)
        noise_fn, state = mf_noise(
            grad_template, identity_strategy(), stddev=1.0, key=None
        )

        # Run 3 training steps
        step_noise_values = []
        step_stds = []
        for _step in range(3):
            grads = {"weight": torch.zeros(batch_size, param_dim, device=device)}
            noisy, state = noise_fn(grads, state)
            step_noise_values.append(noisy["weight"].clone())
            step_stds.append(noisy["weight"].std().item())

        # Verify each step produces different noise
        assert not torch.allclose(step_noise_values[0], step_noise_values[1])
        assert not torch.allclose(step_noise_values[1], step_noise_values[2])

        # Verify noise scales are reasonable (std should be ~1.0)
        for step_idx, std in enumerate(step_stds):
            assert 0.8 < std < 1.2, f"Step {step_idx}: std {std} out of range"

        # Verify all noise values are finite
        for step_idx, noise_val in enumerate(step_noise_values):
            assert torch.isfinite(noise_val).all(), f"Step {step_idx}: non-finite noise"

        # Verify cross-rank agreement
        for step_idx, noise_val in enumerate(step_noise_values):
            gathered = [torch.zeros_like(noise_val) for _ in range(world_size)]
            dist.all_gather(gathered, noise_val)
            if rank == 0:
                for other in gathered[1:]:
                    assert torch.allclose(gathered[0], other, atol=1e-6), (
                        f"Step {step_idx}: rank 0 and other ranks disagree"
                    )

    finally:
        _cleanup_ddp()


class TestIdentityMFMultiStep:
    """Multi-step identity MF noise tests."""

    def test_identity_mf_three_steps_cross_rank(self):
        """Identity MF noise should be deterministic across 3 steps and all ranks."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        port = _find_free_port()
        mp.spawn(_worker_identity_mf_three_steps, args=(2, port), nprocs=2, join=True)
