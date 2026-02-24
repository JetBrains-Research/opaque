"""Tests for distributed noise generation.

These tests verify that gaussian_stateful() with distributed=True properly
generates deterministic seed-based noise for each device.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from opaque.noise import gaussian_noise, identity_mf_noise
from opaque.random import key


class TestDistributedNoise:
    """Tests for gaussian_noise() behavior."""

    def test_distributed_false_matches_single_device(self):
        """distributed=False behaves same as no distributed parameter."""
        stddev = 1.0
        seed = 42

        # Create two noise functions (functional API)
        noise_fn1, state1 = gaussian_noise(stddev, key=key(seed))
        noise_fn2, state2 = gaussian_noise(stddev, key=key(seed))

        grads = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}

        noisy1, state1 = noise_fn1(grads, state1)
        noisy2, state2 = noise_fn2(grads, state2)

        # Should produce same noise
        assert torch.allclose(noisy1["weight"], noisy2["weight"])
        assert torch.allclose(noisy1["bias"], noisy2["bias"])

    def test_rank_fold_in_produces_distinct_streams(self):
        """Folding rank into key produces distinct but deterministic streams."""
        from opaque.random import fold_in

        # Simulate two ranks by folding different ranks into the same base key
        noise_fn1, state1 = gaussian_noise(1.0, key=fold_in(key(42), 0))
        noise_fn2, state2 = gaussian_noise(1.0, key=fold_in(key(42), 1))

        grads = {"weight": torch.zeros(4)}

        noisy1, state1 = noise_fn1(grads, state1)
        noisy2, state2 = noise_fn2(grads, state2)

        # Different ranks should produce different noise
        assert not torch.allclose(noisy1["weight"], noisy2["weight"])

    def test_distributed_noise_is_deterministic(self):
        """Noise with same seed+rank is reproducible across resets."""
        stddev = 1.1
        seed = 123

        # Create two separate noise functions with same seed
        noise_fn1, state1 = gaussian_noise(stddev, key=key(seed))
        noise_fn2, state2 = gaussian_noise(stddev, key=key(seed))

        grads = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}

        # Apply noise with both functions (reset generator)
        noisy1, state1 = noise_fn1(grads, state1)
        noisy2, state2 = noise_fn2(grads, state2)

        # Should produce same noise (deterministic from same seed)
        assert torch.allclose(noisy1["weight"], noisy2["weight"])
        assert torch.allclose(noisy1["bias"], noisy2["bias"])

    def test_distributed_false_different_seeds_produce_different_noise(self):
        """Different seeds produce different noise."""
        stddev = 1.0
        grads = {"weight": torch.randn(10, 5)}

        noise_fn1, state1 = gaussian_noise(stddev, key=key(42))
        noise_fn2, state2 = gaussian_noise(stddev, key=key(43))

        noisy1, state1 = noise_fn1(grads, state1)
        noisy2, state2 = noise_fn2(grads, state2)

        # Should produce different noise
        assert not torch.allclose(noisy1["weight"], noisy2["weight"], atol=1e-3)

    def test_distributed_preserves_stddev(self):
        """Distributed noise respects stddev parameter."""
        seed = 42
        grads = {"weight": torch.randn(10, 5)}

        # Small stddev
        noise_fn_small, state_small = gaussian_noise(stddev=0.1, key=key(seed))
        noisy_small, state_small = noise_fn_small(grads, state_small)

        # Large stddev
        noise_fn_large, state_large = gaussian_noise(stddev=10.0, key=key(seed))
        noisy_large, state_large = noise_fn_large(grads, state_large)

        # Noise magnitude should differ significantly
        diff_small = (noisy_small["weight"] - grads["weight"]).abs().mean()
        diff_large = (noisy_large["weight"] - grads["weight"]).abs().mean()

        assert diff_large > diff_small * 10  # At least 10x larger

    def test_zero_stddev_no_noise(self):
        """stddev=0 adds no noise (with or without distributed)."""
        seed = 42
        grads = {"weight": torch.randn(10, 5), "bias": torch.randn(5)}

        noise_fn, state = gaussian_noise(stddev=0.0, key=key(seed))
        noisy, state = noise_fn(grads, state)

        # Should be unchanged
        assert torch.allclose(noisy["weight"], grads["weight"])
        assert torch.allclose(noisy["bias"], grads["bias"])

    def test_negative_stddev_raises(self):
        """Negative stddev raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            gaussian_noise(stddev=-1.0, key=key(42))


class TestDistributedNoiseWithPyTree:
    """Tests for distributed noise with different PyTree structures."""

    def test_nested_pytree(self):
        """Distributed noise works with nested PyTrees."""
        stddev = 1.0
        seed = 42

        grads = {
            "layer1": {"weight": torch.randn(10, 5), "bias": torch.randn(5)},
            "layer2": {"weight": torch.randn(5, 3), "bias": torch.randn(3)},
        }

        noise_fn, state = gaussian_noise(stddev, key=key(seed))
        noisy, state = noise_fn(grads, state)

        # Should preserve structure
        assert "layer1" in noisy
        assert "layer2" in noisy
        assert "weight" in noisy["layer1"]
        assert "bias" in noisy["layer1"]

        # Should add noise
        assert not torch.allclose(noisy["layer1"]["weight"], grads["layer1"]["weight"])

    def test_list_of_tensors(self):
        """Distributed noise works with list of tensors."""
        stddev = 1.0
        seed = 42

        grads = [torch.randn(10, 5), torch.randn(5)]

        noise_fn, state = gaussian_noise(stddev, key=key(seed))
        noisy, state = noise_fn(grads, state)

        # Should preserve structure
        assert len(noisy) == 2
        assert noisy[0].shape == grads[0].shape
        assert noisy[1].shape == grads[1].shape

    def test_single_tensor(self):
        """Distributed noise works with single tensor (not dict)."""
        stddev = 1.0
        seed = 42

        grad = torch.randn(10, 5)

        noise_fn, state = gaussian_noise(stddev, key=key(seed))
        noisy, state = noise_fn(grad, state)

        # Should add noise
        assert not torch.allclose(noisy, grad)
        assert noisy.shape == grad.shape

    def test_preserves_dtype(self):
        """Distributed noise preserves tensor dtype."""
        stddev = 1.0
        seed = 42

        grads = {
            "float32": torch.randn(5, dtype=torch.float32),
            "float64": torch.randn(5, dtype=torch.float64),
        }

        noise_fn, state = gaussian_noise(stddev, key=key(seed))
        noisy, state = noise_fn(grads, state)

        assert noisy["float32"].dtype == torch.float32
        assert noisy["float64"].dtype == torch.float64

    def test_preserves_device(self, device):
        """Distributed noise preserves tensor device."""
        stddev = 1.0
        seed = 42

        grads = {
            "weight": torch.randn(10, 5, device=device),
            "bias": torch.randn(5, device=device),
        }

        noise_fn, state = gaussian_noise(stddev, key=key(seed))
        noisy, state = noise_fn(grads, state)

        assert noisy["weight"].device.type == device.type
        assert noisy["bias"].device.type == device.type


class TestNoiseCalibration:
    """Tests for noise calibration in distributed settings."""

    def test_noise_std_matches_stddev_roughly(self):
        """Generated noise has approximately the specified standard deviation."""
        stddev = 2.0
        seed = 42
        n_samples = 10000

        # Generate large sample of noise
        grad = torch.zeros(n_samples)
        noise_fn, state = gaussian_noise(stddev, key=key(seed))
        noisy, state = noise_fn(grad, state)

        # Noise should have mean ≈ 0 and std ≈ stddev
        noise = noisy - grad  # Extract noise
        assert abs(noise.mean().item()) < 0.1  # Mean close to 0
        assert abs(noise.std().item() - stddev) < 0.1  # Std close to stddev


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup_ddp(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker_mf_shared_noise(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grad_template = {"weight": torch.zeros(4, device=device)}
        noise_fn, state = identity_mf_noise(grad_template, stddev=1.0, key=key(0))
        grads = {"weight": torch.zeros(4, device=device)}
        noisy, _ = noise_fn(grads, state)

        gathered = [torch.zeros_like(noisy["weight"]) for _ in range(world_size)]
        dist.all_gather(gathered, noisy["weight"])
        if rank == 0:
            for other in gathered[1:]:
                assert torch.allclose(gathered[0], other)
    finally:
        _cleanup_ddp()


class TestDistributedMFNoise:
    """Tests for MF noise determinism in distributed mode."""

    def test_mf_noise_shared_seed(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Requires >= 2 CUDA devices")
        port = _find_free_port()
        mp.spawn(_worker_mf_shared_noise, args=(2, port), nprocs=2, join=True)
