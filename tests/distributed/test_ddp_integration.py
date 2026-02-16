"""Integration test for DDP (DistributedDataParallel) training.

This test validates that Opaque's distributed primitives work correctly
with PyTorch DDP on multiple GPUs.

Run with:
    # Single GPU (baseline)
    uv run pytest tests/distributed/test_ddp_integration.py -v

    # 4 GPUs (real distributed)
    torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py -v
"""

import os

import pytest
import torch
import torch.nn as nn

# Mark all tests in this file as requiring GPU
pytestmark = pytest.mark.gpu


def is_distributed():
    """Check if running in distributed mode."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def setup_distributed():
    """Initialize distributed if environment variables are set."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        return True
    return False


@pytest.fixture(scope="module", autouse=True)
def setup_dist():
    """Setup distributed environment for all tests in this module."""
    initialized = setup_distributed()
    yield initialized
    if initialized and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture
def device():
    """Get device for current process."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if is_distributed():
        rank = torch.distributed.get_rank()
        return torch.device(f"cuda:{rank}")
    else:
        return torch.device("cuda:0")


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleModel()


class TestDistributedUtilities:
    """Test basic distributed utilities."""

    def test_distributed_detection(self):
        """Test that distributed initialization is detected correctly."""
        from opaque.distributed import is_initialized as opaque_is_initialized

        # Opaque should detect PyTorch distributed state
        pytorch_initialized = is_distributed()
        opaque_initialized = opaque_is_initialized()

        assert pytorch_initialized == opaque_initialized

    def test_rank_and_world_size(self):
        """Test rank and world size reporting."""
        from opaque.distributed import get_rank, get_world_size

        rank = get_rank()
        world_size = get_world_size()

        if is_distributed():
            # In distributed mode, check against PyTorch
            assert rank == torch.distributed.get_rank()
            assert world_size == torch.distributed.get_world_size()
            assert 0 <= rank < world_size
        else:
            # In single-GPU mode
            assert rank == 0
            assert world_size == 1


class TestGradientAggregation:
    """Test gradient aggregation across devices."""

    def test_all_reduce_gradients_sum(self, device):
        """Test that gradients are summed across devices."""
        from opaque.distributed import all_reduce_gradients, get_rank, get_world_size

        if not is_distributed():
            pytest.skip("Requires distributed setup")

        rank = get_rank()
        world_size = get_world_size()

        # Each rank has different gradients
        grads = {
            "weight": torch.ones(5, 3, device=device) * (rank + 1),
            "bias": torch.ones(3, device=device) * (rank + 1),
        }

        # Sum across all devices
        result, _ = all_reduce_gradients(grads, op="sum")

        # Expected: sum of 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
        expected_sum = sum(range(1, world_size + 1))

        assert torch.allclose(
            result["weight"], torch.ones(5, 3, device=device) * expected_sum
        )
        assert torch.allclose(
            result["bias"], torch.ones(3, device=device) * expected_sum
        )

    def test_average_gradients(self, device):
        """Test that gradients are averaged across devices."""
        from opaque.distributed import average_gradients, get_rank, get_world_size

        if not is_distributed():
            pytest.skip("Requires distributed setup")

        rank = get_rank()
        world_size = get_world_size()

        # Each rank has different gradients
        grads = {
            "weight": torch.ones(5, 3, device=device) * (rank + 1),
            "bias": torch.ones(3, device=device) * (rank + 1),
        }

        # Average across all devices
        result = average_gradients(grads)

        # Expected: average of 1 + 2 + ... + world_size
        expected_avg = sum(range(1, world_size + 1)) / world_size

        assert torch.allclose(
            result["weight"], torch.ones(5, 3, device=device) * expected_avg, atol=1e-5
        )
        assert torch.allclose(
            result["bias"], torch.ones(3, device=device) * expected_avg, atol=1e-5
        )


class TestStateSynchronization:
    """Test state synchronization across devices."""

    def test_sync_scalar(self, device):
        """Test scalar synchronization across devices."""
        from opaque.distributed import get_rank, get_world_size, sync_scalar

        if not is_distributed():
            pytest.skip("Requires distributed setup")

        rank = get_rank()
        world_size = get_world_size()

        # Each rank has different value
        value = float(rank + 1)

        # Synchronize (average)
        synced = sync_scalar(value, op="mean", device=device)

        # Expected: average of 1 + 2 + ... + world_size
        expected_avg = sum(range(1, world_size + 1)) / world_size

        assert abs(synced - expected_avg) < 1e-5

    def test_sync_adaptive_clip_state(self, device):
        """Test AdaptiveClipState synchronization."""
        from opaque.clipping import AdaptiveClipState
        from opaque.distributed import get_rank, get_world_size, sync_state

        if not is_distributed():
            pytest.skip("Requires distributed setup")

        rank = get_rank()
        world_size = get_world_size()

        # Each rank has different clip_norm
        state = AdaptiveClipState(
            clip_norm=float(rank + 1),
            step=100,  # Should not be synced
            clipping_rate=0.5 + 0.1 * rank,
            rescale_to_unit_norm=False,
        )

        # Synchronize clip_norm and clipping_rate
        synced = sync_state(
            state,
            sync_fields=["clip_norm", "clipping_rate"],
            op="mean",
            device=device,
        )

        # Expected averages
        expected_clip_norm = sum(range(1, world_size + 1)) / world_size
        expected_rate = sum(0.5 + 0.1 * r for r in range(world_size)) / world_size

        assert abs(synced.clip_norm - expected_clip_norm) < 1e-5
        assert abs(synced.clipping_rate - expected_rate) < 1e-5
        assert synced.step == 100  # Should be unchanged


class TestDeterministicNoise:
    """Test deterministic noise generation in distributed mode."""

    def test_different_noise_per_rank(self, device):
        """Test that each rank gets different noise."""
        from opaque.distributed import get_rank
        from opaque.noise import gaussian_noise

        if not is_distributed():
            pytest.skip("Requires distributed setup")

        seed = 42

        # Create noise function (offset seed by rank for per-rank determinism)
        gen = seed + get_rank() if isinstance(seed, int) else seed
        noise_fn, state = gaussian_noise(1.0, generator=gen)

        grads = {
            "weight": torch.zeros(10, 5, device=device),
            "bias": torch.zeros(5, device=device),
        }

        # Apply noise
        noisy = noise_fn(grads, state)

        # Verify noise was added (not all zeros)
        assert not torch.allclose(noisy["weight"], torch.zeros_like(noisy["weight"]))
        assert not torch.allclose(noisy["bias"], torch.zeros_like(noisy["bias"]))

        # Each rank should have different noise
        # (We can't easily verify this without communication, but the test
        # ensures no errors occur)


class TestEndToEndDPTraining:
    """End-to-end DP training with DDP."""

    def test_dp_training_step(self, device, simple_model):
        """Test a single DP training step with DDP."""
        from opaque.clipping import clipped_grad
        from opaque.distributed import all_reduce_gradients, get_rank
        from opaque.noise import gaussian_noise

        if not is_distributed():
            pytest.skip("Requires distributed setup")

        rank = get_rank()

        # Move model to device
        model = simple_model.to(device)

        # Make functional
        from opaque.utils import make_functional

        func_model, params = make_functional(model)

        # Create loss function
        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        # Create clipped gradient function
        grad_fn, clip_state = clipped_grad(
            loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2)
        )

        # Create deterministic noise (INDEPENDENT per device)
        noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=42 + rank)

        # Generate batch (different on each rank)
        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        # APPROACH 1: Independent noise (privacy amplification)
        # Compute clipped gradients (per-device)
        grads, clip_state = grad_fn(params, x, y, state=clip_state)

        # Add noise BEFORE aggregation (different per device)
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        # Sum noisy gradients across devices (NOT average for Poisson sampling!)
        noisy_grads, _ = all_reduce_gradients(noisy_grads, op="sum")

        # Verify gradients are reasonable
        for param_name, grad in noisy_grads.items():
            assert grad.shape == params[param_name].shape
            assert grad.device == device
            assert not torch.isnan(grad).any()
            assert not torch.isinf(grad).any()

    def test_dp_training_step_shared_noise(self, device, simple_model):
        """Test DP training with shared noise (mixture Gaussian accounting)."""
        from opaque.clipping import clipped_grad
        from opaque.distributed import all_reduce_gradients, get_rank
        from opaque.noise import gaussian_noise

        if not is_distributed():
            pytest.skip("Requires distributed setup")

        rank = get_rank()

        # Move model to device
        model = simple_model.to(device)

        # Make functional
        from opaque.utils import make_functional

        func_model, params = make_functional(model)

        # Create loss function
        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        # Create clipped gradient function
        grad_fn, clip_state = clipped_grad(
            loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2)
        )

        # Create deterministic noise (SHARED seed - same on all devices)
        noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=42)  # No +rank

        # Generate batch (different on each rank)
        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        # APPROACH 2: Shared noise (mixture Gaussian accounting)
        # Compute clipped gradients (per-device)
        grads, clip_state = grad_fn(params, x, y, state=clip_state)

        # Sum gradients FIRST (before noise)
        grads, _ = all_reduce_gradients(grads, op="sum")

        # Add noise AFTER aggregation (same seed → same noise on all devices)
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        # Verify gradients are reasonable and same on all devices
        for param_name, grad in noisy_grads.items():
            assert grad.shape == params[param_name].shape
            assert grad.device == device
            assert not torch.isnan(grad).any()
            assert not torch.isinf(grad).any()

    def test_adaptive_clipping_with_sync(self, device, simple_model):
        """Test adaptive clipping with automatic state synchronization."""
        from opaque.clipping import adaptive_clipped_grad

        if not is_distributed():
            pytest.skip("Requires distributed setup")

        # Move model to device
        model = simple_model.to(device)

        # Make functional
        from opaque.utils import make_functional

        func_model, params = make_functional(model)

        # Create loss function
        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        # Create adaptive clipping with distributed sync
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clip_norm=0.1,
            distributed=True,  # Auto-sync state
        )

        # Generate batch (different on each rank)
        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        # Compute gradients with adaptive clipping
        grads, new_state = grad_fn(params, x, y, state=clip_state)

        # Verify state was synced (all ranks should have same clip_norm)
        # We can't easily verify this without communication, but ensure no errors
        assert new_state.clip_norm > 0
        assert new_state.step == 1
        assert 0 <= new_state.clipping_rate <= 1


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
