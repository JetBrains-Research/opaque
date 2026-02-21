"""Example: Distributed DP training with DDP (DistributedDataParallel).

This example demonstrates how to use Opaque's distributed primitives for
differential privacy training across multiple GPUs using SHARDED Poisson sampling
with standard DP-SGD accounting.

Run with torchrun:
    # 4 GPUs
    torchrun --nproc_per_node=4 examples/distributed_dp_training.py

    # 2 GPUs
    torchrun --nproc_per_node=2 examples/distributed_dp_training.py

Features demonstrated:
- Per-device gradient clipping (vmap on each GPU)
- Cross-device gradient aggregation (all-reduce)
- Deterministic noise generation (same seed on all devices)
- Adaptive clipping with state synchronization
- Sharded Poisson sampling (each device samples disjoint data partition)
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import opaque
import opaque.distributed as dist_utils
from opaque.clipping import sync_adaptive_clip_state
from opaque.random import fold_in
from opaque.random import key as rng_key
from opaque.sampling.distributed import local_shard


def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    return rank, world_size, device


def create_model(device):
    """Create a simple model for demonstration."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )
    return model.to(device)


def create_dataset(n_samples=1000):
    """Create synthetic dataset."""
    X = torch.randn(n_samples, 10)
    y = torch.randn(n_samples, 1)
    return TensorDataset(X, y)


def main():
    # Setup
    rank, world_size, device = setup_distributed()

    if rank == 0:
        print("🚀 Starting distributed DP training")
        print(f"   World size: {world_size}")
        print(f"   Device: {device}")

    # Create model and make it functional
    model = create_model(device)
    func_model, params = opaque.make_functional(model)

    # Create dataset and dataloader
    dataset = create_dataset(n_samples=1000)

    # Shard dataset for distributed training
    shard = local_shard(dataset, rank=rank, world_size=world_size)

    # Poisson sampler on local shard with per-rank key
    sampler = opaque.PoissonSampler(
        shard,
        sample_rate=0.01,
        num_epochs=1,
        key=fold_in(rng_key(42), rank),
    )

    # DataLoader with batch_sampler (use shard, not full dataset)
    dataloader = DataLoader(shard, batch_sampler=sampler)

    # Define loss function
    def loss_fn(params, x, y):
        pred = func_model(params, x.unsqueeze(0))
        return ((pred - y) ** 2).mean()

    # Create adaptive clipping with distributed sync
    grad_fn, clip_state = opaque.adaptive_clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        initial_clip_norm=1.0,
        key=rng_key(7),
        target_quantile=0.5,
    )

    # Create deterministic noise function (functional API)
    # Same key on all ranks produces identical noise (prevents model divergence)
    noise_fn, noise_state = opaque.gaussian_noise(stddev=1.1, key=rng_key(42))

    # Privacy accounting (same on all ranks)
    epsilon_target = 3.0
    delta = 1e-5

    if rank == 0:
        print(f"\n📊 Privacy budget: ε={epsilon_target}, δ={delta}")
        print("   Approach: Standard DP-SGD (shared noise across devices)")
        print(f"   Clip norm (initial): {clip_state.clip_norm}")
        print(f"   Noise multiplier: {1.1}")

    # Training loop
    for step, (batch_x, batch_y) in enumerate(dataloader):
        # Batches are already indexed/collated by the DataLoader.
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Skip if batch is empty
        if batch_x.numel() == 0:
            continue

        # Standard DP-SGD with sharded Poisson sampling:
        # Step 1: Compute clipped gradients (per-device on disjoint data shard)
        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Step 1b: Synchronize adaptive clip state across devices
        clip_state = sync_adaptive_clip_state(clip_state)

        # Step 2: Aggregate gradients across devices
        grads = dist_utils.sum_gradients(grads)

        # Step 3: Add noise on ALL devices (all with same seed → identical noise)
        # ⚠️ CRITICAL: noise_fn() is called on EVERY device (rank 0, 1, 2, ...)
        #    NOT just rank 0! Each device independently generates the same noise.
        #    This prevents model divergence across devices.
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        # Step 4: Update parameters (all devices have identically noisy gradients)
        lr = 0.01
        for name in params:
            params[name] = params[name] - lr * noisy_grads[name]

        # Log progress
        if rank == 0 and step % 10 == 0:
            print(
                f"Step {step:3d}: "
                f"batch_size={len(batch_x):3d}, "
                f"clip_norm={clip_state.clip_norm:.4f}, "
                f"clipping_rate={clip_state.clipping_rate:.2%}"
            )

    if rank == 0:
        print("\n✅ Training complete!")
        print(f"   Final clip norm: {clip_state.clip_norm:.4f}")
        print(f"   Total steps: {clip_state.step}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
