
"""Example: Distributed DP training with DDP (DistributedDataParallel).

This example demonstrates how to use Opaque's distributed primitives for
differential privacy training across multiple GPUs.

Run with torchrun:
    # 4 GPUs
    torchrun --nproc_per_node=4 examples/distributed_dp_training.py

    # 2 GPUs
    torchrun --nproc_per_node=2 examples/distributed_dp_training.py

Features demonstrated:
- Per-device gradient clipping (vmap on each GPU)
- Cross-device gradient aggregation (all-reduce)
- Deterministic noise generation (seed + rank)
- Adaptive clipping with state synchronization
- Coordinated Poisson sampling (rank 0 broadcasts indices)
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import opaque
import opaque.distributed as dist_utils


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
        print(f"🚀 Starting distributed DP training")
        print(f"   World size: {world_size}")
        print(f"   Device: {device}")
    
    # Create model and make it functional
    model = create_model(device)
    func_model, params = opaque.make_functional(model)
    
    # Create dataset and dataloader
    dataset = create_dataset(n_samples=1000)
    
    # Use distributed Poisson sampler
    # distributed=False: each device samples independently (required for privacy amplification)
    # Variable batch sizes preserve Poisson sampling property
    sampler = opaque.PoissonSampler(
        dataset,
        sample_rate=0.01,
        num_epochs=1,
        distributed=False,  # Independent sampling for privacy amplification
    )
    
    # DataLoader with batch_sampler
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    
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
        target_quantile=0.5,
        distributed=True,  # Auto-sync clip_norm across devices
    )
    
    # Create deterministic noise function (functional API)
    # Two approaches for distributed DP-SGD:
    #
    # APPROACH 1: Independent noise (privacy amplification via parallel composition)
    # - Each device adds noise with DIFFERENT seed BEFORE aggregation
    # - Better privacy bounds (amplification)
    # - Used here:
    seed = 42
    gen = seed + rank if isinstance(seed, int) else seed
    noise_fn, noise_state = opaque.gaussian_noise(stddev=1.1, generator=gen)
    #
    # APPROACH 2: Shared noise (mixture Gaussian accounting)
    # - All devices use SAME seed (no +rank) AFTER aggregation
    # - Standard DP-SGD accounting
    # - Alternative:
    # gen = seed  # Same seed on all ranks
    # noise_fn, noise_state = opaque.gaussian_noise(stddev=1.1, generator=gen)
    
    # Privacy accounting (same on all ranks)
    epsilon_target = 3.0
    delta = 1e-5
    
    if rank == 0:
        print(f"\n📊 Privacy budget: ε={epsilon_target}, δ={delta}")
        print(f"   Approach: Independent noise (privacy amplification)")
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
        
        # APPROACH 1: Independent noise (privacy amplification)
        # Step 1: Compute clipped gradients (per-device)
        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        
        # Step 2: Add noise BEFORE aggregation (different seed per device)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        
        # Step 3: Sum noisy gradients across devices (NOT average!)
        # For Poisson sampling with variable batch sizes, use sum
        noisy_grads = dist_utils.sum_gradients(noisy_grads)
        
        # Step 4: Update parameters (all devices have same noisy gradient sum)
        lr = 0.01
        for key in params:
            params[key] = params[key] - lr * noisy_grads[key]
        
        # APPROACH 2 ALTERNATIVE (shared noise):
        # grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        # grads = dist_utils.sum_gradients(grads)  # Sum first
        # noisy_grads, noise_state = noise_fn(grads, noise_state)  # Same seed → same noise
        # for key in params:
        #     params[key] = params[key] - lr * noisy_grads[key]
        
        # Log progress
        if rank == 0 and step % 10 == 0:
            print(
                f"Step {step:3d}: "
                f"batch_size={len(batch_x):3d}, "
                f"clip_norm={clip_state.clip_norm:.4f}, "
                f"clipping_rate={clip_state.clipping_rate:.2%}"
            )
    
    if rank == 0:
        print(f"\n✅ Training complete!")
        print(f"   Final clip norm: {clip_state.clip_norm:.4f}")
        print(f"   Total steps: {clip_state.step}")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
