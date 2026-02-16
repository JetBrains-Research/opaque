# Distributed Training with DDP

Opaque supports distributed training using PyTorch's **DistributedDataParallel (DDP)**. This guide explains how to train differentially private models across multiple GPUs.

## Overview

Distributed training with DP requires careful handling because:

1. **Per-example gradients** must be computed before averaging across GPUs
2. **Noise is added locally** on each GPU to maintain DP guarantees
3. **Gradient aggregation** happens after clipping and noise injection

Opaque provides utilities for distributed DP-SGD that maintain privacy guarantees while scaling to multiple GPUs.

## Quick Start

Here's a minimal DDP training script:

```python
import torch.distributed as dist
from opaque.clipping import clipped_grad
from opaque.distributed import average_gradients
from opaque.noise import gaussian_noise

# 1. Initialize distributed training
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# 2. Create model (same as single-GPU)
model = MyModel().to(device)
fmodel, params = make_functional(model)

# 3. Create DP gradient function
clipped_grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0)
# Create deterministic noise using functional API; offset seed by rank if desired
gen = 42 + rank if isinstance(42, int) else 42
noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=gen)

# 4. Training loop with gradient averaging
for batch in dataloader:
    # Compute clipped gradients locally
    grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
    
    # Add noise locally
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # Average gradients across GPUs
    noisy_grads = average_gradients(noisy_grads)
    
    # Update parameters
    params = optimizer_update(params, noisy_grads)
```

**Launch with**:
```bash
torchrun --nproc_per_node=4 train.py
```

## Key Concepts

### DDP for DP-SGD

Standard DDP aggregates gradients **before** applying them. DP-SGD requires a different approach:

1. **Compute per-example gradients** on local batch (via `vmap`)
2. **Clip each gradient** independently (maintains sensitivity)
3. **Sum clipped gradients** locally
4. **Add noise** to local sum
5. **Average noisy gradients** across GPUs (via `all_reduce`)

!!! warning "Order Matters"
    You **must** add noise **before** averaging gradients. Adding noise after averaging would break DP guarantees!

### Privacy Guarantees

With DDP, privacy guarantees are maintained because:

- Each GPU processes **disjoint batches** (no data overlap)
- Clipping happens **per-example** before aggregation
- Noise is added **locally** with appropriate scale
- Gradient averaging is a **post-processing** step (doesn't affect privacy)

The overall privacy budget is the same as single-GPU training with the combined batch size.

### Effective Batch Size

With DDP, your **effective batch size** is:

```
effective_batch_size = local_batch_size × num_gpus
```

For example:
- 4 GPUs × 8 samples/GPU = 32 effective batch size
- This is what you use for privacy accounting

## API Reference

### Gradient Aggregation

Opaque provides two functions for gradient aggregation:

#### `average_gradients()`

Average gradients across all GPUs (most common):

```python
from opaque.distributed import average_gradients

# After computing local noisy gradients
noisy_grads = noise_fn(clipped_grads)

# Average across all GPUs
avg_grads = average_gradients(noisy_grads)
```

#### `all_reduce_gradients()`

Sum gradients across all GPUs (no averaging):

```python
from opaque.distributed import all_reduce_gradients

# Sum across all GPUs (no division by world_size)
summed_grads = all_reduce_gradients(noisy_grads)
```

!!! tip "Which to use?"
    Use `average_gradients()` for most cases. It's equivalent to single-GPU training with larger batch size.

### State Synchronization

Synchronize optimizer state or other values across GPUs:

```python
from opaque.distributed import sync_state, sync_scalar

# Synchronize dataclass state (e.g., adaptive clipping)
clip_state = sync_state(clip_state)

# Synchronize single scalar value
clip_norm = sync_scalar(clip_norm)
```

### Distributed Utilities

```python
from opaque.distributed import is_initialized, get_rank, get_world_size

# Check if distributed training is active
if is_initialized():
    rank = get_rank()        # GPU index (0, 1, 2, ...)
    world_size = get_world_size()  # Total number of GPUs
    is_main = (rank == 0)    # Main process for logging
```

## Complete Example

Here's a complete DDP training script with LoRA:

```python
"""Train a model with DP-SGD using DDP across multiple GPUs."""
import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

from opaque.clipping import adaptive_clipped_grad
from opaque.distributed import average_gradients, is_initialized, get_rank
from opaque.noise import gaussian
from opaque.utils import make_functional, merge


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        return True, rank, world_size, device
    
    # Single GPU fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def main():
    # Setup
    distributed, rank, world_size, device = setup_distributed()
    is_main = (rank == 0)
    
    if is_main:
        print(f"Training on {world_size} GPU(s)")
    
    # Model setup
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    
    # Add LoRA
    lora_config = LoraConfig(r=8, target_modules=["c_attn"])
    model = get_peft_model(model, lora_config)
    
    # Convert to functional
    fmodel, trainable, frozen = make_functional(
        model, 
        partition_trainable=True
    )
    
    # DP-SGD setup
    def loss_fn(params, batch):
        all_params = merge(frozen, params)
        logits = fmodel(all_params, batch)
        return compute_loss(logits, batch)
    
    grad_fn, clip_state = adaptive_clipped_grad(
        loss_fn,
        initial_clip_norm=1.0,
        microbatch_size=2,
    )
    
    noise_fn = gaussian(stddev=1.1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Each GPU processes different data
        local_dataloader = get_local_dataloader(rank, world_size)
        
        for batch in local_dataloader:
            # 1. Compute clipped gradients locally
            (grads, aux), clip_state = grad_fn(
                trainable, batch, state=clip_state
            )
            
            # 2. Add noise locally
            noisy_grads = noise_fn(grads)
            
            # 3. Average across GPUs
            if distributed:
                noisy_grads = average_gradients(noisy_grads)
            
            # 4. Update parameters
            trainable = optimizer_update(trainable, noisy_grads)
            
            if is_main and step % 10 == 0:
                print(f"Step {step}: loss={aux.loss_values.mean():.4f}")
    
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

**Launch script**:
```bash
#!/bin/bash
# train_ddp.sh

NUM_GPUS=4
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    train.py \
    --batch_size=8 \
    --epochs=3
```

## Data Loading

### Per-GPU Data Sharding

Each GPU should process **different data** to maximize training efficiency:

```python
from torch.utils.data.distributed import DistributedSampler

# Create distributed sampler
train_sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

# Create dataloader with sampler
train_loader = DataLoader(
    dataset,
    batch_size=local_batch_size,
    sampler=train_sampler,
)

# In training loop
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # Shuffle differently each epoch
    for batch in train_loader:
        ...
```

### Effective Batch Size

Calculate your effective batch size for privacy accounting:

```python
local_batch_size = 8      # Per GPU
num_gpus = 4
effective_batch_size = local_batch_size * num_gpus  # 32

# Use effective batch size for privacy
sample_rate = effective_batch_size / len(dataset)
```

## Privacy Accounting

Privacy accounting with DDP is the **same as single-GPU** using the **effective batch size**:

```python
import opaque.accounting as acc

# Calculate effective batch size
effective_batch_size = local_batch_size * world_size
sample_rate = effective_batch_size / len(dataset)

# Calibrate noise (only on main process)
if is_main:
    noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
        epsilon=3.0,
        delta=1e-5,
        sample_rate=sample_rate,
        num_steps=num_steps,
    )
    print(f"Using noise_multiplier={noise_multiplier:.3f}")

# Broadcast noise_multiplier to all GPUs if needed
if distributed:
    noise_tensor = torch.tensor([noise_multiplier], device=device)
    dist.broadcast(noise_tensor, src=0)
    noise_multiplier = noise_tensor.item()

# Track privacy during training
privacy_state = acc.create()
for step in range(num_steps):
    # ... training ...
    
    # Compose privacy
    privacy_state = acc.compose_poisson_gaussian(
        privacy_state,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
    )
    
    # Check privacy (only on main)
    if is_main and step % 100 == 0:
        eps = acc.get_epsilon(privacy_state, delta=1e-5)
        print(f"Step {step}: ε={eps:.2f}")
```

## Adaptive Clipping with DDP

Adaptive clipping requires state synchronization across GPUs:

```python
from opaque.clipping import adaptive_clipped_grad
from opaque.distributed import average_gradients

# Create adaptive clipping function
grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,
    target_quantile=0.75,
)

for batch in dataloader:
    # Compute gradients with adaptive clipping
    (grads, aux), new_clip_state = grad_fn(
        params, batch, state=clip_state
    )
    
    # Add noise and average
    noisy_grads = noise_fn(grads)
    if distributed:
        noisy_grads = average_gradients(noisy_grads)
    
    # Synchronize clip state across GPUs
    if distributed:
        from opaque.distributed import sync_state
        new_clip_state = sync_state(new_clip_state)
    
    clip_state = new_clip_state
    params = optimizer_update(params, noisy_grads)
```

!!! note "State Synchronization"
    Adaptive clipping maintains state (clip norm, sum of per-example norms). This state must be synchronized across GPUs to ensure all processes use the same clip norm.

## Best Practices

### 1. Use Gradient Checkpointing

Reduce memory per GPU with gradient checkpointing:

```python
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
```

This trades compute for memory (slower but fits larger models).

### 2. Balance Batch Sizes

Choose batch sizes to maximize GPU utilization:

```python
# Good: 4 GPUs × 8 samples = 32 effective batch
local_batch_size = 8
num_gpus = 4

# Better: 4 GPUs × 16 samples = 64 effective batch (if memory allows)
local_batch_size = 16
num_gpus = 4
```

Larger effective batches = better privacy-utility tradeoff.

### 3. Synchronize Before Validation

Ensure all GPUs finish training before validation:

```python
if distributed:
    dist.barrier()  # Wait for all GPUs

if is_main:
    # Only main process does validation
    validate(model, val_loader)
```

### 4. Save Only on Main Process

Avoid race conditions by saving only on rank 0:

```python
if is_main:
    torch.save(model.state_dict(), "checkpoint.pt")
```

### 5. Log Only on Main Process

Reduce logging overhead:

```python
if is_main:
    print(f"Step {step}: loss={loss:.4f}")
    wandb.log({"loss": loss})
```

## Troubleshooting

### Out of Memory

**Problem**: `CUDA out of memory` on DDP training

**Solutions**:
1. Reduce `local_batch_size` (not `microbatch_size`)
2. Enable gradient checkpointing
3. Use smaller model or LoRA rank
4. Increase `microbatch_size` (processes fewer samples at once)

### Hanging During Training

**Problem**: Training hangs indefinitely

**Causes**:
- Mismatched collective operations (some GPUs call `dist.barrier()`, others don't)
- Different number of batches per GPU

**Solutions**:
```python
# Ensure same number of iterations per GPU
if len(dataset) % world_size != 0:
    # Pad dataset or drop last batch
    train_loader = DataLoader(..., drop_last=True)

# Use timeout for debugging
dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
```

### Different Results Across Runs

**Problem**: DP training gives different results each run

**Cause**: Non-deterministic noise generation

**Solution**: Seed RNG per rank:
```python
def setup_seed(rank):
    torch.manual_seed(42 + rank)  # Different seed per GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)
```

### Gradient Explosion

**Problem**: Gradients become NaN or inf

**Solutions**:
1. Check clip norm: `print(clip_state.clip_norm)`
2. Reduce learning rate
3. Check noise multiplier (too high = unstable)
4. Enable gradient checkpointing

## Examples

See working examples in the repository:

- **[examples/train_qwen_ddp.py](../../examples/train_qwen_ddp.py)** - Complete DDP training script with Qwen2
- **[tests/distributed/](../../tests/distributed/)** - Distributed training tests

## Limitations

Current DDP support limitations:

1. **No FSDP support** - Fully Sharded Data Parallel not yet supported (coming soon)
2. **Single-node only** - Multi-node DDP not tested (but should work)
3. **NCCL backend only** - Other backends (Gloo, MPI) not tested

## Next Steps

- **[Memory Profiling](memory-profiling.md)** - Optimize memory usage for multi-GPU
- **[LoRA Fine-tuning](lora.md)** - Parameter-efficient training with DP
- **[Tutorial 07](../tutorials/07_distributed_training.ipynb)** - Interactive DDP tutorial

---

**Questions?** Open an issue on [GitHub](https://github.com/JetBrains-Research/opaque/issues)
