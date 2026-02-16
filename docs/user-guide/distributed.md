# Distributed Training with DDP

Opaque supports distributed training using PyTorch's **DistributedDataParallel (DDP)**. This guide explains how to train differentially private models across multiple GPUs.

## Overview

Distributed training with DP requires careful handling because:

1. **Per-example gradients** must be computed before averaging across GPUs
2. **Noise is added independently on EVERY GPU** to maintain DP guarantees
3. **Gradient aggregation** happens after clipping and noise injection

Opaque provides utilities for distributed DP-SGD that maintain privacy guarantees while scaling to multiple GPUs.

!!! danger "Critical: Noise Must Be Applied on EVERY Device with SAME Seed"
    
    For DP guarantees to hold, **every device must independently apply noise with the SAME seed**:
    
    ```python
    # ✅ CORRECT: All devices use SAME seed (prevents model divergence)
    noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=42)  # SAME seed
    
    for batch in dataloader:
        grads = clipped_grad_fn(params, batch)
        noisy_grads = noise_fn(grads, noise_state)  # THIS HAPPENS ON EVERY DEVICE
        noisy_grads = all_reduce_gradients(noisy_grads)
        params = optimizer_update(params, noisy_grads)
    ```
    
    ```python
    # ❌ WRONG #1: Different noise per device (causes model divergence)
    noise_fn = gaussian_noise(stddev=1.1, generator=42 + rank)
    # Models diverge because each device computes differently with different noise!
    ```
    
    ```python
    # ❌ WRONG #2: Only rank 0 applies noise (breaks DP for other devices)
    if rank == 0:
        noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=42)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        dist.broadcast(noisy_grads)  # ← DON'T DO THIS!
    # Other ranks receive noisy_grads but didn't apply DP → No privacy!
    ```
    
    **Why this matters:**
    - DP guarantees are **per-device per-batch**
    - Every device must call `noise_fn()` in its own training loop
    - Same seed ensures all devices generate identical noise (synchronized, not broadcast)
    - Different seeds cause model divergence (training failure)
    - Never compute noise on rank 0 and broadcast it

## Quick Start

### Recommended: Sharded Poisson Sampling

**Standard approach with simple accounting and automatic seed management:**

```python
import torch.distributed as dist
from opaque.clipping import clipped_grad
from opaque.distributed import all_reduce_gradients, get_rank
from opaque.noise import gaussian_noise
from opaque.sampling import PoissonSampler

# Initialize distributed
dist.init_process_group(backend="nccl")
rank = get_rank()
device = torch.device(f"cuda:{rank}")

# Model and clipping
model = MyModel().to(device)
fmodel, params = make_functional(model)
clipped_grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0)

# Noise: automatically synchronized across devices when distributed
# No seed management needed - all devices use same seed!
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Sampling: automatically detects distributed and uses SHARDED mode
# Seed is automatically shifted by rank for sampling diversity
sampler = PoissonSampler(dataset, sample_rate=0.01, generator=42)

# Training loop
for batch in dataloader:
    # 1. Compute clipped gradients on each device's shard
    grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
    
    # 2. Aggregate across devices
    grads = all_reduce_gradients(grads, op="sum")
    
    # 3. Add synchronized noise on ALL devices (no manual seed management)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # 4. Update parameters
    params = optimizer_update(params, noisy_grads)
```

**Advantages:**
- ✅ **Simple accounting**: Standard DP-SGD, textbook algorithms
- ✅ **No duplicates**: Sharded data ensures examples don't repeat across devices
- ✅ **Automatic**: Both sampler and noise auto-detect distributed mode
- ✅ **No seed management**: Just pass `generator=None` (or omit it) and it works across all devices

### Advanced: Independent Poisson Sampling

**No synchronization needed, but requires mixture Gaussian accounting:**

```python
from opaque.sampling import PoissonSampler, SamplingMode

# Noise: automatically synchronized across devices when distributed
# No seed management needed!
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# INDEPENDENT sampling: each device samples full dataset
# Seed shifted by rank for independent RNG streams (if provided)
sampler = PoissonSampler(dataset, sample_rate=0.01, mode=SamplingMode.INDEPENDENT, generator=42)

# Training loop (same as sharded)
for batch in dataloader:
    grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
    grads = all_reduce_gradients(grads, op="sum")
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = optimizer_update(params, noisy_grads)
```

**Advantages:**
- ✅ **No synchronization**: Each device samples independently (no sharding coordination)
- ✅ **Limited privacy cost**: Mixture Gaussian accounting adds minimal privacy overhead in practice
- ✅ **Simple setup**: No need to partition dataset across devices

**Trade-off:**
- Examples may appear multiple times across devices (handled by mixture Gaussian accounting)
```

**Launch with**:
```bash
torchrun --nproc_per_node=4 train.py
```

## Key Concepts

### Automatic Seed Management in Distributed Training

Opaque automatically handles seed management to prevent common distributed training issues:

**When no seed is provided** (`generator=None`):
- **Sampler**: Uses unseeded RNG (different samples per run)
- **Noise**: Auto-detects distributed mode and uses **same seed (0)** across all devices for synchronized noise

**When seed is provided** (e.g., `generator=42`):
- **Sampler**: Automatically shifts seed by rank (42 → 42, 43, 44, ...) for independent sampling
- **Noise**: Uses seed as-is (user responsible for consistency)

**Result**: The API is simple - just pass `generator=None` (or omit it) and everything works:
- ✅ Sampler gives different data per device (rank shifting)
- ✅ Noise is synchronized across devices (same seed)
- ✅ No model divergence, no data duplication

### Two Poisson Sampling Strategies for Distributed DP

The choice is **NOT between different vs same noise** (both use same noise to avoid model divergence), but between **sampling strategies**:

#### Strategy 1: Sharded Poisson Sampling (Recommended)

**Data partitioning**: Each device samples from disjoint data partition

```python
# Noise: automatically synchronized when distributed (no seed needed)
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Sampler: seed automatically shifted by rank (42 → 42, 43, 44, ...)
sampler = PoissonSampler(dataset, sample_rate=0.01, mode=SamplingMode.SHARDED, generator=42)

# Device 0: samples with seed 42, gets examples 0-49999 (5% of shard)
# Device 1: samples with seed 43, gets examples 50000-99999 (different 5%)
# → No duplication, independent sampling, synchronized noise

for batch in dataloader:
    grads = clipped_grad_fn(params, batch)
    grads = all_reduce_gradients(grads)  # Aggregate across devices
    noisy_grads = noise_fn(grads, noise_state)  # Synchronized noise
```

**Characteristics:**
- Each device processes **disjoint data** (no duplicates across devices)
- Guaranteed to see different data across training
- Requires synchronization to partition dataset
- **Simple accounting**: Standard DP-SGD (no mixture Gaussian needed)

#### Strategy 2: Independent Poisson Sampling (Advanced)

**Independent sampling**: Each device samples full dataset independently

```python
# Noise: automatically synchronized when distributed (no seed needed)
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Sampler: seed automatically shifted by rank (42 → 42, 43, 44, ...)
sampler = PoissonSampler(dataset, sample_rate=0.01, mode=SamplingMode.INDEPENDENT, generator=42)

# Device 0: samples with seed 42: examples 0, 15, 42, ... (5% of 1M)
# Device 1: samples with seed 43: examples 3, 12, 100, ... (5% of 1M, different)
# → Some examples may appear in both device 0 and device 1 (handled by mixture Gaussian)

for batch in dataloader:
    grads = clipped_grad_fn(params, batch)
    grads = all_reduce_gradients(grads)  # Aggregate across devices
    noisy_grads = noise_fn(grads, noise_state)  # Synchronized noise
    # (no sharding synchronization needed!)
```

**Characteristics:**
- Each device samples **independently from full dataset**
- Examples may appear in **multiple devices** (mixture Gaussian property)
- **No synchronization needed** for data partitioning
- **Mixture Gaussian accounting**: Handles duplicate examples across devices
- **Limited privacy cost** in practice despite duplicates

!!! warning "Noise Synchronization: Automatic with Sensible Defaults"
    
    By default, **noise is automatically synchronized across devices**:
    
    ```python
    # ✅ DEFAULT (no seed needed): All devices use same seed (0)
    noise_fn, noise_state = gaussian_noise(stddev=1.1)
    ```
    
    If you provide an explicit seed, it's used as-is:
    
    ```python
    # ✅ EXPLICIT SEED: Same seed on all devices (user provides it)
    noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=42)
    
    # ❌ DIFFERENT PER DEVICE (causes model divergence - avoid!)
    # Don't manually shift by rank for noise:
    noise_fn = gaussian_noise(stddev=1.1, generator=42 + rank)  # Wrong!
    ```

!!! tip "Which Sampling Strategy?"
    
    **Choose SHARDED if:**
    - Simple accounting is preferred (standard DP-SGD)
    - You can coordinate dataset partitioning
    - You need predictable data distribution
    
    **Choose INDEPENDENT if:**
    - Avoiding synchronization complexity is important
    - Mixture Gaussian accounting is acceptable
    - Very limited privacy overhead in practice

### Privacy Guarantees

Both sampling strategies maintain DP guarantees only if:

1. **Same noise is applied on every device** (not different noise per device!)
   - Different noise per device causes model divergence
   - All devices must independently call `noise_fn()` with same seed

2. **Noise is applied on every device** in the training loop
   - Never compute noise on rank 0 and broadcast
   - Each device executes `noisy_grads = noise_fn(grads, noise_state)`

3. **Strategy-specific requirements:**
   - **Sharded**: Each device processes disjoint data (no duplicates)
   - **Independent**: Examples may appear in multiple devices (handled by mixture Gaussian)

**Critical Implementation Detail**: The noise function `noise_fn()` must be called on **every** device with the **same seed**. It is **never** computed on rank 0 and broadcast to others.

The effective batch size for accounting: `local_batch_size × num_gpus`

### Effective Batch Size

With DDP, your **effective batch size** is:

```
effective_batch_size = local_batch_size × num_gpus
```

For example:
- 4 GPUs × 8 samples/GPU = 32 effective batch size
- This is what you use for privacy accounting

### Gradient Flow with Poisson Sampling

**What `clipped_grad` returns:**
```python
grads = clipped_grad_fn(params, batch)  # batch has B examples

# Internally:
# 1. Compute B per-example gradients (via vmap)
# 2. Clip each to L2 norm ≤ C
# 3. SUM the B clipped gradients
# Result: grads = SUM of B clipped gradients (not average!)
```

#### Poisson Sampling: Variable Batch Sizes

**What `clipped_grad` returns:**
```python
grads = clipped_grad_fn(params, batch)  # batch has B examples

# Internally:
# 1. Compute B per-example gradients (via vmap)
# 2. Clip each to L2 norm ≤ C
# 3. SUM the B clipped gradients
# Result: grads = SUM of B clipped gradients (scaled by sampling ratio!)
```

**Batch sizes are variable** due to independent Poisson sampling on each device:

```python
# PoissonSampler auto-detects distributed environment
# In distributed: automatically uses SHARDED mode
# In single device: uses INDEPENDENT mode
sampler = PoissonSampler(dataset, sample_rate=0.01)

# Device 0: batch_size=6  → grads_0 = sum of 6 clipped grads
# Device 1: batch_size=10 → grads_1 = sum of 10 clipped grads
# Device 2: batch_size=7  → grads_2 = sum of 7 clipped grads
# Device 3: batch_size=9  → grads_3 = sum of 9 clipped grads
# Total across devices: 32 examples, but not synchronized
```

**For SHARDED mode** (data partitioning):
- Each device sees different examples
- Batch sizes vary per step on each device
- Grid search recommended for learning rates

```python
# Sharded Poisson sampling (default in distributed)
sampler = PoissonSampler(dataset, sample_rate=0.01)  # Auto SHARDED

for batch in dataloader:
    grads = clipped_grad_fn(params, batch)
    grads = all_reduce_gradients(grads, op="sum")  # Sum across devices
    
    # All devices use same noise (synchronized)
    noisy_grads = noise_fn(grads, noise_state)
    
    # Update with fixed learning rate
    params = params - lr * noisy_grads
```

**For INDEPENDENT mode** (full dataset per device):
- Each device sees full dataset but samples independently
- Requires mixture Gaussian accounting for handling duplicates

```python
from opaque.sampling import SamplingMode

# Independent sampling (each device sees full dataset)
sampler = PoissonSampler(dataset, sample_rate=0.01, mode=SamplingMode.INDEPENDENT)

for batch in dataloader:
    grads = clipped_grad_fn(params, batch)
    grads = all_reduce_gradients(grads, op="sum")
    
    # All devices use same noise (synchronized)
    noisy_grads = noise_fn(grads, noise_state)
    
    # Update
    params = params - lr * noisy_grads
```

**⚠️ Important: Use `sum`, NOT `average_gradients()`**

`average_gradients()` divides by `world_size`, but Poisson sampling creates **different total batch sizes per step**:

```python
# WRONG: Dividing by world_size instead of total batch size
grads = average_gradients(grads)  # Divides by 4
# Effective update: (sum of 6+10+7+9=32 clipped grads) / 4

# ✅ CORRECT: Sum without dividing
grads = all_reduce_gradients(grads, op="sum")
# Effective update: sum of all clipped grads (not normalized)
```
    
    **Use independent sampling** (`distributed=False`) with variable batch sizes for proper DP guarantees.

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

!!! tip "Sum (Not Average) for Poisson Sampling"
    **For proper Poisson sampling privacy amplification**, use `all_reduce_gradients(op="sum")`:
    
    - **`all_reduce_gradients(op="sum")`** (recommended for Poisson sampling):
        - No division by world_size
        - `params -= lr * (sum of all clipped grads)`
        - ✅ Preserves independent Poisson sampling property
        - ✅ Correct for variable batch sizes
        - Effective LR varies per step (acceptable for DP-SGD)
    
    - **`average_gradients()`** is WRONG for independent Poisson sampling:
        - Divides by world_size (not total examples)
        - ❌ Breaks gradient scaling when batch sizes differ
        - Only use if all devices have exact same batch size (not recommended for DP)
    
    **Privacy accounting:** Use total examples across all devices per step.
    
    **What `clipped_grad` returns:** The **SUM** of clipped per-example gradients from the local batch (B examples → sum of B clipped grads).

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
    
    # Independent noise: offset seed by rank for privacy amplification
    from opaque.noise import gaussian_noise
    noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=42 + rank)
    
    # Training loop
    for epoch in range(num_epochs):
        # Each GPU processes different data
        local_dataloader = get_local_dataloader(rank, world_size)
        
        for batch in local_dataloader:
            # 1. Compute clipped gradients locally
            grads, clip_state = grad_fn(
                trainable, batch, state=clip_state
            )
            
            # 2. Add noise locally (different per device)
            noisy_grads, noise_state = noise_fn(grads, noise_state)
            
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
