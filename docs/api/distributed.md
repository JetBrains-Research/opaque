# opaque.distributed

Distributed training utilities for differential privacy.

This module provides PyTorch-native distributed primitives for DP training with DDP (DistributedDataParallel).

## Overview

The `opaque.distributed` module provides composable primitives for distributed DP training:

- **Core utilities**: Check initialization, get rank/world_size
- **Gradient aggregation**: Sum/average PyTrees of gradients across GPUs
- **State synchronization**: Sync adaptive clipping state across devices
- **No custom hooks**: Works with Opaque's functional API that already produces clipped gradients

## Design Philosophy

1. **Composable primitives**, not heavyweight abstractions
2. **PyTorch-native patterns** (DDP)
3. **No backward hooks** (functional API already produces clipped gradients)
4. **Explicit control** over when gradients are aggregated

## Supported Parallelism

**Opaque supports DDP only.** FSDP, Tensor Parallelism, and Pipeline Parallelism are not supported.

| Strategy | Status | Notes |
|----------|--------|-------|
| **DDP** | ✅ Fully Supported | Multi-GPU data parallelism with synchronized noise |
| **FSDP** | ❌ Not Supported | Parameter sharding not compatible, may be explored in future |
| **TP/PP** | ❌ Not Supported | Tensor/Pipeline parallelism incompatible with vmap-based per-example gradients |

For current distributed training capabilities, see
[docs/development/parallelism_compatibility.md](../development/parallelism_compatibility.md).

## Quick Start

### Critical Requirement: Deterministic Synchronized Noise

**Opaque uses deterministic noise generation to keep distributed processes synchronized.** When `seed=None` or a fixed seed is provided to `gaussian_noise()`, all processes generate **identical noise from the same seed**. This is essential for:

1. **Maintaining DP guarantees** - Each device must apply the same noise to ensure privacy bounds hold
2. **Preventing model divergence** - Different noise per device causes training to diverge
3. **Correct privacy accounting** - Accounting assumes synchronized noise across all devices

**Pattern:**
```python
# ✅ CORRECT: All devices use SAME seed
import torch.distributed as dist
from opaque.noise import gaussian_noise

noise_fn, noise_state = gaussian_noise(stddev=1.1, seed=42)  # Same seed on ALL devices

for batch in dataloader:
    grads = clipped_grad_fn(...)
    grads = sum_gradients(grads)  # Aggregate first
    noisy_grads, noise_state = noise_fn(grads, noise_state)  # Then add synchronized noise
    optimizer.update(params, noisy_grads)
```

**Why synchronization matters:**
- Each device independently computes noise with the same seed → identical values
- This is not broadcasting (which would be inefficient); it's deterministic generation
- `seed=None` automatically selects a deterministic shared seed when distributed training is detected
- Different seeds per device will cause training failure (model divergence)

See [docs/user-guide/distributed.md](../user-guide/distributed.md) for detailed examples.

## Quick Start (Legacy Section)

```python
import torch.distributed as dist
import opaque.distributed as dist_utils
from opaque.clipping import clipped_grad
from opaque.noise import gaussian_noise

# Initialize PyTorch distributed
dist.init_process_group(backend='nccl')

# Create DP gradient function
clipped_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Training loop
for batch in dataloader:
    # 1. Compute clipped gradients (local)
    grads = clipped_grad_fn(params, batch)
    
    # 2. Aggregate across GPUs (SUM)
    if dist_utils.is_distributed():
        grads = dist_utils.sum_gradients(grads)
    
    # 3. Add noise (same deterministic seed on every device by default)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # 4. Update parameters
    params = optimizer_update(params, noisy_grads)
```

## Core Utilities

::: opaque.distributed.is_distributed
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.get_rank
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.get_world_size
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.all_reduce
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.barrier
    options:
        show_source: true
        heading_level: 3

## Gradient Aggregation

!!! tip "DP Training"
    Use `sum_gradients()` for DP training - it sums clipped gradients across devices.
    For generic PyTree reduction, use `reduce_pytree()`.

::: opaque.distributed.sum_gradients
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.reduce_pytree
    options:
        show_source: true
        heading_level: 3

## State Synchronization

!!! tip "Scalar Reduction"
    Use `reduce_scalar()` to reduce scalar values across devices.

::: opaque.distributed.reduce_scalar
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.sync_state
    options:
        show_source: true
        heading_level: 3

## Examples

### Basic DDP Training

```python
import torch
import torch.distributed as dist
from opaque.clipping import clipped_grad
from opaque.noise import gaussian_noise
import opaque.distributed as dist_utils

# Initialize distributed
dist.init_process_group(backend='nccl')
rank = dist_utils.get_rank()
device = torch.device(f'cuda:{rank}')

# Setup DP-SGD
clipped_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Training loop
for batch in dataloader:
    batch = batch.to(device)
    
    # Compute clipped gradients
    grads = clipped_grad_fn(params, batch)
    
    # Sum across GPUs
    grads = dist_utils.sum_gradients(grads)
    
    # Add noise (same deterministic seed on every device by default)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # Update
    params = optimizer_update(params, noisy_grads)
```

### Adaptive Clipping with DDP

```python
from opaque.clipping import adaptive_clipped_grad
import opaque.distributed as dist_utils

# Create adaptive clipping function
grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,
    target_quantile=0.75,
)

for batch in dataloader:
    # Compute clipped gradients
    (grads, aux), new_clip_state = grad_fn(
        params, batch, state=clip_state
    )
    
    # Synchronize clip state across GPUs
    if dist_utils.is_distributed():
        new_clip_state = dist_utils.sync_state(
            new_clip_state,
            sync_fields=["clip_norm", "clipping_rate"],
            op="mean",
        )
    
    clip_state = new_clip_state
    
    # Sum first, then add noise (same seed across devices by default)
    grads = dist_utils.sum_gradients(grads)
    noise_fn, noise_state = gaussian_noise(
        stddev=1.1 * clip_state.sensitivity()
    )
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # Update
    params = optimizer_update(params, noisy_grads)
```

### Complete Training Script

See [examples/train_qwen_ddp.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_qwen_ddp.py) for a complete DDP training script with:

- Multi-GPU setup
- LoRA fine-tuning
- Adaptive clipping
- Privacy accounting

## Privacy Guarantees

### Key Principle

With DDP, privacy guarantees are maintained because:

1. **Each GPU processes disjoint batches** (no data overlap)
2. **Clipping happens per-example** before aggregation
3. **Gradients are summed across devices** before noise is applied
4. **Noise is added on every device** with the same deterministic seed by default

### Effective Batch Size

Your **effective batch size** for privacy accounting is:

```python
effective_batch_size = local_batch_size × num_gpus
```

Use this for calculating `sample_rate`:

```python
import opaque.accounting as acc

# Calculate sample rate
effective_batch_size = local_batch_size * world_size
sample_rate = effective_batch_size / dataset_size

# Calibrate noise
noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,
    delta=1e-5,
    sample_rate=sample_rate,
    num_steps=num_steps,
)
```

### Critical Ordering

The order of operations is **critical** for privacy:

✅ **Correct** (privacy-preserving):
```python
grads = clipped_grad_fn(params, batch)       # 1. Clip
grads = sum_gradients(grads)                  # 2. Sum
noisy_grads, state = noise_fn(grads, state)   # 3. Noise (same seed on each device)
```

❌ **Wrong** (violates privacy / causes divergence):
```python
grads = clipped_grad_fn(params, batch)       # 1. Clip
noisy_grads, state = noise_fn(grads, state)  # 2. Noise (too early!)
noisy_grads = sum_gradients(noisy_grads)     # 3. Sum (after noise)
```

## Best Practices

### 1. Use Effective Batch Size

```python
# Configuration
local_batch_size = 16  # Per GPU
num_gpus = 4
effective_batch_size = local_batch_size * num_gpus  # 64 total

# Use for privacy
sample_rate = effective_batch_size / dataset_size
```

### 2. Synchronize Adaptive Clipping State

```python
# After each gradient computation
if dist_utils.is_distributed():
    clip_state = dist_utils.sync_state(
        clip_state,
        sync_fields=["clip_norm"],  # Don't sync step count!
        op="mean",
    )
```

### 3. Barrier Before Validation

```python
# Wait for all GPUs to finish training
if dist_utils.is_distributed():
    dist_utils.barrier()

# Only main process validates
if dist_utils.get_rank() == 0:
    accuracy = validate(model, val_loader)
```

### 4. Save Only on Main Process

```python
# Avoid race conditions
if dist_utils.get_rank() == 0:
    torch.save(model.state_dict(), 'checkpoint.pt')
```

### 5. Keep Noise Deterministic Across Ranks

```python
# Use seed=None to get the same deterministic seed on every device
noise_fn, noise_state = gaussian_noise(stddev=1.1)
```

## Troubleshooting

### Hanging During Training

**Cause**: Mismatched collective operations (some GPUs call `barrier()`, others don't)

**Solution**: Ensure all GPUs execute the same control flow:
```python
# Use drop_last=True to ensure same number of batches
sampler = DistributedSampler(dataset, drop_last=True)
dataloader = DataLoader(dataset, sampler=sampler)
```

### Different Results Across Runs

**Cause**: Noise seed mismatch across devices

**Solution**: Use the default generator (same seed on each device):
```python
noise_fn, noise_state = gaussian_noise(stddev=1.1)
```

### Out of Memory

**Cause**: Batch size too large per GPU

**Solution**: Reduce `local_batch_size` (not `microbatch_size`):
```python
# Reduce per-GPU batch size
local_batch_size = 8  # Was 16

# Or enable gradient checkpointing
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
```

## See Also

- **[User Guide: Distributed Training](../user-guide/distributed.md)** - Complete guide
- **[Tutorial 07](../tutorials/07_distributed_training.ipynb)** - Interactive notebook
- **[examples/train_qwen_ddp.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_qwen_ddp.py)** - Working example

---

**Questions?** Open an issue on [GitHub](https://github.com/JetBrains-Research/opaque/issues)
