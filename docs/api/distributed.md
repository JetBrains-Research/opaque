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
2. **PyTorch-native patterns** (DDP/FSDP/DTensor)
3. **No backward hooks** (functional API already produces clipped gradients)
4. **Explicit control** over when gradients are aggregated

## Quick Start

```python
import torch.distributed as dist
import opaque.distributed as dist_utils
from opaque.clipping import clipped_grad
from opaque.noise import gaussian

# Initialize PyTorch distributed
dist.init_process_group(backend='nccl')

# Create DP gradient function
clipped_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = gaussian(stddev=1.1)

# Training loop
for batch in dataloader:
    # 1. Compute clipped gradients (local)
    grads = clipped_grad_fn(params, batch)
    
    # 2. Add noise (local)
    noisy_grads = noise_fn(grads)
    
    # 3. Sum across GPUs
    if dist_utils.is_initialized():
        noisy_grads = dist_utils.sum_gradients(noisy_grads)
    
    # 4. Update parameters
    params = optimizer_update(params, noisy_grads)
```

## Core Utilities

::: opaque.distributed.is_initialized
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

!!! info "Recommended API"
    Use `sum_gradients()` for DP training (sums clipped gradients across devices).
    For generic PyTree reduction, use `reduce_pytree()`.

::: opaque.distributed.sum_gradients
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.reduce_pytree
    options:
        show_source: true
        heading_level: 3

### Deprecated Functions

!!! warning "Deprecated"
    The following functions are deprecated and will be removed in v3.0.0.
    Use `sum_gradients()` or `reduce_pytree()` instead.

::: opaque.distributed.all_reduce_gradients
    options:
        show_source: true
        heading_level: 4

::: opaque.distributed.average_gradients
    options:
        show_source: true
        heading_level: 4

## State Synchronization

::: opaque.distributed.sync_scalar
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
from opaque.noise import gaussian
import opaque.distributed as dist_utils

# Initialize distributed
dist.init_process_group(backend='nccl')
rank = dist_utils.get_rank()
device = torch.device(f'cuda:{rank}')

# Setup DP-SGD
clipped_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = gaussian(stddev=1.1)

# Training loop
for batch in dataloader:
    batch = batch.to(device)
    
    # Compute clipped gradients
    grads = clipped_grad_fn(params, batch)
    
    # Add noise
    noisy_grads = noise_fn(grads)
    
    # Sum across GPUs
    noisy_grads = dist_utils.sum_gradients(noisy_grads)
    
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
    if dist_utils.is_initialized():
        new_clip_state = dist_utils.sync_state(
            new_clip_state,
            sync_fields=["clip_norm", "clipping_rate"],
            op="mean",
        )
    
    clip_state = new_clip_state
    
    # Add noise and sum
    noise_fn = gaussian(stddev=1.1 * clip_state.sensitivity())
    noisy_grads = noise_fn(grads)
    noisy_grads = dist_utils.sum_gradients(noisy_grads)
    
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
3. **Noise is added locally** with appropriate scale
4. **Gradient averaging is post-processing** (doesn't affect privacy)

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
noisy_grads = noise_fn(grads)                 # 2. Noise
noisy_grads = sum_gradients(noisy_grads)      # 3. Sum
```

❌ **Wrong** (violates privacy):
```python
grads = clipped_grad_fn(params, batch)       # 1. Clip
grads = sum_gradients(grads)                  # 2. Sum (too early!)
noisy_grads = noise_fn(grads)                 # 3. Noise (too late!)
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
if dist_utils.is_initialized():
    clip_state = dist_utils.sync_state(
        clip_state,
        sync_fields=["clip_norm"],  # Don't sync step count!
        op="mean",
    )
```

### 3. Barrier Before Validation

```python
# Wait for all GPUs to finish training
if dist_utils.is_initialized():
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

### 5. Set Different Seeds Per Rank

```python
# Ensure different noise on each GPU
rank = dist_utils.get_rank()
torch.manual_seed(42 + rank)
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

**Cause**: Non-deterministic noise generation

**Solution**: Seed RNG per rank:
```python
def setup_seed(rank, base_seed=42):
    torch.manual_seed(base_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed + rank)
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
