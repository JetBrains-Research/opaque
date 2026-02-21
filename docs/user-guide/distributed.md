# Distributed Training with DDP

Opaque supports distributed training using PyTorch's **DistributedDataParallel (DDP)**. This guide explains how to train differentially private models across multiple GPUs.

!!! info "Local-first API (new pattern)"

    Opaque components are moving to a **local-only functional core**:

    - Local step: run component functions on local data and get `(output, state)`
    - Distributed step: explicitly aggregate/sync via component-specific helpers

    Generic collectives remain in `opaque.distributed`.
    Component-specific synchronization lives in submodules like:

    - `opaque.clipping.distributed`
    - `opaque.noise.distributed`
    - `opaque.sampling.distributed`

    For clipping specifically, use:

    - `opaque.distributed.sum_gradients(...)` for gradient aggregation
    - `opaque.clipping.distributed.sync_clip_state(...)`
    - `opaque.clipping.distributed.sync_adaptive_clip_state(...)`
    - `opaque.clipping.distributed.sync_adaptive_clipped_grad_aux(...)`

## Overview

Distributed training with DP requires careful handling because:

1. **Per-example gradients** must be computed before averaging across GPUs
2. **Gradients are aggregated** across devices (AllReduce SUM)
3. **Same noise is added on EVERY device** after aggregation to maintain DP guarantees

Opaque provides utilities for distributed DP-SGD that maintain privacy guarantees while scaling to multiple GPUs.

## Supported Parallelism

Opaque supports **DDP only** today. FSDP/TP/PP are not supported yet. See
[docs/development/parallelism_compatibility.md](../development/parallelism_compatibility.md)
for compatibility research and current limitations.

!!! danger "Critical: Noise Must Be Applied on EVERY Device with SAME Seed"
    
    For DP guarantees to hold, **every device must independently apply noise with the SAME seed**:
    
    ```python
    # ✅ CORRECT: All devices use SAME seed (prevents model divergence)
    from opaque.random import key
    
    noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))  # SAME seed
    
    for batch in dataloader:
        grads = clipped_grad_fn(params, batch)
        grads = sum_gradients(grads)  # Aggregate FIRST
        noisy_grads, noise_state = noise_fn(grads, noise_state)  # Then add noise
        params = optimizer_update(params, noisy_grads)
    ```
    
    ```python
    # ❌ WRONG #1: Different noise per device (causes model divergence)
    from opaque.random import fold_in, key
    
    noise_fn = gaussian_noise(stddev=1.1, key=fold_in(key(42), rank))  # Wrong pattern!
    # Models diverge because each device computes differently with different noise!
    ```
    
    ```python
    # ❌ WRONG #2: Only rank 0 applies noise (breaks DP for other devices)
    if rank == 0:
        noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        dist.broadcast(noisy_grads)  # ← DON'T DO THIS!
    # Other ranks receive noisy_grads but didn't apply DP → No privacy!
    ```
    
    **Why this matters:**
    - DP guarantees are **per-device per-batch**
    - Every device must call `noise_fn()` in its own training loop
    - Same seed ensures all devices generate identical noise (synchronized, not broadcast)
    - When `key=None`, Opaque auto-selects a deterministic shared seed
    - Different seeds cause model divergence (training failure)
    - Never compute noise on rank 0 and broadcast it

## Quick Start

### Example 1: Gaussian Noise with DDP

**Complete working example with standard DP-SGD:**

```python
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.func import functional_call
from opaque.clipping import clipped_grad
from opaque.distributed import sum_gradients
from opaque.noise import gaussian_noise
from opaque.sampling import PoissonSampler

# Initialize distributed (run with: torchrun --nproc_per_node=4 train.py)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")

# Model setup
model = MyModel().to(device)
model = DDP(model, device_ids=[rank])

# Create functional params for clipping (from base model, not DDP wrapper)
params = {k: v.detach() for k, v in model.module.named_parameters()}

# Clipping function
def loss_fn(params, batch):
    x, y = batch
    logits = functional_call(model.module, params, (x,))
    return F.cross_entropy(logits, y)

clipped_grad_fn, clip_state = clipped_grad(
    loss_fn, 
    l2_clip_norm=1.0,
    batch_size=32,
)

# Noise: No seed needed - automatically uses same seed on all devices!
from opaque.random import key
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Sampler: Automatically shards data across devices (seed shifts by rank)
sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_fn,
)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Privacy accounting (calibrate noise for target epsilon)
import opaque.accounting as acc
from opaque.accounting import calibration as cal
from opaque.accounting.accountant import Accountant

dataset_size = len(dataset)
sample_rate = 0.01
num_steps = num_epochs * (dataset_size * sample_rate // 32)  # Approx steps

# Find noise multiplier for (ε=3.0, δ=1e-5)
if rank == 0:
    budget = cal.epsilon_budget(3.0, delta=1e-5)
    result = cal.calibrate(
        budget,
        lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
        param_min=0.5,
        param_max=5.0,
    )
    noise_multiplier = result.param
    print(f"Noise multiplier: {noise_multiplier:.3f}")
    
    # Track privacy during training
    from opaque.accounting.accountant import Accountant

    step_process = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)
    acct = Accountant()

# Broadcast noise multiplier to all devices
if world_size > 1:
    noise_tensor = torch.tensor([noise_multiplier], device=device)
    dist.broadcast(noise_tensor, src=0)
    noise_multiplier = noise_tensor.item()

# Update noise function with calibrated multiplier
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier)

# Training loop
for epoch in range(num_epochs):
    for step_idx, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        
        # 1. Compute per-example clipped gradients (local data)
        grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
        
        # 2. Aggregate gradients across devices (SUM)
        grads = sum_gradients(grads)
        
        # 3. Add noise (same noise on all devices - no manual seed management!)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        
        # 4. Assign gradients and update
        for (name, p), g in zip(model.named_parameters(), noisy_grads.values()):
            p.grad = g.to(p.dtype)
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Update params dict for next iteration
        params = {k: v.detach() for k, v in model.named_parameters()}
        
        # Track privacy (only on main process)
        if rank == 0:
            acct = acct | step_process
            if step_idx % 100 == 0:
                eps = acct.epsilon_at(1e-5)
                print(f"Step {step_idx}: ε={eps:.2f} (budget: 3.0)")

dist.destroy_process_group()
```

### Example 2: Matrix Factorization (BandMF) with DDP

**Identical pattern with correlated noise:**

```python
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.func import functional_call
from opaque.clipping import clipped_grad
from opaque.distributed import sum_gradients
from opaque.noise import band_mf_noise  # ← Only difference!
from opaque.sampling import PoissonSampler

# Initialize distributed (same as Gaussian)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Model setup (same as Gaussian)
model = MyModel().to(device)
model = DDP(model, device_ids=[rank])
params = {k: v.detach() for k, v in model.module.named_parameters()}

# Clipping (same as Gaussian)
def loss_fn(params, batch):
    x, y = batch
    logits = functional_call(model.module, params, (x,))
    return F.cross_entropy(logits, y)

clipped_grad_fn, clip_state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_size=32,
)

# Create gradient template for MF
grad_template = {k: torch.zeros_like(v) for k, v in params.items()}

# Correlated noise: Same API as gaussian_noise!
# No seed needed - automatically uses same seed on all devices!
noise_fn, noise_state = band_mf_noise(
    grad_template,
    n=1000,           # Total training steps
    bands=4,          # Correlation bands
    stddev=1.1,       # Same as Gaussian
)

# Sampler (same as Gaussian)
from opaque.random import key
sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_fn,
)

# Optimizer (same as Gaussian)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop (IDENTICAL to Gaussian!)
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        # 1. Compute clipped gradients (local data)
        grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
        
        # 2. Aggregate across devices (SUM)
        grads = sum_gradients(grads)
        
        # 3. Add correlated noise (same noise on all devices!)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        
        # 4. Update parameters
        for (name, p), g in zip(model.named_parameters(), noisy_grads.values()):
            p.grad = g.to(p.dtype)
        
        optimizer.step()
        optimizer.zero_grad()
        params = {k: v.detach() for k, v in model.module.named_parameters()}

dist.destroy_process_group()
```

**Key Takeaways:**

✅ **Identical training loop** - Only difference is noise function initialization  
✅ **Same seed management** - Both automatically synchronize seeds in distributed mode  
✅ **Same flow** - Clip → Aggregate → Add Noise → Update  
✅ **Drop-in replacement** - Switch between Gaussian and MF by changing 1 line  

### All Noise Mechanisms - Quick Reference

**All noise mechanisms work identically in DDP. Only the initialization differs:**

```python
import torch.distributed as dist
from opaque.noise import (
    gaussian_noise, band_mf_noise, blt_mf_noise, 
    dense_mf_noise, custom_mf_noise, identity_mf_noise
)

# Initialize distributed
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Model + clipping setup (same for all)
model = DDP(MyModel().to(device), device_ids=[rank])
params = {k: v.detach() for k, v in model.named_parameters()}
grad_template = {k: torch.zeros_like(v) for k, v in params.items()}
clipped_grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_size=32)

# Choose ONE noise mechanism (no seed needed - automatically syncs in DDP!):
# ============================================================================

# 1. Gaussian Noise (Standard DP-SGD)
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# 2. BandMF (Banded Toeplitz) - 10-50% better utility
noise_fn, noise_state = band_mf_noise(grad_template, n=1000, bands=4, stddev=1.1)

# 3. BLT (Buffered Linear Toeplitz) - State-of-the-art
noise_fn, noise_state = blt_mf_noise(grad_template, n=10000, stddev=1.1, min_buffers=1, max_buffers=5)

# 4. Dense MF - Best for small n (< 100 steps)
noise_fn, noise_state = dense_mf_noise(grad_template, n=100, stddev=1.1)

# 5. Custom MF - Your own strategy matrix
strategy_matrix = torch.eye(1000)  # Your C^{-1}
noise_fn, noise_state = custom_mf_noise(grad_template, noising=strategy_matrix, stddev=1.1)

# 6. Identity MF - DP-SGD via MF API
noise_fn, noise_state = identity_mf_noise(grad_template, stddev=1.1)

# Training loop (IDENTICAL for all 6 mechanisms!)
for batch in dataloader:
    batch = tuple(t.to(device) for t in batch)
    
    # 1. Clip gradients
    grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
    
    # 2. Aggregate
    grads = sum_gradients(grads)
    
    # 3. Add noise (mechanism determined by initialization above)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # 4. Update
    for (name, p), g in zip(model.named_parameters(), noisy_grads.values()):
        p.grad = g.to(p.dtype)
    optimizer.step()
    optimizer.zero_grad()
    params = {k: v.detach() for k, v in model.named_parameters()}
```

**Comparison Table:**

| Mechanism | Initialization Complexity | Memory | Utility vs Gaussian | Best For |
|-----------|---------------------------|--------|---------------------|----------|
| `gaussian_noise` | ★☆☆☆☆ (simplest) | O(1) | Baseline | Quick start, benchmarking |
| `identity_mf_noise` | ★★☆☆☆ | O(1) | Same as Gaussian | Testing MF infrastructure |
| `band_mf_noise` | ★★★☆☆ | O(bands) | +10-50% | **Recommended default** |
| `blt_mf_noise` | ★★★★☆ | O(buffers) | +20-60% | Long training (n > 5000) |
| `dense_mf_noise` | ★★☆☆☆ | O(n²) | Optimal | Small n (< 100 steps) |
| `custom_mf_noise` | ★★★★★ | O(matrix) | Varies | Research, custom strategies |

All mechanisms support the same features:
- ✅ Automatic distributed seed synchronization
- ✅ Reproducible with explicit `key=key(42)`
- ✅ Works with all optimizers (SGD, Adam, etc.)
- ✅ Compatible with all sampling strategies

See [Matrix Factorization Guide](matrix-factorization.md) for detailed explanations of each mechanism.  

### Recommended: Sharded Poisson Sampling

**Simplified version without explicit DDP wrapper:**

**Standard approach with simple accounting and automatic seed management:**

```python
import torch.distributed as dist
from opaque.clipping import clipped_grad
from opaque.distributed import sum_gradients, get_rank
from opaque.noise import gaussian_noise
from opaque.sampling import PoissonSampler
from opaque.random import key

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
sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))

# Training loop
for batch in dataloader:
    # 1. Compute clipped gradients on each device's shard
    grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
    
    # 2. Aggregate across devices
    grads = sum_gradients(grads)
    
    # 3. Add synchronized noise on ALL devices (no manual seed management)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # 4. Update parameters
    params = optimizer_update(params, noisy_grads)
```

**Advantages:**
- ✅ **Simple accounting**: Standard DP-SGD, textbook algorithms
- ✅ **No duplicates**: Sharded data ensures examples don't repeat across devices
- ✅ **Automatic**: Both sampler and noise auto-detect distributed mode
- ✅ **No seed management**: Just pass `key=None` (or omit it) and it works across all devices

!!! abstract "See Also: Privacy Accounting"
    For complete accounting example, see [Privacy Accounting](#privacy-accounting) section below.

### Advanced: Independent Poisson Sampling

**No synchronization needed, but requires parallel Poisson accounting:**

```python
from opaque.sampling import PoissonSampler

# Noise: automatically synchronized across devices when distributed
# No seed management needed!
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Independent sampling: each device samples full dataset
# Seed shifted by rank for independent RNG streams (if provided)
from opaque.random import key
sampler = PoissonSampler(
    dataset,
    sample_rate=0.01,
    distributed=False,
    key=key(42),
)

# Training loop (same as sharded)
for batch in dataloader:
    grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
    grads = sum_gradients(grads, op="sum")
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = optimizer_update(params, noisy_grads)
```

**Advantages:**
- ✅ **No synchronization**: Each device samples independently (no sharding coordination)
- ✅ **Limited privacy cost**: Parallel Poisson accounting adds minimal privacy overhead in practice
- ✅ **Simple setup**: No need to partition dataset across devices

**Trade-off:**
- Examples may appear multiple times across devices (handled by parallel Poisson accounting)

!!! warning "Advanced Feature - Requires Parallel Poisson Accounting"
    Independent sampling uses **parallel Poisson accounting** to handle examples appearing on multiple devices:
    
    ```python
    # Standard (sharded): One chance per example
    step = acc.poisson(acc.gaussian(noise_mult), sample_rate)
    
    # Independent (parallel): world_size chances per example
    step = acc.parallel_poisson(
        acc.poisson(acc.gaussian(noise_mult), sample_rate),
        num_workers=world_size,
    )
    ```
    
    See [Privacy Accounting](#privacy-accounting) section for complete calibration example.
```

**Launch with**:
```bash
torchrun --nproc_per_node=4 train.py
```

!!! tip "Matrix Factorization with DDP"
    
    **Correlated noise mechanisms** (BandMF, BLT) also support distributed training with the same centralized pattern:
    
    ```python
    from opaque.noise import band_mf_noise
    
    # Automatically uses same seed on all devices (centralized pattern)
    noise_fn, noise_state = band_mf_noise(
        grad_template, 
        n=1000, 
        bands=4, 
        stddev=1.1,
    )
    
    # Training loop identical to Gaussian noise
    for batch in dataloader:
        grads = clipped_grad_fn(params, batch)
        grads = sum_gradients(grads)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        params = optimizer_update(params, noisy_grads)
    ```
    
    For details, see [Matrix Factorization Guide - Distributed Training](matrix-factorization.md#distributed-training-ddp).

## Key Concepts

### How Distributed DP-SGD Works

**Critical insight**: All devices generate **the SAME noise** and add it to their local copy of the **aggregated** gradient. The noise is **not summed** across devices.

**Step-by-step example with 2 GPUs:**

```python
# Step 1: Each device computes clipped gradients on its local data
Device 0: clipped_grad = [1.0, 2.0, 3.0]  # From batch_0
Device 1: clipped_grad = [4.0, 5.0, 6.0]  # From batch_1

# Step 2: AllReduce SUM aggregates gradients
# Both devices now have the SAME aggregated gradient:
Device 0: aggregated = [5.0, 7.0, 9.0]  # 1+4, 2+5, 3+6
Device 1: aggregated = [5.0, 7.0, 9.0]  # (identical)

# Step 3: Each device generates IDENTICAL noise (same seed)
Device 0: noise = [0.1, 0.2, 0.3]  # torch.randn(..., seed=seed_0)
Device 1: noise = [0.1, 0.2, 0.3]  # Same seed → same noise!

# Step 4: Each device adds noise to its local copy
Device 0: noisy_grad = [5.1, 7.2, 9.3]  # aggregated + noise
Device 1: noisy_grad = [5.1, 7.2, 9.3]  # (identical)

# Step 5: Optimizer updates parameters
# Both devices have identical parameters after update
```

**Key points:**

- ✅ **Same noise everywhere** - Prevents model divergence (parameters stay synchronized)
- ✅ **Noise NOT summed** - We don't sum noise across devices (would give √k scaling)
- ✅ **Conceptually noise after aggregation** - But physically added on each device
- ✅ **Standard privacy accounting** - Noise magnitude is exactly as calibrated

**What if we used DIFFERENT seeds per device?** ❌

```python
# BAD: Different noise per device
Device 0: noise = [0.1, 0.2, 0.3]  # seed=42
Device 1: noise = [0.9, 0.8, 0.7]  # seed=43 (DIFFERENT!)

Device 0: noisy_grad = [5.1, 7.2, 9.3]
Device 1: noisy_grad = [5.9, 7.8, 9.7]  # Different from Device 0!

# Result: Models DIVERGE! Parameters become different across devices.
# Training fails because devices compute different updates.
```

**What if we added noise BEFORE aggregation?** ❌

```python
# BAD: Add noise before AllReduce (local noise addition)
Device 0: clipped = [1.0, 2.0, 3.0] + noise_0 = [1.1, 2.2, 3.3]
Device 1: clipped = [4.0, 5.0, 6.0] + noise_1 = [4.9, 5.8, 6.7]

# AllReduce SUM accumulates BOTH gradients AND noise
aggregated = [6.0, 8.0, 10.0]  # Combined gradient + noise

# Problem: Noise magnitude scales by √k where k = num devices
# With 4 GPUs: noise is 2x larger, with 16 GPUs: 4x larger!
# Privacy accounting must use parallel Poisson (more complex)
```

**Our approach (central noise addition)** ✅

```python
# GOOD: Aggregate THEN add noise
Device 0: clipped = [1.0, 2.0, 3.0]  # No noise yet
Device 1: clipped = [4.0, 5.0, 6.0]  # No noise yet

# AllReduce SUM (only gradients)
aggregated = [5.0, 7.0, 9.0]

# Each device adds SAME noise to aggregated gradient
both_devices: noisy = [5.0, 7.0, 9.0] + [0.1, 0.2, 0.3] = [5.1, 7.2, 9.3]

# Benefits:
# ✅ Noise magnitude is exactly as calibrated (no scaling)
# ✅ Standard DP-SGD accounting (no parallel Poisson needed)
# ✅ Parameters stay synchronized (no divergence)
```

### Automatic Seed Management in Distributed Training

Opaque automatically handles seed management to prevent common distributed training issues:

**When no seed is provided** (`key=None`):
- **Sampler**: Uses unseeded RNG (different samples per run)
- **Noise**: Auto-detects distributed mode and uses **same seed (0)** across all devices for synchronized noise

**When seed is provided** (e.g., `key=key(42)`):
- **Sampler**: Automatically shifts seed by rank for independent sampling
- **Noise**: Uses seed as-is (user responsible for consistency)

**Result**: The API is simple - just pass `key=None` (or omit it) and everything works:
- ✅ Sampler gives different data per device (rank shifting)
- ✅ Noise is synchronized across devices (same seed)
- ✅ No model divergence, no data duplication

### Two Poisson Sampling Strategies for Distributed DP

The choice is **NOT between different vs same noise** (both use same noise to avoid model divergence), but between **sampling strategies**:

#### Strategy 1: Sharded Poisson Sampling (Recommended)

**Data partitioning**: Each device samples from disjoint data partition

```python
# Noise: automatically synchronized when distributed (no seed needed)
from opaque.random import key
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Sampler: seed automatically shifted by rank (42 → 42, 43, 44, ...)
sampler = PoissonSampler(
    dataset,
    sample_rate=0.01,
    distributed=True,
    key=key(42),
)

# Device 0: samples with seed 42, gets examples 0-49999 (5% of shard)
# Device 1: samples with seed 43, gets examples 50000-99999 (different 5%)
# → No duplication, independent sampling, synchronized noise

for batch in dataloader:
    grads = clipped_grad_fn(params, batch)
    grads = sum_gradients(grads)  # Aggregate across devices
    noisy_grads = noise_fn(grads, noise_state)  # Synchronized noise
```

**Characteristics:**
- Each device processes **disjoint data** (no duplicates across devices)
- Guaranteed to see different data across training
- Requires synchronization to partition dataset
- **Simple accounting**: Standard DP-SGD (no parallel Poisson needed)
- **[See accounting example →](#privacy-accounting)**

#### Strategy 2: Independent Poisson Sampling (Advanced)

**Independent sampling**: Each device samples full dataset independently

```python
# Noise: automatically synchronized when distributed (no seed needed)
from opaque.random import key
noise_fn, noise_state = gaussian_noise(stddev=1.1)

# Sampler: seed automatically shifted by rank (42 → 42, 43, 44, ...)
sampler = PoissonSampler(
    dataset,
    sample_rate=0.01,
    distributed=False,
    key=key(42),
)

# Device 0: samples with seed 42: examples 0, 15, 42, ... (5% of 1M)
# Device 1: samples with seed 43: examples 3, 12, 100, ... (5% of 1M, different)
# → Some examples may appear in both device 0 and device 1 (handled by parallel Poisson)

for batch in dataloader:
    grads = clipped_grad_fn(params, batch)
    grads = sum_gradients(grads)  # Aggregate across devices
    noisy_grads = noise_fn(grads, noise_state)  # Synchronized noise
    # (no sharding synchronization needed!)
```

**Characteristics:**
- Each device samples **independently from full dataset**
- Examples may appear in **multiple devices** (parallel Poisson property)
- **No synchronization needed** for data partitioning
- **Parallel Poisson accounting**: Handles duplicate examples across devices
- **Limited privacy cost** in practice despite duplicates
- **[See accounting example →](#privacy-accounting)**

!!! warning "Noise Synchronization: Automatic with Sensible Defaults"
    
    By default, **noise is automatically synchronized across devices**:
    
    ```python
    # ✅ DEFAULT (no seed needed): All devices use same seed (0)
    noise_fn, noise_state = gaussian_noise(stddev=1.1)
    ```
    
    If you provide an explicit seed, it's used as-is:
    
    ```python
    # ✅ EXPLICIT SEED: Same seed on all devices (user provides it)
    from opaque.random import key, fold_in
    noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))
    
    # ❌ DIFFERENT PER DEVICE (causes model divergence - avoid!)
    # Don't manually shift by rank for noise:
    noise_fn = gaussian_noise(stddev=1.1, key=fold_in(key(42), rank))  # Wrong!
    ```

!!! tip "Which Sampling Strategy?"
    
    **Choose SHARDED if:**
    - Simple accounting is preferred (standard DP-SGD)
    - You can coordinate dataset partitioning
    - You need predictable data distribution
    
    **Choose INDEPENDENT if:**
    - Avoiding synchronization complexity is important
    - Parallel Poisson accounting is acceptable
    - Very limited privacy overhead in practice
    
    **For accounting details**, see the [Privacy Accounting](#privacy-accounting) section below.

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
    - **Independent**: Examples may appear in multiple devices (handled by parallel Poisson)

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
    grads = sum_gradients(grads)  # Sum across devices
    
    # All devices use same noise (synchronized)
    noisy_grads = noise_fn(grads, noise_state)
    
    # Update with fixed learning rate
    params = params - lr * noisy_grads
```

**For independent sampling** (full dataset per device):
- Each device sees full dataset but samples independently
- Requires parallel Poisson accounting for handling duplicates

```python
# Independent sampling (each device sees full dataset)
sampler = PoissonSampler(dataset, sample_rate=0.01, distributed=False)

for batch in dataloader:
    grads = clipped_grad_fn(params, batch)
    grads = sum_gradients(grads)
    
    # All devices use same noise (synchronized)
    noisy_grads = noise_fn(grads, noise_state)
    
    # Update
    params = params - lr *noisy_grads
```

**⚠️ Important: Use `sum`, NOT `average`**

`reduce_pytree(op="mean")` divides by `world_size`, but Poisson sampling creates **different total batch sizes per step**:

```python
# WRONG: Dividing by world_size instead of total batch size
grads, _ = reduce_pytree(grads, op="mean")  # Divides by 4
# Effective update: (sum of 6+10+7+9=32 clipped grads) / 4

# ✅ CORRECT: Sum without dividing
grads = sum_gradients(grads)
# Effective update: sum of all clipped grads (not normalized)
```
    
    **Use independent sampling** (`distributed=False`) with variable batch sizes for proper DP guarantees.

## API Reference

### Gradient Aggregation

Opaque provides two functions for gradient aggregation across GPUs:

#### `sum_gradients()`

**Primary function** for DP training - sums clipped gradients across devices:

```python
from opaque.distributed import sum_gradients

# After computing local clipped gradients
clipped_grads = clipped_grad_fn(params, batch)

# Sum across all GPUs
grads = sum_gradients(clipped_grads)
```

**Why sum, not average?**
- DP-SGD needs total of clipped gradients before noise
- Sensitivity is C (clip norm), not C/world_size
- Noise calibration: σ = noise_multiplier * C

#### `reduce_pytree()`

**Generic reduction** for any PyTree of tensors (not just gradients):

```python
from opaque.distributed import reduce_pytree

# Sum operation (default)
result, _ = reduce_pytree(pytree, op="sum")

# Mean operation
result, _ = reduce_pytree(pytree, op="mean")

# Other operations: "max", "min", "product"
```

**Use `reduce_pytree` when you need:**
- Averaging metrics or scalars
- Custom reduction operations
- Non-gradient aggregation

!!! tip "Use `sum_gradients()` for DP Training"
    **For differential privacy**, always use `sum_gradients()`:
    
    - ✅ **`sum_gradients(grads)`**: Correct for DP-SGD
        - No division by world_size
        - Preserves DP sensitivity (C, not C/world_size)
        - Works with variable batch sizes (Poisson sampling)
        - Standard privacy accounting
    
    - ❌ **Averaging is WRONG for DP**:
        - Changes gradient scaling incorrectly
        - Breaks privacy sensitivity calculations
        - Incompatible with proper noise calibration
    
    **Privacy accounting:** Use total examples across all devices per step.
    
    **What `clipped_grad` returns:** The **SUM** of clipped per-example gradients from the local batch (B examples → sum of B clipped grads).

### State Synchronization

Synchronize optimizer state or other values across GPUs:

```python
from opaque.distributed import sync_state, reduce_scalar

# Synchronize dataclass state (e.g., adaptive clipping)
clip_state = sync_state(clip_state)

# Reduce single scalar value
clip_norm = reduce_scalar(clip_norm)
```

### Optimizer State Synchronization (TorchOpt)

**TL;DR:** TorchOpt optimizer states stay synchronized automatically in DDP—no manual sync needed.

#### How It Works

TorchOpt follows JAX's Optax functional design:

```python
import torchopt

# Initialize optimizer and state
opt = torchopt.adam(lr=1e-3)
opt_state = opt.init(params)  # State: (ScaleByAdamState(mu, nu, count), EmptyState())

# Update step (pure function)
updates, opt_state = opt.update(grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

**Key property:** `opt.update()` is a **pure function**—given the same inputs (grads, state, params), it always produces the same outputs.

#### Why States Stay Synchronized in DDP

1. **Identical initial states:** All devices call `opt.init(params)` with the same parameters → identical `opt_state` on all devices
2. **Identical gradients:** After `sum_gradients()` and noise injection (same seed), all devices have identical `noisy_grads`
3. **Deterministic updates:** `opt.update(noisy_grads, opt_state)` is pure → produces identical `opt_state` on all devices

**Result:** Optimizer states evolve identically across all devices without explicit synchronization.

#### Validation

This assumption is theoretically sound but **never empirically validated** in Opaque's test suite. Key risks:

- **Floating-point drift:** Accumulation errors over many steps (especially with FP16)
- **Non-determinism:** If any device computes gradients differently (rare with DDP)
- **Bugs:** Implementation errors in gradient aggregation or noise injection

**Recommendation:** For production use, consider adding validation by:

```python
# Periodically check optimizer state drift across devices
if step % 100 == 0 and distributed:
    from opaque.distributed import gather_tensors
    
    # Gather first momentum tensor from all devices
    local_mu = opt_state[0].mu[0]  # First parameter's momentum
    all_mus = gather_tensors(local_mu, rank, world_size)
    
    if rank == 0:
        # Check if all devices have identical state
        max_diff = max(torch.max(torch.abs(all_mus[i] - all_mus[0])) for i in range(1, world_size))
        if max_diff > 1e-5:
            print(f"⚠️ Warning: Optimizer state drift detected: {max_diff:.2e}")
```

### TorchOpt.distributed vs opaque.distributed

**Important:** TorchOpt has a `torchopt.distributed` module—this is **NOT for DDP training**.

#### TorchOpt.distributed (RPC-based)

**Design:** Master-worker parameter server parallelism via PyTorch RPC

**API:**
```python
from torchopt.distributed import parallelize, mean_reducer, sum_reducer

# Parallelize a function across RPC workers
@parallelize(partitioner=batch_partitioner, reducer=mean_reducer)
def distributed_forward(params, batch):
    return model(params, batch)
```

**Use case:** Synchronous distributed function evaluation with custom reducers (research/experimental)

#### opaque.distributed (DDP-focused)

**Design:** Peer-to-peer AllReduce with NCCL backend (production standard)

**API:**
```python
from opaque.distributed import sum_gradients, reduce_pytree, sync_state

# Aggregate gradients across all GPUs
grads = sum_gradients(grads)  # Uses dist.all_reduce under the hood
```

**Use case:** Standard data-parallel DDP training (recommended for DP-SGD)

#### Why We Don't Integrate TorchOpt.distributed

**Architectural mismatch:**

- **TorchOpt.distributed:** Targets function-level parallelization with RPC (like JAX's `pmap` with explicit device assignment)
- **opaque.distributed:** Targets gradient-level synchronization with AllReduce (like PyTorch's native DDP)

**Practical reasons:**

- **Performance:** AllReduce (NCCL) is optimized for gradient aggregation; RPC adds overhead
- **Simplicity:** DDP is standard practice; RPC requires complex orchestration
- **DP-SGD patterns:** Gradient summing + noise injection fits naturally with AllReduce

**User takeaway:** Use `opaque.distributed` for DDP training, ignore `torchopt.distributed` (it's for different use cases).

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
from opaque.distributed import sum_gradients, get_rank
from opaque.noise import gaussian_noise
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
    
    # Noise: Same seed on all devices for synchronized noise
    noise_fn, noise_state = gaussian_noise(stddev=1.1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Each GPU processes different data
        local_dataloader = get_local_dataloader(rank, world_size)
        
        for batch in local_dataloader:
            # 1. Compute clipped gradients locally
            grads, clip_state = grad_fn(
                trainable, batch, state=clip_state
            )
            
            # 2. Sum gradients across GPUs
            if distributed:
                grads = sum_gradients(grads)
            
            # 3. Add noise (same noise on all devices!)
            noisy_grads, noise_state = noise_fn(grads, noise_state)
            
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

Privacy accounting is the same for DDP as single-device training—use the same `sample_rate` for accounting regardless of how data is distributed.

### Sharded Sampling (Recommended)

**Standard DP-SGD accounting** - identical to single-device:

```python
import opaque.accounting as acc
from opaque.accounting import calibration as cal

sample_rate = 0.01  # Same rate used in sampler
num_steps = 1000

# Calibrate noise for target (ε, δ)
if rank == 0:
    budget = cal.epsilon_budget(3.0, delta=1e-5)
    result = cal.calibrate(
        budget,
        lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
        param_min=0.5, param_max=5.0,
    )
    noise_multiplier = result.param
    step_process = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)
    acct = Accountant()

# Broadcast noise multiplier to all devices
if world_size > 1:
    noise_tensor = torch.tensor([noise_multiplier], device=device)
    dist.broadcast(noise_tensor, src=0)
    noise_multiplier = noise_tensor.item()

# Training loop with privacy tracking
for step_idx, batch in enumerate(dataloader):
    # ... clip → aggregate → add noise → update ...
    
    # Track privacy (only on main)
    if rank == 0:
        acct = acct | step_process
        if step_idx % 100 == 0:
            eps = acct.epsilon_at(1e-5)
            print(f"Step {step_idx}: ε={eps:.2f}")
```

**Why it's the same:** Sharded sampling means each example appears on exactly one device, so from the DP perspective it's identical to single-device sampling at `sample_rate`.

### Independent Sampling (Advanced)

**Parallel Poisson accounting** - models sampling each example multiple times:

```python
# Each device samples full dataset independently
# Example may appear on k devices (k ∈ [0, world_size])

from opaque.accounting.accountant import Accountant

if rank == 0:
    # Binary search for noise multiplier
    budget = cal.epsilon_budget(3.0, delta=1e-5)
    result = cal.calibrate(
        budget,
        lambda nm: acc.parallel_poisson(
            acc.poisson(acc.gaussian(nm), sample_rate),
            num_workers=world_size
        ) * num_steps,
        param_min=0.5, param_max=5.0,
    )
    noise_multiplier = result.param
    
    # Track with parallel Poisson process
    step_process = acc.parallel_poisson(
        acc.poisson(acc.gaussian(noise_multiplier), sample_rate),
        num_workers=world_size
    )
    acct = Accountant()

# Training loop
for step_idx, batch in enumerate(dataloader):
    # ... training ...
    if rank == 0:
        acct = acct | step_process
        if step_idx % 100 == 0:
            eps = acct.epsilon_at(1e-5)
            print(f"Step {step_idx}: ε={eps:.2f}")
```

**Trade-off:** Parallel Poisson accounting is slightly more conservative (higher ε for same noise) because examples can be selected by multiple devices.

!!! warning "Matrix Factorization (BandMF, BLT, etc.)"
    **Privacy accounting for MF noise mechanisms is not yet implemented.** Use MF for improved utility, but track privacy using Gaussian accounting with the same noise multiplier as a conservative estimate.
    
    ```python
    # For MF noise
    noise_fn = band_mf_noise(grad_template, n=1000, bands=4, stddev=1.1)
    
    # Track privacy using Gaussian mechanism (conservative)
    from opaque.accounting.accountant import Accountant

    step_process = acc.poisson(acc.gaussian(1.1), sample_rate)
    acct = Accountant()
    for step in training:
        acct = acct | step_process  # Conservative bound
    ```

## Adaptive Clipping with DDP

Adaptive clipping requires state synchronization across GPUs:

```python
from opaque.clipping import adaptive_clipped_grad
from opaque.distributed import sum_gradients

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
    
    # Add noise and sum
    noisy_grads = noise_fn(grads)
    if distributed:
        noisy_grads = sum_gradients(noisy_grads)
    
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

1. **Single-node only** - Multi-node DDP not extensively tested (but should work)
2. **NCCL backend recommended** - Other backends (Gloo, MPI) may work but are not tested
3. **FSDP not yet supported** - Fully Sharded Data Parallel requires additional implementation

## Next Steps

- **[Matrix Factorization](matrix-factorization.md)** - Correlated noise (BandMF, BLT) with DDP support
- **[Memory Profiling](memory-profiling.md)** - Optimize memory usage for multi-GPU
- **[LoRA Fine-tuning](lora.md)** - Parameter-efficient training with DP

---

**Questions?** Open an issue on [GitHub](https://github.com/JetBrains-Research/opaque/issues)
