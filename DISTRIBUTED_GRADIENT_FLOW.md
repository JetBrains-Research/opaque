# Distributed Gradient Flow: Poisson Sampling Requirement

## Critical: Why Variable Batch Sizes Are Required

**For proper differential privacy with Poisson sampling**, batch sizes MUST be variable across steps. Fixed batch sizes break privacy amplification guarantees.

## What Clipping Returns

**`clipped_grad` by default returns the SUM of clipped per-example gradients:**

```python
# Single device with batch_size=8
grads = clipped_grad_fn(params, batch)  # batch has 8 examples

# Internally:
# 1. Compute 8 per-example gradients via vmap
# 2. Clip each to max L2 norm = C
# 3. SUM the 8 clipped gradients
# Result: grads = sum of 8 clipped gradients (single gradient dict)
```

## Distributed Training with Poisson Sampling (Variable Batch Sizes)

**Privacy amplification requires independent Poisson sampling on each device:**

```python
# Each device samples DIFFERENT data → different batch sizes
# This is REQUIRED for privacy amplification theorems
sampler = PoissonSampler(dataset, sample_rate=0.01, distributed=False)

# Device 0: batch_size=6  → grads_0 = sum of 6 clipped grads
# Device 1: batch_size=10 → grads_1 = sum of 10 clipped grads  
# Device 2: batch_size=7  → grads_2 = sum of 7 clipped grads
# Device 3: batch_size=9  → grads_3 = sum of 9 clipped grads
```
import torch
from opaque.distributed import all_reduce_gradients, sync_scalar

# Count examples on this device
batch_size = len(batch)

# Sum gradients and batch sizes
grads = all_reduce_gradients(grads, op="sum")
total_batch_size = sync_scalar(batch_size, op="sum", device=device)

# Divide by actual total count
grads = {k: v / total_batch_size for k, v in grads.items()}

# Update with fixed effective LR per example
params = params - lr * grads  
```

**Tradeoff**: Extra communication (sync_scalar) for stable LR per example.

## Why Not Fixed Batch Sizes?

**❌ Coordinated sampling breaks privacy amplification:**

```python
# BAD: All devices get same indices
sampler = PoissonSampler(dataset, sample_rate=0.01, distributed=True)
# or DistributedSampler with fixed batch size

# Problem: Removes independent sampling property
# No known tight privacy bounds for synchronized Poisson sampling
```

**Privacy amplification theorems assume each example is sampled independently with probability p.** Coordinating sampling across devices creates correlation that breaks this assumption.

## Example Code (Correct)

```python
from opaque.distributed import all_reduce_gradients
from opaque.sampling import PoissonSampler

# Independent Poisson sampling (variable batches)
sampler = PoissonSampler(dataset, sample_rate=0.01, distributed=False)
# Result: grads = sum of 32 clipped gradients (not divided by anything)
```

**Learning rate equivalence:**
- Same as single-GPU with batch_size=32 and learning rate divided by world_size
- Example: `lr=0.01` → effective step is `0.01 * sum` vs single-GPU `lr=0.01/4` → `0.0025 * sum`
- To match single-GPU behavior: use `lr / world_size` with distributed sum

## Which to Use?

**Use `average_gradients()` (Option 1)** because:
1. ✅ Same learning rate as single-GPU training
2. ✅ Easier to reason about (effective LR doesn't change with num GPUs)
3. ✅ Standard PyTorch DDP convention
4. ✅ Works with adaptive optimizers (Adam, etc.) without adjustment

**Privacy accounting is the SAME** for both:
- Effective batch size = 32 (local_batch_size × num_gpus)
- Noise scale is determined by this effective batch size
- Clipping norm C is per-example, not affected by aggregation

## Key Insight

**Averaging vs summing only affects the learning rate scale!**

For privacy:
- What matters: effective batch size = 32
- Noise stddev: `σ = C × noise_multiplier` 
- Privacy accounting: uses batch_size=32

For optimization:
- Average: `params -= lr * (avg of 32 grads)` 
- Sum: `params -= lr * (sum of 32 grads)` = `params -= (lr * world_size) * (avg of 32 grads)`

They're mathematically equivalent if you adjust learning rate accordingly.

## Example Code (Correct)

```python
# Clip locally (returns SUM of local clipped grads)
grads = clipped_grad_fn(params, local_batch)  # local_batch_size=8

# Add noise (independent strategy)
noisy_grads = noise_fn(grads, noise_state)

# Average across devices (divide by world_size=4)
noisy_grads = average_gradients(noisy_grads)  # Now: avg of 32 clipped+noisy grads

# Update with same learning rate as single-GPU
lr = 0.01
params = params - lr * noisy_grads  # Equivalent to single-GPU with batch=32
```
Training loop
for batch in dataloader:
    # Clip locally (returns SUM of local clipped grads)
    grads = clipped_grad_fn(params, batch)  # Variable batch_size

    # Add noise (independent strategy)
    noisy_grads = noise_fn(grads, noise_state)

    # Sum across devices (NOT average!)
    noisy_grads = all_reduce_gradients(noisy_grads, op="sum")

    # Update (effective LR varies with total batch size)
    params = params - lr * noisy_grads