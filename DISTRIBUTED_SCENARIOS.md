# Distributed DP-SGD: Two Noise Strategies

## Key Principle: DP-SGD Adds Noise to SUM

Standard DP-SGD algorithm:
```
1. Compute per-example gradients: g_1, g_2, ..., g_B
2. Clip each: ḡ_i = clip(g_i, C)
3. Sum: G = Σ ḡ_i  (sensitivity = C)
4. Add noise: G̃ = G + N(0, σ²C²I)
5. Update: θ = θ - η · G̃
```

**Why sum?** The sensitivity of the sum is C (adding/removing one example changes the sum by ≤ C).

---

## Scenario 1: Independent Noise (Privacy Amplification) ✅ Recommended

**Each device adds noise to its local sum BEFORE aggregation.**

### Single Device Baseline
```python
# Batch size = 32
grads = clipped_grad_fn(params, batch)  # Sum of 32 clipped grads
noisy_grads = noise_fn(grads, noise_state)  # Add N(0, σ²C²I)
params = params - lr * noisy_grads
```

### Distributed (4 devices, Poisson sampling)
```python
from opaque.distributed import all_reduce_gradients, get_rank
from opaque.sampling import PoissonSampler

# Independent Poisson sampling (variable batch sizes)
sampler = PoissonSampler(dataset, sample_rate=0.01, distributed=False)

# Different seed per device
noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=42 + get_rank())

for batch in dataloader:  # batch_size varies: e.g., 6, 10, 7, 9
    # Step 1: Clip and sum locally
    grads = clipped_grad_fn(params, batch)  # Sum of B_k clipped grads
    
    # Step 2: Add noise to local sum
    # Device 0: sum of 6 + noise → sensitivity = C
    # Device 1: sum of 10 + noise → sensitivity = C
    # Device 2: sum of 7 + noise → sensitivity = C
    # Device 3: sum of 9 + noise → sensitivity = C
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # Step 3: Sum noisy gradients across devices
    noisy_grads = all_reduce_gradients(noisy_grads, op="sum")
    # Total: sum of 32 clipped grads + 4 independent noise samples
    
    # Step 4: Update
    params = params - lr * noisy_grads
```

**Privacy accounting:**
- Each device adds noise with variance σ²C²
- Total noise variance: 4 × σ²C² (independent noise adds in quadrature)
- **Privacy amplification:** Better bounds via parallel composition
- Use total examples per step (sum of local batch sizes) for accounting

---

## Scenario 2: Shared Noise (Mixture Gaussian)

**All devices aggregate FIRST, then add noise ONCE to the global sum.**

### Distributed (4 devices, Poisson sampling)
```python
from opaque.distributed import all_reduce_gradients
from opaque.sampling import PoissonSampler

# Independent Poisson sampling
sampler = PoissonSampler(dataset, sample_rate=0.01, distributed=False)

# Same seed on all devices
noise_fn, noise_state = gaussian_noise(stddev=1.1, generator=42)

for batch in dataloader:  # batch_size varies: e.g., 6, 10, 7, 9
    # Step 1: Clip and sum locally
    grads = clipped_grad_fn(params, batch)
    # Device 0: sum of 6 clipped grads
    # Device 1: sum of 10 clipped grads
    # Device 2: sum of 7 clipped grads
    # Device 3: sum of 9 clipped grads
    
    # Step 2: Sum across devices
    grads = all_reduce_gradients(grads, op="sum")
    # Global sum: sum of 32 clipped grads (sensitivity = C)
    
    # Step 3: Add noise to global sum (same seed → same noise on all devices)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    # Total: sum of 32 clipped grads + 1 noise sample
    
    # Step 4: Update (all devices have same noisy gradient)
    params = params - lr * noisy_grads
```

**Privacy accounting:**
- Single noise addition with variance σ²C²
- **Standard DP-SGD accounting** (mixture Gaussian)
- Use total examples per step for accounting

---

## Helper Functions We Provide

### Core Aggregation

**`all_reduce_gradients(grads, op="sum")`** ✅
- Sum PyTrees across all devices
- **Use this for Poisson sampling**
- Returns: sum of all local gradients
- In-place operation for efficiency

```python
from opaque.distributed import all_reduce_gradients

grads = all_reduce_gradients(grads, op="sum")
# grads is now sum across all devices
```

**`average_gradients(grads)`** ⚠️
- Sums then divides by world_size
- **Only correct if all devices have SAME batch size**
- **Do NOT use with Poisson sampling** (variable batch sizes)
- Kept for compatibility with fixed-batch scenarios

```python
from opaque.distributed import average_gradients

# Only use if batch sizes are identical across devices!
grads = average_gradients(grads)
# grads = (sum across devices) / world_size
```

### State Synchronization

**`sync_scalar(value, op="mean", device=None)`**
- Synchronize a single float/int across devices
- Useful for adaptive clipping state, batch size totals

```python
from opaque.distributed import sync_scalar

# Get total batch size across all devices
batch_size = len(batch)
total_batch_size = sync_scalar(batch_size, op="sum", device=device)
```

**`sync_state(state, sync_fields=None, op="mean", device=None)`**
- Synchronize dataclass fields across devices
- Used for adaptive clipping state synchronization

```python
from opaque.distributed import sync_state

# Sync clip_norm and clipping_rate across devices
clip_state = sync_state(
    clip_state,
    sync_fields=["clip_norm", "clipping_rate"],
    op="mean",
    device=device
)
```

### Device Utilities

**`is_initialized()`** - Check if distributed training is active
**`get_rank()`** - Get current device rank (0 to world_size-1)
**`get_world_size()`** - Get total number of devices
**`barrier()`** - Synchronize all devices

```python
from opaque.distributed import is_initialized, get_rank, get_world_size

if is_initialized():
    rank = get_rank()
    world_size = get_world_size()
    print(f"Device {rank}/{world_size}")
```

---

## Summary Table

| Aspect | Independent Noise | Shared Noise |
|--------|------------------|--------------|
| **When to add noise** | Before aggregation | After aggregation |
| **Seed per device** | `42 + rank` | `42` (same) |
| **Aggregation** | `all_reduce_gradients(op="sum")` | `all_reduce_gradients(op="sum")` |
| **Noise samples** | K (one per device) | 1 (shared) |
| **Privacy bounds** | Better (amplification) | Standard (mixture Gaussian) |
| **Recommended** | ✅ Yes | Use if needed |

**Key insight:** Both scenarios use **sum aggregation** because DP-SGD adds noise to the sum of clipped gradients (sensitivity = C).
