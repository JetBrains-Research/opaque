# Matrix Factorization & DP-FTRL

Matrix factorization mechanisms replace the independent Gaussian noise of standard DP-SGD with **correlated noise** that achieves better utility at the same privacy budget. This is the technique behind DP-FTRL (Differentially Private Follow-The-Regularized-Leader).

## Why Correlated Noise?

In standard DP-SGD, each training step adds independent noise. When you compute running averages or sums (as optimizers do internally), the noise **accumulates** as O(sqrt(n)) over n steps.

Matrix factorization mechanisms instead add noise that is correlated across steps, designed so that the noise **partially cancels out** in the running sum. The result: same privacy guarantee, but lower effective noise on the model parameters.

**Typical improvement**: 10-50% utility gain at the same privacy budget.

## Key Concepts

### The Factorization A = B C

The mechanism factors a **workload matrix** A (what the optimizer computes, e.g., prefix sums) into:

- **C** (strategy matrix): Encodes the privacy mechanism
- **C^{-1}** (noising matrix): Used to generate correlated noise at each step
- **B = A C^{-1}** (decoder matrix): Relates noisy outputs back to workload queries

The **sensitivity** of C determines how much noise is needed, while the **error** of B determines the effective noise on the output.

### Strategy Types

| Strategy | Memory | Best For |
|----------|--------|----------|
| **BandMF** (Banded Toeplitz) | O(bands) | General purpose, good default |
| **BLT** (Buffered Linear Toeplitz) | O(num_buffers) | Large n, state-of-the-art |
| **Dense** | O(n^2) | Small n, exact solutions |

## Quick Start

### Single-Device Training

All MF constructors follow the same `(noise_fn, state)` pattern as `gaussian_noise`:

```python
import torch
from opaque.noise import band_mf_noise
from opaque.random import key

# A gradient template (zeros with the right shapes)
grad_template = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

# Create noise function — grad_template is always the first argument
noise_fn, noise_state = band_mf_noise(
    grad_template,
    n_steps=1000,
    bands=4,
    stddev=noise_multiplier * clip_norm,
    key=key(42),
)

# Training loop
for batch in dataloader:
    clipped_grad = compute_clipped_grad(model, batch)

    # Add correlated noise
    noisy_grad, noise_state = noise_fn(clipped_grad, noise_state)

    # Assign noisy gradients and step
    for (name, p), g in zip(model.named_parameters(), noisy_grad.values()):
        p.grad = g.to(p.dtype)
    optimizer.step()
```

### Quick Comparison: Gaussian vs BandMF

**The only difference is initialization:**

```python
# Gaussian noise (independent per step)
from opaque.noise import gaussian_noise
from opaque.random import key
noise_fn, state = gaussian_noise(stddev=1.1, key=key(42))

# BandMF (correlated across steps)
from opaque.noise import band_mf_noise
grad_template = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
noise_fn, state = band_mf_noise(grad_template, n_steps=1000, bands=4, stddev=1.1, key=key(42))

# Training loop is IDENTICAL for both:
for batch in dataloader:
    clipped_grad = compute_clipped_grad(model, batch)
    noisy_grad, state = noise_fn(clipped_grad, state)  # ← Same call!
    # ... update parameters
```

**When to use which:**

- **Gaussian** (`gaussian_noise`): Standard DP-SGD, simpler, good baseline
- **BandMF** (`band_mf_noise`): 10-50% better utility, requires knowing `n` (total steps) upfront
- **BLT** (`blt_mf_noise`): State-of-the-art utility, best for long training (n > 5000)

All three work identically in distributed training (see below).

## All Noise Mechanisms: Complete Examples

### Single-Device Training

**Complete working examples for all noise mechanisms:**

```python
import torch
import torch.nn.functional as F
from torch.func import functional_call
from opaque.clipping import clipped_grad
from opaque.random import key
from opaque.noise import (
    gaussian_noise,        # Standard DP-SGD
    band_mf_noise,         # BandMF (banded Toeplitz)
    blt_mf_noise,          # BLT (buffered linear Toeplitz)
    dense_mf_noise,        # Dense matrix (small n)
    custom_mf_noise,       # Bring your own matrix
    identity_mf_noise,     # Identity (DP-SGD via MF API)
)

# Setup (same for all mechanisms)
model = MyModel()
params = {k: v for k, v in model.named_parameters()}
grad_template = {k: torch.zeros_like(v) for k, v in params.items()}

def loss_fn(params, batch):
    x, y = batch
    logits = functional_call(model, params, (x,))
    return F.cross_entropy(logits, y)

clipped_grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Choose ONE of these noise mechanisms:
# ==========================================

# 1. Gaussian Noise (Standard DP-SGD) - Simplest baseline
noise_fn, noise_state = gaussian_noise(
    stddev=1.1,
    key=key(42),  # RngKey for reproducibility
)

# 2. BandMF (Banded Toeplitz) - Good default, 10-50% better than Gaussian
noise_fn, noise_state = band_mf_noise(
    grad_template,
    n_steps=1000,     # Total training steps (required)
    bands=4,          # Correlation bands (default: 4)
    stddev=1.1,
    key=key(42),
)

# 3. BLT (Buffered Linear Toeplitz) - State-of-the-art for long training
noise_fn, noise_state = blt_mf_noise(
    grad_template,
    n_steps=10000,    # Total training steps (required)
    stddev=1.1,
    min_buffers=1,    # Optimize within this range
    max_buffers=5,
    key=key(42),
)

# 4. Dense MF - Best utility for small n_steps (< 100 steps)
noise_fn, noise_state = dense_mf_noise(
    grad_template,
    n_steps=100,      # Total training steps (required)
    stddev=1.1,
    key=key(42),
)

# 5. Custom MF - Bring your own strategy matrix C_inv
import torch
strategy_matrix = torch.eye(1000)  # Your custom C^{-1} matrix
noise_fn, noise_state = custom_mf_noise(
    grad_template,
    noising=strategy_matrix,
    stddev=1.1,
    key=key(42),
)

# 6. Identity MF - DP-SGD via MF API (for testing/validation)
from opaque.noise.matrix_factorization import identity
noise_fn, noise_state = identity_mf_noise(
    grad_template,
    stddev=1.1,
    key=key(42),
)

# Training loop (IDENTICAL for all mechanisms!)
for batch in dataloader:
    # Compute clipped gradients
    grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
    
    # Add noise (mechanism-specific, but same API!)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # Update parameters
    for (name, p), g in zip(model.named_parameters(), noisy_grads.values()):
        p.grad = g.to(p.dtype)
    optimizer.step()
    optimizer.zero_grad()
    
    # Update params for next iteration
    params = {k: v.detach() for k, v in model.named_parameters()}
```

### Distributed Training (DDP)

**All mechanisms work identically in distributed mode:**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from opaque.distributed import sum_gradients
from opaque.noise import gaussian_noise, band_mf_noise, blt_mf_noise

# Initialize distributed
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Model setup
model = MyModel().to(device)
model = DDP(model, device_ids=[rank])
params = {k: v.detach() for k, v in model.named_parameters()}
grad_template = {k: torch.zeros_like(v) for k, v in params.items()}

# Clipping (same as single-device)
clipped_grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=1)

# Choose ONE noise mechanism:
# All use key=key(42) for reproducibility; synchronized="auto" (default)
# ensures identical noise across devices in distributed mode.
# ===============================================================================
from opaque.random import key

# 1. Gaussian Noise
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))

# 2. BandMF
noise_fn, noise_state = band_mf_noise(grad_template, n_steps=1000, bands=4, stddev=1.1, key=key(42))

# 3. BLT
noise_fn, noise_state = blt_mf_noise(grad_template, n_steps=10000, stddev=1.1, key=key(42))

# 4. Dense MF
noise_fn, noise_state = dense_mf_noise(grad_template, n_steps=100, stddev=1.1, key=key(42))

# 5. Custom MF
noise_fn, noise_state = custom_mf_noise(grad_template, noising=strategy_matrix, stddev=1.1, key=key(42))

# 6. Identity MF
noise_fn, noise_state = identity_mf_noise(grad_template, stddev=1.1, key=key(42))

# Training loop (IDENTICAL for all mechanisms AND identical to single-device!)
for batch in dataloader:
    batch = tuple(t.to(device) for t in batch)
    
    # 1. Compute clipped gradients (per-device data)
    grads, clip_state = clipped_grad_fn(params, batch, state=clip_state)
    
    # 2. Aggregate across devices
    grads = sum_gradients(grads)
    
    # 3. Add noise (same on all devices - automatically synchronized!)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # 4. Update parameters
    for (name, p), g in zip(model.named_parameters(), noisy_grads.values()):
        p.grad = g.to(p.dtype)
    optimizer.step()
    optimizer.zero_grad()
    params = {k: v.detach() for k, v in model.named_parameters()}

dist.destroy_process_group()
```

**Key Takeaways:**

✅ **Same training loop** for ALL mechanisms (Gaussian + all 5 MF variants)  
✅ **Drop-in replacement** - Change only noise initialization line  
✅ **Automatic distributed support** - `synchronized="auto"` ensures identical noise across devices  
✅ **Identical API** - `noise_fn(grads, state) -> (noisy_grads, new_state)`  

**When to use which:**

| Mechanism | Memory | Utility | Best For |
|-----------|--------|---------|----------|
| **Gaussian** | O(1) | Baseline | Quick experiments, benchmarking |
| **BandMF** | O(bands) | +10-50% | General purpose (bands=4 default) |
| **BLT** | O(buffers) | Best | Long training (n > 5000) |
| **Dense** | O(n²) | Optimal | Small n (< 100 steps) |
| **Custom** | O(matrix) | Varies | Research, custom strategies |
| **Identity** | O(1) | Baseline | Testing MF infrastructure |

## BandMF: Banded Toeplitz Strategies

BandMF uses banded lower-triangular Toeplitz matrices. The `bands` parameter controls the trade-off between memory and utility:

```python
from opaque.noise import band_mf_noise

noise_fn, state = band_mf_noise(
    grad_template,
    n_steps=1000,
    bands=4,
    stddev=noise_multiplier * clip_norm,
    key=key(42),
)
```

- `bands=1`: Equivalent to DP-SGD (no correlation)
- `bands=4`: Good default, significant improvement over DP-SGD
- `bands=n`: Optimal but uses O(n) memory

## BLT: Buffered Linear Toeplitz

BLT matrices provide state-of-the-art utility with O(num_buffers) memory:

```python
from opaque.noise import blt_mf_noise

noise_fn, state = blt_mf_noise(
    grad_template,
    n_steps=10000,
    stddev=noise_multiplier * clip_norm,
    min_buffers=1,
    max_buffers=5,
    key=key(42),
)
```

## Dense Factorization

For small n, you can use a dense (optimal) factorization:

```python
from opaque.noise import dense_mf_noise

noise_fn, state = dense_mf_noise(
    grad_template,
    n_steps=100,
    stddev=noise_multiplier * clip_norm,
    key=key(42),
)
```

## Identity (DP-SGD Equivalent)

For comparison, `identity_mf_noise` uses the identity matrix (equivalent to standard DP-SGD):

```python
from opaque.noise import identity_mf_noise

noise_fn, state = identity_mf_noise(
    grad_template,
    stddev=noise_multiplier * clip_norm,
)
```

## Custom Matrix

Bring your own noising matrix with `custom_mf_noise`:

```python
from opaque.noise import custom_mf_noise
from opaque.noise.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)

coefs = optimal_max_error_strategy_coefs(1000)
noising = inverse_as_streaming_matrix(coefs)

noise_fn, state = custom_mf_noise(
    grad_template,
    noising,
    stddev=noise_multiplier * clip_norm,
    key=key(42),
)
```

## Multi-Participation (Multi-Epoch)

When training for multiple epochs, each example participates multiple times. The sensitivity computation must account for this:

```python
from opaque.noise import blt_mf_noise

# min_sep = epoch_length / batch_size (minimum steps between participations)
noise_fn, state = blt_mf_noise(
    grad_template,
    n_steps=5000,
    stddev=noise_multiplier * clip_norm,
    min_sep=100,
    max_participations=5,  # 5 epochs
)
```

## Sensitivity Computation

The sensitivity of the strategy matrix C determines the noise calibration:

```python
from opaque.noise.matrix_factorization.sensitivity import (
    single_participation_sensitivity,
    get_sensitivity_banded,
)

# Single-participation sensitivity
sens = single_participation_sensitivity(C_matrix)

# For banded matrices with min-sep
sens = get_sensitivity_banded(C_matrix, min_sep=100, max_participations=5)
```

## Comparison: DP-SGD vs BandMF vs BLT

For a linear regression with n=1000 steps, epsilon=1.0:

| Method | Mechanism | Final MSE | Memory |
|--------|-----------|-----------|--------|
| DP-SGD | Independent noise | ~2.5 | O(1) |
| BandMF (bands=4) | Banded Toeplitz | ~1.8 | O(4) |
| BLT (3 buffers) | Buffered Toeplitz | ~1.5 | O(3) |

*Values are illustrative; actual results depend on problem specifics.*

## Distributed Training (DDP)

Matrix factorization noise generation works seamlessly with distributed training using the **same pattern as Gaussian noise**:

```python
import torch.distributed as dist
from opaque.noise import band_mf_noise
from opaque.random import key

# Initialize distributed
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Create noise function - works exactly like gaussian_noise!
# key=key(42) with synchronized="auto" (default) ensures identical noise on all devices
noise_fn, noise_state = band_mf_noise(
    grad_template, 
    n_steps=1000, 
    bands=4, 
    stddev=1.1,
    key=key(42),
)

# Training loop - identical to single-device
for batch in dataloader:
    # 1. Compute clipped gradients (per device)
    clipped_grad = compute_clipped_grad(model, batch)
    
    # 2. Aggregate across devices
    clipped_grad = sum_gradients(clipped_grad)
    
    # 3. Add correlated noise (same API as gaussian_noise)
    noisy_grad, noise_state = noise_fn(clipped_grad, noise_state)
    
    # 4. Update parameters
    optimizer.step()
```

**Key points:**

- ✅ **Same API as `gaussian_noise()`** - No special distributed handling needed
- ✅ **Automatic synchronization** - `synchronized="auto"` broadcasts key to all devices
- ✅ **Centralized pattern** - All devices use same key (prevents model divergence)
- ✅ **Standard privacy accounting** - Use simple RDP/PLD accounting (no composition needed)

For details on distributed training, see [Distributed Training Guide](distributed.md).

## References

- **BandMF**: [Choquette-Choo et al., 2023](https://arxiv.org/abs/2306.08153)
- **BLT**: [McMahan et al., 2024](https://arxiv.org/abs/2404.16706)
- **Multi-epoch BLT**: [Choquette-Choo et al., 2024](https://arxiv.org/abs/2408.08868)
- **Inversion theorem**: [McMahan et al., 2025](https://arxiv.org/abs/2504.21413)
- **DP-FTRL**: [Kairouz et al., 2021](https://arxiv.org/abs/2103.00039)

## See Also

- **[Noise Addition](noise.md)**: Standard Gaussian noise
- **[Gradient Clipping](clipping.md)**: Per-example clipping
- **[Sampling](sampling.md)**: Cyclic Poisson sampling for BandMF
