# Poisson Sampling & Microbatching

**Poisson sampling** provides privacy amplification by randomly selecting training examples, while **microbatching**
enables memory-efficient DP training for large models.

## Why Poisson Sampling?

In DP-SGD, we don't train on the entire dataset at once—we sample batches. **How we sample matters for privacy!**

**Key insight**: If each example is selected independently with small probability, privacy is amplified (you get
stronger guarantees for the same noise).

### Privacy Amplification

Training on a **subset** of data provides better privacy than training on **all** data:

```python
import opaque.accounting as acc

# No sampling (full dataset)
training = acc.gaussian(noise_multiplier=1.0)
epsilon = training.epsilon_at(1e-5)  # ε ≈ 15.0

# With Poisson sampling (sample_rate=0.01)
training = acc.poisson(acc.gaussian(1.0), sample_rate=0.01)
epsilon = training.epsilon_at(1e-5)  # ε ≈ 0.1  (150x better!)
```

**Why?** Attackers can't be sure if a specific person's data was in the sampled batch.

## Poisson Sampling Basics

### What is Poisson Sampling?

Each example in the dataset is **independently** included in the batch with probability `sample_rate`:

```python
from opaque.sampling import PoissonSampler

sampler = PoissonSampler(sample_rate=0.01)

# Each call returns random subset
batch_indices = sampler.sample(dataset_size=10000)
# batch_indices ≈ 100 examples (but varies!)
```

**Key property**: Batch sizes are **variable** (Poisson distributed around `sample_rate × dataset_size`).

### Sample Rate

The **sample rate** is the probability each example is included:

```python
sample_rate = batch_size / dataset_size

# Example: batch_size=32, dataset_size=10000
sample_rate = 32 / 10000 = 0.0032  # 0.32% chance per example
```

**Privacy rule**: Lower sample rate → stronger privacy amplification → need less noise

### Standard Poisson Sampling

```python
from opaque.sampling import PoissonSampler

sampler = PoissonSampler(sample_rate=0.01)

noise_fn, noise_state = gaussian_noise(stddev=noise_mult * clip_norm)

for step in range(num_steps):
    # Sample batch (variable size!)
    indices = sampler.sample(dataset_size=10000)
    batch = dataset[indices]

    # Compute gradients
    grads, clip_state = dp_grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = update(params, noisy_grads)

# Track privacy (compose all steps)
training = acc.poisson(acc.gaussian(noise_mult), sample_rate=0.01) * num_steps
epsilon = training.epsilon_at(delta=1e-5)
```

## Truncated Poisson Sampling ⭐

**Problem**: Standard Poisson sampling has variable batch sizes (can be 0 or very large!)

**Solution**: **Truncated Poisson** bounds batch sizes while maintaining privacy amplification.

### Why Truncated?

Variable batch sizes are problematic:

- Batch size 0 → No gradient update
- Very large batch → Memory issues
- Inconsistent training dynamics

**Truncated Poisson solves this** by rejecting samples outside `[min_size, max_size]` range.

### Using Truncated Poisson

```python
from opaque.sampling import TruncatedPoissonSampler

sampler = TruncatedPoissonSampler(
    sample_rate=0.01,
    truncated_batch_size=32,  # Target batch size
    dataset_size=10000,
)

for step in range(num_steps):
    # Sample batch (bounded size!)
    indices = sampler.sample()
    # len(indices) ≈ 32, always in reasonable range

    batch = dataset[indices]
    grads, clip_state = dp_grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = update(params, noisy_grads)

# Track privacy (compose all steps)
training = acc.truncated_poisson(
    acc.gaussian(noise_mult),
    sample_rate=0.01,
    batch_size_cap=32,
    dataset_size=10000,
) * num_steps
epsilon = training.epsilon_at(delta=1e-5)
```

### Advantages

✅ **Consistent batch sizes**: No empty or huge batches
✅ **Tighter privacy bounds**: Up to 20% better than standard Poisson
✅ **Better training dynamics**: More stable convergence

!!! tip "Use truncated Poisson by default"
Unless you have a specific reason not to, use truncated Poisson for best results.

## Microbatching for Memory Efficiency

**Problem**: Per-example gradients require more memory than batch gradients.

**Solution**: Process the batch in smaller **microbatches**, accumulating gradients.

### Why Microbatch?

Computing per-example gradients is memory-intensive:

```python
# Standard batching (low memory)
batch_grad = compute_batch_average_gradient(batch)  # Memory: O(model_size)

# Per-example gradients (high memory!)
per_example_grads = [compute_gradient(ex) for ex in batch]  # Memory: O(batch_size × model_size)
```

For large models or large batches, this can cause **out-of-memory** errors.

### Manual Microbatching

Process the batch in chunks:

```python
def compute_clipped_gradients_microbatched(params, batch, microbatch_size=8):
    """Compute clipped gradients using microbatching."""
    total_grads = None

    for i in range(0, len(batch), microbatch_size):
        microbatch = batch[i : i + microbatch_size]

        # Compute clipped gradients for microbatch
        grads = dp_grad_fn(params, microbatch)

        # Accumulate
        if total_grads is None:
            total_grads = grads
        else:
            total_grads = {k: total_grads[k] + grads[k] for k in grads}

    return total_grads

# Training loop
for step in range(num_steps):
    batch = sample_batch(dataset, batch_size=32)

    # Use microbatching (memory-efficient!)
    grads = compute_clipped_gradients_microbatched(params, batch, microbatch_size=8)

    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = update(params, noisy_grads)
```

### Microbatch Size Selection

Choose `microbatch_size` based on GPU memory:

```python
# Rule of thumb: Start small and increase until OOM
microbatch_sizes = [4, 8, 16, 32]

for mbs in microbatch_sizes:
    try:
        grads = compute_clipped_gradients_microbatched(params, batch, microbatch_size=mbs)
        print(f"microbatch_size={mbs} works!")
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"microbatch_size={mbs} OOM, trying smaller...")
            torch.cuda.empty_cache()
        else:
            raise
```

### Privacy Guarantees with Microbatching

!!! success "Microbatching doesn't affect privacy!"
Microbatching is just a memory optimization. Privacy guarantees are unchanged.

```python
# These are equivalent for privacy:

# (1) Full batch
grads, clip_state = dp_grad_fn(params, batch_of_32, state=clip_state)

# (2) Microbatched
grads_mb1, clip_state = dp_grad_fn(params, batch[:16], state=clip_state)
grads_mb2, clip_state = dp_grad_fn(params, batch[16:], state=clip_state)
grads = {k: grads_mb1[k] + grads_mb2[k] for k in grads_mb1}

# Same clipped sum of per-example gradients!
```

## Complete Example: Truncated Poisson + Microbatching

Here's a full training loop combining both techniques:

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise
from opaque.sampling import TruncatedPoissonSampler

# Setup
clip_norm = 1.0
batch_size = 32
dataset_size = 10000
sample_rate = batch_size / dataset_size
microbatch_size = 8  # Process 8 examples at a time
num_steps = 1000

# Create sampler
sampler = TruncatedPoissonSampler(
    sample_rate=sample_rate,
    truncated_batch_size=batch_size,
    dataset_size=dataset_size,
)

# Calibrate noise
build = lambda nm: acc.truncated_poisson(
    acc.gaussian(nm), sample_rate, batch_size_cap=batch_size, dataset_size=dataset_size
) * num_steps

result = acc.calibrate(acc.epsilon_budget(3.0, delta=1e-5), build, 0.1, 10.0)
noise_multiplier = result.param

# Create DP gradient function
dp_grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=clip_norm, batch_argnums=1)

# Create noise function
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * clip_norm)

for step in range(num_steps):
    # Sample batch with truncated Poisson
    indices = sampler.sample()
    batch = dataset[indices]

    # Compute gradients with microbatching
    total_grads = None
    for i in range(0, len(batch), microbatch_size):
        microbatch = batch[i : i + microbatch_size]
        grads, clip_state = dp_grad_fn(params, microbatch, state=clip_state)

        if total_grads is None:
            total_grads = grads
        else:
            total_grads = {k: total_grads[k] + grads[k] for k in grads}

    # Add noise
    noisy_grads, noise_state = noise_fn(total_grads, noise_state)

    # Update
    params = update(params, noisy_grads)

# Check final privacy
training = acc.truncated_poisson(
    acc.gaussian(noise_multiplier),
    sample_rate=sample_rate,
    batch_size_cap=batch_size,
    dataset_size=dataset_size,
) * num_steps
print(f"Final privacy: ε={training.epsilon_at(1e-5):.2f}")
```

## Sampling Methods Comparison

| Method                | Batch Size                  | Privacy Bound | When to Use                  |
|-----------------------|-----------------------------|---------------|------------------------------|
| **Full batch**        | Fixed (= dataset_size)      | Weak          | Never for DP                 |
| **Fixed batching**    | Fixed                       | Moderate      | Simple, no amplification     |
| **Poisson**           | Variable (~sample_rate × n) | Strong        | Research, flexible           |
| **Truncated Poisson** | Bounded                     | **Strongest** | **Production (recommended)** |

## Privacy Amplification Effect

Smaller sample rates provide stronger amplification:

```python
# Large batches (sample_rate=0.1)
training_large = acc.poisson(acc.gaussian(1.0), sample_rate=0.1) * 100
eps_large = training_large.epsilon_at(1e-5)  # ε ≈ 1.5

# Small batches (sample_rate=0.01)
training_small = acc.poisson(acc.gaussian(1.0), sample_rate=0.01) * 100
eps_small = training_small.epsilon_at(1e-5)  # ε ≈ 0.15

print(f"Large batches: ε={eps_large:.2f}")
print(f"Small batches: ε={eps_small:.2f}")  # 10x better!
```

**Tradeoff**: Smaller batches → stronger privacy BUT more training steps needed for same number of epochs.

## Best Practices

### 1. Use Truncated Poisson by Default

```python
from opaque.sampling import TruncatedPoissonSampler

sampler = TruncatedPoissonSampler(sample_rate, batch_size, dataset_size)
```

### 2. Choose Appropriate Sample Rate

```python
# Good: sample_rate ∈ [0.001, 0.05]
sample_rate = 32 / 10000  # 0.0032 ✓

# Too high: sample_rate > 0.1
sample_rate = 1000 / 10000  # 0.1 (weak amplification) ⚠️

# Too low: sample_rate < 0.0001
sample_rate = 1 / 10000  # 0.0001 (too many steps) ⚠️
```

### 3. Use Microbatching for Large Models

```python
# For large models (e.g., LLMs), always use microbatching
microbatch_size = 4  # Start small
```

### 4. Monitor Batch Sizes

```python
batch_sizes = []
for step in range(num_steps):
    indices = sampler.sample()
    batch_sizes.append(len(indices))

print(f"Mean batch size: {np.mean(batch_sizes):.1f}")
print(f"Batch size range: [{min(batch_sizes)}, {max(batch_sizes)}]")
```

## See Also

- **[Tutorial 05](../tutorials/05_sampling_and_microbatching.ipynb)**: Interactive sampling tutorial
- **[Privacy Accounting](accounting.md)**: How sampling affects privacy
- **[API Reference](../api/sampling.md)**: Detailed sampling API
- **[LoRA Fine-tuning](lora.md)**: Use sampling with large models

---

**Next**: Learn about [LoRA Fine-tuning](lora.md) for parameter-efficient DP training
