# RngKey Quick Reference

One-page reference for Opaque's functional random number generation.

---

## Essential Imports

```python
# Core primitives
from opaque.random import RngKey, split, fold_in, key

# Convenience helpers
from opaque.random import random_key, training_key
```

---

## Creating Keys

```python
# From integer seed (reproducible)
key = RngKey(42)

# Shorter alias
key = key(42)

# Non-deterministic (for prototyping)
key = random_key()  # Uses system entropy
```

**Note**: Use `random_key()` only for quick experiments. For production, use `RngKey(fixed_seed)` or `training_key()`.

---

## Splitting Keys (Most Important Pattern)

```python
# Binary split (most common)
key1, key2 = split(key, num=2)

# Multi-way split
keys = split(key, num=10)  # Returns list

# Loop pattern - split and advance
for step in range(100):
    key, step_key = split(key, num=2)
    result = do_something_random(step_key)
```

**Golden Rule**: Never reuse keys. Always split to create independent randomness.

---

## Training Loop Pattern

**Setup-once pattern** (keys created once, state manages per-step derivation):

```python
# Setup
key = RngKey(42)
key_sampling, key_noise = split(key, num=2)

sampler = PoissonSampler(..., key=key_sampling)
noise_fn, noise_state = gaussian_noise(..., key=key_noise)

# Loop
for batch in dataloader:
    # Compute and clip gradients
    grads = compute_grads(model, batch)
    
    # Add noise (state manages per-step keys internally)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # Update
    optimizer.step(noisy_grads)
```

**Per-step pattern** (using `training_key()` helper):

```python
from opaque.random import training_key

# Training loop
for step in range(num_steps):
    # Derive step-specific key
    step_key = training_key(base_seed=42, step=step)
    
    # Use for this step's operations
    noise_fn, state = gaussian_noise(1.0, 1.1, key=step_key)
    grads = compute_grads(model, batch)
    noisy_grads, state = noise_fn(grads, state)
    optimizer.step(noisy_grads)
```

---

## Distributed Training Pattern

**Using `training_key()` helper** (recommended):

```python
import torch.distributed as dist
from opaque.random import training_key

rank = dist.get_rank()

# Training loop
for step in range(num_steps):
    # Sampling: synchronized=True (same key on all ranks)
    sampling_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=True,  # Coordinated
    )
    sampler = PoissonSampler(..., key=sampling_key)
    
    # Noise: synchronized=False (different per rank)
    noise_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=False,  # Independent
    )
    noise_fn, state = gaussian_noise(..., key=noise_key)
```

**Manual splitting approach** (for reference):

```python
from opaque.random import RngKey, split

world_size = dist.get_world_size()
rank = dist.get_rank()

# Split global key per rank
global_key = RngKey(42)

# Sampling: SAME key on all ranks (coordinated)
sampling_key, noise_master = split(global_key, num=2)
sampler = PoissonSampler(..., key=sampling_key)  # Shared

# Noise: DIFFERENT keys per rank (independent)
rank_keys = split(noise_master, num=world_size)
my_noise_key = rank_keys[rank]  # Unique per rank
noise_fn, state = gaussian_noise(..., key=my_noise_key)
```

**Key insight**: `training_key()` enforces the correct `step → rank` derivation order and makes synchronized/unsynchronized explicit.

---

## Per-Step Keys with `fold_in()`

```python
# Base key + step counter → deterministic per-step key
base_key = RngKey(42)

for step in range(1000):
    step_key = fold_in(base_key, step)
    train_step(model, batch, key=step_key)
    
# Restart from checkpoint
resume_key = fold_in(base_key, checkpoint_step)
```

**Use case**: Checkpointing, debugging specific steps, adaptive clipping.

---

## Common Code Patterns

### Component Splitting
```python
# Split once for different components
master_key = RngKey(42)
data_key, model_key, noise_key = split(master_key, num=3)

sampler = setup_sampler(data_key)
model = initialize_model(model_key)
noise_fn = setup_noise(noise_key)
```

### Sequential Splitting
```python
# Training loop with advancing key
key = RngKey(42)
for epoch in range(num_epochs):
    key, epoch_key = split(key, num=2)
    
    for step in range(steps_per_epoch):
        epoch_key, step_key = split(epoch_key, num=2)
        train_step(model, batch, key=step_key)
```

### Hierarchical Splitting
```python
# Train/eval get independent subtrees
master_key = RngKey(42)
train_key, eval_key = split(master_key, num=2)

# Training subtree
train_sample, train_noise = split(train_key, num=2)

# Eval subtree (independent from training)
eval_sample, eval_noise = split(eval_key, num=2)
```

---

## Using with Opaque APIs

### Noise
```python
from opaque.noise import gaussian_noise

noise_fn, state = gaussian_noise(
    stddev=1.1,
    key=RngKey(42),  # ← Explicit key
)
```

### Sampling
```python
from opaque.sampling import PoissonSampler

sampler = PoissonSampler(
    dataset_size=10000,
    sample_rate=0.01,
    key=RngKey(42),  # ← Explicit key
)
```

### Adaptive Clipping with Quantile Noise
```python
from opaque.clipping import adaptive_clipped_grad

grad_fn, state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,
    quantile_noise_std=0.1,  # Requires key
    key=RngKey(42),  # ← For quantile noise
    batch_argnums=(1, 2),
)
```

### Auditing
```python
from opaque.auditing import setup, evaluate

experiment = setup(
    model_fn=...,
    key=RngKey(42),  # ← For audit experiments
)
result = evaluate(experiment)
```

---

## Reproducibility Checklist

✅ **Do**:
- Use RngKey for all Opaque randomness
- Split keys to create independent streams
- Thread keys explicitly through functions
- Document which key is for what purpose

❌ **Don't**:
- Mix RngKey with global seeds (`torch.manual_seed`)
- Reuse the same key for different operations
- Create random keys inside loops
- Share noise keys across distributed ranks

---

## Troubleshooting

### Results not reproducible?

```python
# ✅ For Opaque operations
key = RngKey(42)
noise_fn = gaussian_noise(..., key=key)

# ✅ For PyTorch operations (if needed)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
```

### Distributed ranks have correlated noise?

```python
# ❌ Bad: All ranks same key
key = RngKey(42)
noise_fn = gaussian_noise(key=key)  # Violates DP!

# ✅ Good: Split per rank
rank_keys = split(RngKey(42), num=world_size)
my_key = rank_keys[rank]
noise_fn = gaussian_noise(key=my_key)
```

### Need to restart from checkpoint?

```python
# Use fold_in() for deterministic per-step keys
base_key = RngKey(42)
resume_key = fold_in(base_key, checkpoint_step)

for step in range(checkpoint_step, total_steps):
    step_key = fold_in(base_key, step)
    train_step(model, batch, key=step_key)
```

---

## API Summary

| Function | Purpose | Example |
|----------|---------|---------|
| `RngKey(seed)` | Create key | `key = RngKey(42)` |
| `key(seed)` | Alias for `RngKey(seed)` | `k = key(42)` |
| `random_key()` | Non-deterministic key | `k = random_key()` |
| `split(key, num)` | Split into subkeys | `k1, k2 = split(key, num=2)` |
| `fold_in(key, data)` | Derive key from counter | `step_key = fold_in(key, step)` |
| `training_key(...)` | Training loop key with proper derivation | `k = training_key(42, step=0, rank=0, synchronized=False)` |

---

## Complete Example

**Using `training_key()` helper** (recommended):

```python
from opaque.random import training_key
from opaque import clipped_grad, gaussian_noise
from opaque.sampling import PoissonSampler
import torch.distributed as dist

# Setup
grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=1)
rank = dist.get_rank() if dist.is_initialized() else None

# Training loop
for step in range(1000):
    # Coordinated sampling
    sampling_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=True if rank is not None else None,
    )
    sampler = PoissonSampler(10000, sample_rate=0.01, key=sampling_key)
    
    # Independent noise
    noise_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=False if rank is not None else None,
    )
    noise_fn, noise_state = gaussian_noise(stddev=1.1, key=noise_key)
    
    # Train
    for batch in sampler:
        grads, clip_state = grad_fn(params, batch, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        optimizer.step(noisy_grads)
```

**Manual splitting approach** (for comparison):

```python
from opaque.random import RngKey, split

# Master seed
master_key = RngKey(42)
sampling_key, noise_master = split(master_key, num=2)

# Distributed: per-rank noise keys
if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank_keys = split(noise_master, num=world_size)
    noise_key = rank_keys[rank]
else:
    noise_key = noise_master

# Setup and training
sampler = PoissonSampler(10000, sample_rate=0.01, key=sampling_key)
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=noise_key)

for batch in sampler:
    grads = compute_grads(model, batch)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    optimizer.step(noisy_grads)
```

---

## Further Reading

- [RngKey User Guide](rng-key.md) - Comprehensive documentation
- [Noise APIs](noise.md) - Using keys with noise
- [Sampling](sampling.md) - Using keys with samplers
- [Distributed Training](distributed.md) - Keys in DDP/FSDP
