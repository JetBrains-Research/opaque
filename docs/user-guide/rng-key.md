# Random Number Generation with RngKey

## Overview

Opaque uses **explicit random keys** (`RngKey`) instead of global random seeds for all randomness. This functional approach, inspired by JAX's PRNG system, provides stronger guarantees for reproducibility, determinism, and correctness in distributed training.

**Why RngKey matters**:
- ✅ **Explicit control**: Every random operation requires an explicit key
- ✅ **No hidden state**: Reproducibility doesn't depend on global seed state
- ✅ **Distributed-safe**: Each device gets independent, non-overlapping randomness
- ✅ **Composable**: Split keys to create independent random streams
- ✅ **Debuggable**: Track exactly where randomness comes from

**For PyTorch users**: RngKey replaces patterns like `torch.manual_seed(42)` with explicit key passing. This may seem verbose at first, but provides much stronger guarantees for differential privacy and distributed training.

---

## Basic Concepts

### What is RngKey?

`RngKey` is a lightweight wrapper around a random seed that provides **deterministic, splittable** pseudo-random number generation:

```python
from opaque.random import RngKey

# Create a key from a seed
key = RngKey(42)

# Key is immutable and hashable
print(key)  # RngKey(seed=42, impl='opaque_threefry_like')
```

**Key properties**:
- **Immutable**: Once created, keys never change
- **Deterministic**: Same key → same random numbers
- **Splittable**: Create independent keys from a parent key
- **Lightweight**: Just an integer seed + implementation tag

### Why Not Global Seeds?

Traditional PyTorch code uses global seeds:

```python
# ❌ Traditional approach (hidden state, hard to reproduce)
torch.manual_seed(42)
noise1 = torch.randn(100)  # Where did 42 come from?
torch.manual_seed(42)
noise2 = torch.randn(100)  # Same as noise1, but fragile
```

**Problems**:
- **Hidden state**: Global RNG state is invisible and mutable
- **Order-dependent**: Random values depend on execution order
- **Not composable**: Can't create independent random streams easily
- **Distributed issues**: Each device needs different seeds, easy to overlap
- **Hard to debug**: "Why aren't my results reproducible?"

With RngKey, randomness is **explicit and functional**:

```python
# ✅ Functional approach (explicit, reproducible)
key = RngKey(42)
noise1 = some_random_function(key)  # Explicit key input
noise2 = some_random_function(key)  # Same key → same output
```

---

## Getting Started

### Creating Keys

```python
from opaque.random import RngKey

# From an integer seed (reproducible)
key = RngKey(42)

# From a random seed (different each time - for prototyping)
import secrets
key = RngKey(secrets.randbits(64))
```

**For quick prototyping**, use the `random_key()` helper instead of manually generating random seeds:

```python
from opaque.random import random_key

# Non-deterministic key for prototyping (uses system entropy)
key = random_key()

# Use immediately
noise_fn, state = gaussian_noise(l2_norm_bound=1.0, noise_multiplier=1.1, key=random_key())
```

**Important**: `random_key()` is convenient for experiments but makes results non-reproducible. For production training, always use `RngKey(fixed_seed)` or the `training_key()` helper (introduced below).

### Using Keys with Opaque APIs

All Opaque APIs that involve randomness accept a `key` parameter:

```python
from opaque.random import RngKey
from opaque.noise import gaussian_noise
import torch

# Create key
key = RngKey(42)

# Use key with noise API
noise_fn, state = gaussian_noise(
    l2_norm_bound=1.0,
    noise_multiplier=1.1,
    key=key,  # ← Explicit key
)

# Generate noise (deterministic with same key)
grads = {"w": torch.randn(100), "b": torch.randn(10)}
noisy_grads = noise_fn(grads, state)
```

**Key point**: The same `key` passed to `gaussian_noise()` will always produce the same noise sequence, making your DP training fully reproducible.

---

## Key Splitting: Creating Independent Randomness

The fundamental pattern in functional RNG is **key splitting**: creating multiple independent keys from a single parent key.

### Why Split Keys?

You need independent randomness for different purposes:
- Different training steps
- Different model components (sampling vs. noise)
- Different processes in distributed training

**Traditional approach** (fragile):
```python
# ❌ Hard to maintain independence
torch.manual_seed(42)
noise1 = torch.randn(100)
torch.manual_seed(43)  # Must manually track seeds
noise2 = torch.randn(100)
torch.manual_seed(44)
noise3 = torch.randn(100)
```

**Functional approach** (robust):
```python
from opaque.random import split

# ✅ Split key into independent subkeys
key = RngKey(42)
key1, key2, key3 = split(key, num=3)

# Each subkey is independent
noise1 = some_random_function(key1)
noise2 = some_random_function(key2)  # Independent from noise1
noise3 = some_random_function(key3)  # Independent from noise1 and noise2
```

### Basic Splitting

```python
from opaque.random import RngKey, split

key = RngKey(42)

# Split into 2 keys (most common)
key1, key2 = split(key, num=2)

# Split into N keys
keys = split(key, num=10)  # Returns list of 10 keys

# Common pattern: split and use one, keep other
key, subkey = split(key, num=2)
result = some_random_function(subkey)
# 'key' is now updated for next split
```

### Training Loop Pattern

**Manual approach** - explicit key splitting:

```python
from opaque.random import RngKey, split, fold_in
from opaque.noise import gaussian_noise

# Initialize
base_key = RngKey(42)
noise_fn, noise_state = gaussian_noise(l2_norm_bound=1.0, noise_multiplier=1.1)

# Training loop - manual key derivation
for step in range(num_steps):
    # Derive step-specific key
    step_key = fold_in(base_key, step)
    
    # Update noise state with new key
    noise_state = noise_state._replace(key=step_key)
    
    # Compute noisy gradients
    grads = compute_gradients(model, batch)
    noisy_grads = noise_fn(grads, noise_state)
    
    # Update model
    optimizer.step(noisy_grads)
```

**Ergonomic approach** - using `training_key()` helper:

```python
from opaque.random import training_key
from opaque.noise import gaussian_noise

# Initialize
noise_fn, noise_state = gaussian_noise(l2_norm_bound=1.0, noise_multiplier=1.1)

# Training loop - ergonomic helper
for step in range(num_steps):
    # One-line key derivation with proper ordering
    step_key = training_key(base_seed=42, step=step)
    
    # Update noise state
    noise_state = noise_state._replace(key=step_key)
    
    # Train
    grads = compute_gradients(model, batch)
    noisy_grads = noise_fn(grads, noise_state)
    optimizer.step(noisy_grads)
```

**Key insight**: Both approaches ensure:
- Each step gets independent noise
- The sequence is fully reproducible (same base seed → same noise sequence)
- No hidden global state to corrupt results

The `training_key()` helper is more convenient and follows the canonical derivation chain: `step → rank → worker_id`.

---

## Determinism & Reproducibility

### Reproducibility Guarantees

With RngKey, **same key = same random numbers**, period.

```python
from opaque.random import RngKey
from opaque.noise import gaussian_noise
import torch

# Run 1
key1 = RngKey(42)
noise_fn, state1 = gaussian_noise(l2_norm_bound=1.0, noise_multiplier=1.1, key=key1)
grads = {"w": torch.randn(100)}
noisy1 = noise_fn(grads, state1)

# Run 2 (identical)
key2 = RngKey(42)
noise_fn, state2 = gaussian_noise(l2_norm_bound=1.0, noise_multiplier=1.1, key=key2)
grads = {"w": torch.randn(100)}
noisy2 = noise_fn(grads, state2)

# Guaranteed equal
assert torch.allclose(noisy1["w"], noisy2["w"])
```

### Multi-Step Reproducibility

```python
def train_model(key: RngKey, num_steps: int):
    """Fully reproducible training function."""
    from opaque.random import split
    
    # Initialize
    key, init_key = split(key, num=2)
    model = initialize_model(init_key)
    
    # Training loop
    for step in range(num_steps):
        key, step_key = split(key, num=2)
        
        # All randomness flows from step_key
        key_sampling, key_noise = split(step_key, num=2)
        batch = sample_batch(dataset, key=key_sampling)
        grads = compute_grads(model, batch)
        noisy_grads = add_noise(grads, key=key_noise)
        
        model = update_model(model, noisy_grads)
    
    return model

# Same seed → same final model
model1 = train_model(RngKey(42), num_steps=1000)
model2 = train_model(RngKey(42), num_steps=1000)
# model1 and model2 are identical
```

### Cross-Platform Determinism

RngKey provides **byte-for-byte reproducibility** across:
- ✅ Different Python/PyTorch versions
- ✅ Different hardware (CPU, CUDA, MPS)
- ✅ Different operating systems
- ✅ Different node counts in distributed training

**Caveat**: PyTorch's own operations (like `torch.randn`) may have platform-specific implementations. Opaque's RngKey controls only Opaque's randomness (noise, sampling, auditing).

---

## Distributed Training

### The Distributed Problem

In distributed training (DDP/FSDP), each device needs **different but coordinated** randomness:

- **Same**: Poisson sampling decisions (which examples to include)
- **Different**: Noise added to gradients (independent per device)

**Traditional approach** (error-prone):
```python
# ❌ Manual seed offsets (easy to mess up)
rank = torch.distributed.get_rank()
torch.manual_seed(42 + rank)  # Overlap if misaligned
```

**Functional approach** (correct by construction):
```python
# ✅ Explicit rank-based splitting
from opaque.random import RngKey, split

world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()

# Split global key into per-rank keys
global_key = RngKey(42)
rank_keys = split(global_key, num=world_size)
my_key = rank_keys[rank]  # Each rank gets unique key
```

### Distributed Noise (Independent)

Each device needs **independent noise** to satisfy DP. There are two approaches:

**Manual approach** - explicit rank splitting:

```python
from opaque.random import RngKey, split, fold_in
from opaque.noise import gaussian_noise
import torch.distributed as dist

# Setup
world_size = dist.get_world_size()
rank = dist.get_rank()

# Method 1: Split global key into per-rank keys
global_key = RngKey(42)
noise_keys = split(global_key, num=world_size)
my_noise_key = noise_keys[rank]

# Method 2: Use fold_in for dynamic ranks
base_key = RngKey(42)
my_noise_key = fold_in(fold_in(base_key, step), rank)  # step → rank

# Create noise function with rank-specific key
noise_fn, state = gaussian_noise(
    l2_norm_bound=1.0,
    noise_multiplier=1.1,
    key=my_noise_key,  # Different per rank
)

# Each rank adds independent noise
grads = compute_local_gradients(model, batch)
noisy_grads = noise_fn(grads, state)  # Independent noise per rank
```

**Ergonomic approach** - using `training_key()` with `synchronized=False`:

```python
from opaque.random import training_key
from opaque.noise import gaussian_noise
import torch.distributed as dist

# Setup
rank = dist.get_rank()

# Training loop
for step in range(num_steps):
    # Automatic step → rank derivation
    step_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=False,  # Different noise per rank
    )
    
    noise_fn, state = gaussian_noise(
        l2_norm_bound=1.0,
        noise_multiplier=1.1,
        key=step_key,
    )
    
    grads = compute_local_gradients(model, batch)
    noisy_grads = noise_fn(grads, state)
```

### Distributed Sampling (Coordinated)

For Poisson sampling, all ranks need **the same** sampling decisions.

**Manual approach** - no rank splitting:

```python
from opaque.random import RngKey, fold_in
from opaque.sampling import PoissonSampler

# All ranks use the SAME key for sampling (no rank folding!)
base_key = RngKey(42)

# Training loop
for step in range(num_steps):
    # Fold in step but NOT rank (same key on all ranks)
    sampling_key = fold_in(base_key, step)
    
    sampler = PoissonSampler(
        dataset_size=10000,
        sample_rate=0.01,
        key=sampling_key,  # Same key on all ranks
    )
    
    for indices in sampler:
        # indices is the same on all ranks
        batch = load_batch(indices)  # Load from local shard
        ...
```

**Ergonomic approach** - using `training_key()` with `synchronized=True`:

```python
from opaque.random import training_key
from opaque.sampling import PoissonSampler
import torch.distributed as dist

rank = dist.get_rank()

# Training loop
for step in range(num_steps):
    # synchronized=True means ignore rank (same key on all ranks)
    sampling_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,  # Passed but ignored when synchronized=True
        synchronized=True,  # Coordinated sampling
    )
    
    sampler = PoissonSampler(
        dataset_size=10000,
        sample_rate=0.01,
        key=sampling_key,
    )
    
    for indices in sampler:
        batch = load_batch(indices)
        ...
```

### Complete Distributed Example

**Using `training_key()` helper** (recommended for clarity):

```python
from opaque.random import training_key
from opaque.noise import gaussian_noise
from opaque.sampling import PoissonSampler
import torch.distributed as dist

def distributed_training_loop(base_seed: int, num_steps: int):
    """Complete distributed training with proper key management."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    for step in range(num_steps):
        # Sampling: synchronized (same key on all ranks)
        sampling_key = training_key(
            base_seed=base_seed,
            step=step,
            rank=rank,
            synchronized=True,  # Coordinated sampling
        )
        
        sampler = PoissonSampler(
            dataset_size=10000,
            sample_rate=0.01,
            key=sampling_key,
        )
        
        # Noise: unsynchronized (different key per rank)
        noise_key = training_key(
            base_seed=base_seed,
            step=step,
            rank=rank,
            synchronized=False,  # Independent noise
        )
        
        noise_fn, noise_state = gaussian_noise(
            l2_norm_bound=1.0,
            noise_multiplier=1.1,
            key=noise_key,
        )
        
        # Training
        for indices in sampler:
            batch = dataset[indices]
            grads = compute_gradients(model, batch)
            noisy_grads = noise_fn(grads, noise_state)
            
            # Aggregate
            for param, grad in noisy_grads.items():
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad /= world_size
            
            optimizer.step(noisy_grads)

# Run training
distributed_training_loop(base_seed=42, num_steps=1000)
```

**Manual approach** (for comparison):

```python
from opaque.random import RngKey, split, fold_in

def distributed_training_loop_manual(base_seed: int, num_steps: int):
    """Manual key derivation - more verbose but explicit."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    base_key = RngKey(base_seed)
    
    for step in range(num_steps):
        # Derive step key first
        step_key = fold_in(base_key, step)
        
        # Sampling: no rank folding (coordinated)
        sampling_key = step_key
        sampler = PoissonSampler(10000, 0.01, key=sampling_key)
        
        # Noise: fold in rank (independent)
        noise_key = fold_in(step_key, rank)
        noise_fn, noise_state = gaussian_noise(1.0, 1.1, key=noise_key)
        
        # ... rest of training loop ...
```

**Key insight**: The `training_key()` helper enforces the correct derivation order (`step → rank → worker_id`) and makes the synchronized/unsynchronized distinction explicit. This prevents common mistakes like folding rank for sampling or forgetting to fold rank for noise.

---

## Advanced Patterns

### Using `fold_in()` for Step-Based Keys

When you need a key that depends on a step counter, use `fold_in()`:

```python
from opaque.random import RngKey, fold_in

base_key = RngKey(42)

# Generate step-specific keys
for step in range(100):
    step_key = fold_in(base_key, step)
    # step_key is deterministic function of (base_key, step)
    noise = generate_noise(step_key)
```

**Use cases**:
- Adaptive clipping quantile noise (needs per-step randomness)
- Checkpoint restart (reproducible from any step)
- Debugging specific training steps

**Example: Adaptive Clipping**
```python
from opaque.random import RngKey, fold_in
from opaque.clipping import adaptive_clipped_grad

# Setup
key = RngKey(42)
grad_fn, state = adaptive_clipped_grad(
    loss_fn,
    initial_clip_norm=1.0,
    quantile_noise_std=0.1,  # Requires key
    key=key,
    batch_argnums=(1, 2),
)

# Training loop
for step in range(100):
    # State automatically uses fold_in(key, step) internally
    grads, state = grad_fn(params, batch_x, batch_y, state=state)
    # Each step gets independent noise via state.step counter
```

### Array Keys for Batch Operations

For operations on arrays/tensors, you can create arrays of keys:

```python
from opaque.random import RngKey, split

# Create 100 independent keys for batch processing
base_key = RngKey(42)
batch_keys = split(base_key, num=100)

# Use with vmap for parallel operations
import torch
from torch.func import vmap

def per_example_noise(key: RngKey, grad: torch.Tensor) -> torch.Tensor:
    """Add independent noise to one example's gradient."""
    rng = torch.Generator().manual_seed(key.seed)
    return grad + torch.randn_like(grad, generator=rng) * 0.1

# Vectorized across batch
grads_batch = torch.randn(100, 50)  # [batch_size, feature_dim]
noisy_grads = vmap(per_example_noise)(batch_keys, grads_batch)
# Each example gets independent noise
```

### Key Derivation Patterns

Common patterns for organizing keys:

```python
from opaque.random import RngKey, split

# Pattern 1: Component-based splitting
master_key = RngKey(42)
data_key, model_key, noise_key, audit_key = split(master_key, num=4)

sampler = setup_sampler(data_key)
model = initialize_model(model_key)
noise_fn = setup_noise(noise_key)
auditor = setup_auditor(audit_key)

# Pattern 2: Hierarchical splitting
master_key = RngKey(42)
train_key, eval_key = split(master_key, num=2)

# Training gets its own subtree
train_sampling_key, train_noise_key = split(train_key, num=2)
train_sampler = setup_sampler(train_sampling_key)
train_noise = setup_noise(train_noise_key)

# Evaluation gets independent subtree
eval_sampling_key, eval_noise_key = split(eval_key, num=2)
eval_sampler = setup_sampler(eval_sampling_key)
eval_noise = setup_noise(eval_noise_key)

# Pattern 3: Sequential splitting (training loop)
key = RngKey(42)
for epoch in range(num_epochs):
    key, epoch_key = split(key, num=2)
    
    for step in range(steps_per_epoch):
        epoch_key, step_key = split(epoch_key, num=2)
        
        # Use step_key for this iteration
        batch = sample_batch(step_key)
        grads = compute_grads(batch)
```

---

## API Reference

### Core Types

#### `RngKey`

```python
class RngKey:
    """Immutable random number generator key.
    
    Args:
        seed: Integer seed (0 to 2^32 - 1)
        impl: PRNG implementation name (default: "opaque_threefry_like")
    
    Attributes:
        seed: The integer seed
        impl: PRNG implementation identifier
    """
    def __init__(self, seed: int, impl: str = "opaque_threefry_like") -> None: ...
    
    @property
    def seed(self) -> int: ...
    
    @property
    def impl(self) -> str: ...
    
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
```

**Usage**:
```python
key = RngKey(42)
print(key.seed)  # 42
print(key.impl)  # "opaque_threefry_like"
```

### Core Functions

#### `split()`

```python
def split(key: RngKey, num: int = 2) -> list[RngKey]:
    """Split a key into multiple independent subkeys.
    
    Args:
        key: Parent key to split
        num: Number of subkeys to generate (default: 2)
    
    Returns:
        List of `num` independent RngKey objects
    
    Example:
        >>> key = RngKey(42)
        >>> key1, key2, key3 = split(key, num=3)
        >>> # key1, key2, key3 are independent
    """
```

**Usage**:
```python
from opaque.random import RngKey, split

key = RngKey(42)

# Binary split (most common)
key1, key2 = split(key)

# Multi-way split
keys = split(key, num=10)  # List of 10 keys

# Loop pattern
for i in range(100):
    key, subkey = split(key)
    result = use_randomness(subkey)
```

#### `fold_in()`

```python
def fold_in(key: RngKey, data: int) -> RngKey:
    """Fold integer data into a key to derive a new key.
    
    Creates a new key that is a deterministic function of both the
    original key and the data. Useful for step-dependent keys.
    
    Args:
        key: Base key
        data: Integer to fold in (typically a step counter)
    
    Returns:
        New RngKey derived from key and data
    
    Example:
        >>> base_key = RngKey(42)
        >>> step_key = fold_in(base_key, step=5)
        >>> # step_key is deterministic function of (42, 5)
    """
```

**Usage**:
```python
from opaque.random import RngKey, fold_in

base_key = RngKey(42)

# Generate per-step keys
for step in range(1000):
    step_key = fold_in(base_key, step)
    noise = add_noise(grads, key=step_key)
    
# Restart from checkpoint
checkpoint_step = 500
resume_key = fold_in(base_key, checkpoint_step)
# Resume training with correct key for step 500
```

#### `key()`

```python
def key(seed: int) -> RngKey:
    """Convenience function to create an RngKey.
    
    Alias for RngKey(seed) with shorter name for cleaner code.
    
    Args:
        seed: Integer seed
    
    Returns:
        RngKey with given seed
    
    Example:
        >>> from opaque.random import key
        >>> k = key(42)  # Equivalent to RngKey(42)
    """
```

**Usage**:
```python
from opaque.random import key

# Shorter syntax
k = key(42)  # Instead of RngKey(42)

# Useful for inline key creation
noise_fn, state = gaussian_noise(
    l2_norm_bound=1.0,
    noise_multiplier=1.1,
    key=key(42),  # Clean inline syntax
)
```

#### `random_key()`

```python
def random_key() -> RngKey:
    """Create a non-deterministic key using system entropy.
    
    Useful for prototyping when reproducibility is not critical. For production
    training, prefer training_key() with an explicit base_seed.
    
    Returns:
        A randomly initialized RngKey.
    
    Example:
        >>> from opaque.random import random_key
        >>> from opaque.noise import gaussian_noise
        >>> k = random_key()
        >>> noise_fn, state = gaussian_noise(l2_norm_clip=1.0, noise_multiplier=1.1, key=k)
    """
```

**Usage**:
```python
from opaque.random import random_key

# Quick prototyping (non-reproducible)
key = random_key()
noise_fn, state = gaussian_noise(1.0, 1.1, key=random_key())

# ⚠️ Warning: Results won't be reproducible!
```

#### `training_key()`

```python
def training_key(
    base_seed: int,
    step: int,
    rank: int | None = None,
    worker_id: int | None = None,
    synchronized: bool | Literal["auto"] | None = None,
) -> RngKey:
    """Create a deterministic key for training loops with proper derivation order.
    
    Follows the canonical derivation chain: step → rank → worker_id.
    
    The synchronized parameter controls whether noise is identical across ranks:
    - True: Same key for all ranks (centralized DP-SGD with synchronized noise)
    - False: Different keys per rank via fold_in(rank)
    - "auto": Synchronized if rank is None, unsynchronized otherwise
    - None (default): Must not pass rank (raises ValueError)
    
    Args:
        base_seed: Reproducible seed for the entire training run.
        step: Training step counter (folded first).
        rank: Distributed rank (folded after step if unsynchronized).
        worker_id: DataLoader worker ID (folded last).
        synchronized: Noise synchronization policy for distributed training.
    
    Returns:
        Derived RngKey following step → rank → worker_id order.
    
    Raises:
        ValueError: If rank is passed without specifying synchronized.
        ValueError: If synchronized has an invalid value.
    
    Example:
        >>> from opaque.random import training_key
        >>> 
        >>> # Single-device training
        >>> for step in range(100):
        ...     k = training_key(base_seed=42, step=step)
        ...     # ... train ...
        >>> 
        >>> # Distributed with per-rank noise
        >>> k = training_key(base_seed=42, step=0, rank=local_rank, synchronized=False)
        >>> 
        >>> # Distributed with synchronized noise (for sampling)
        >>> k = training_key(base_seed=42, step=0, rank=local_rank, synchronized=True)
    """
```

**Usage**:
```python
from opaque.random import training_key
import torch.distributed as dist

# Single-device training loop
for step in range(1000):
    key = training_key(base_seed=42, step=step)
    train_step(model, batch, key=key)

# Distributed training - per-rank noise
rank = dist.get_rank()
for step in range(1000):
    noise_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=False,  # Different noise per rank
    )
    train_step_with_noise(model, batch, key=noise_key)

# Distributed training - synchronized sampling
for step in range(1000):
    sampling_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=True,  # Same sampling decisions
    )
    batch = sample_batch(dataset, key=sampling_key)

# With DataLoader workers
for step in range(1000):
    worker_id = torch.utils.data.get_worker_info().id
    key = training_key(
        base_seed=42,
        step=step,
        worker_id=worker_id,
    )
```

### Utility Functions

#### `generator_from_key()`

```python
def generator_from_key(key: RngKey) -> torch.Generator:
    """Convert RngKey to PyTorch Generator.
    
    Args:
        key: RngKey to convert
    
    Returns:
        torch.Generator seeded with key.seed
    
    Note:
        Used internally by Opaque. Typical users don't need this.
    """
```

**Usage** (advanced):
```python
from opaque.random import RngKey, generator_from_key
import torch

key = RngKey(42)
generator = generator_from_key(key)

# Use with PyTorch APIs
noise = torch.randn(100, generator=generator)
```

---

## Best Practices

### ✅ Do This

**1. Split keys liberally**
```python
# Create independent keys for different purposes
key = RngKey(42)
sampling_key, noise_key, audit_key = split(key, num=3)
```

**2. Never reuse keys for different purposes**
```python
# ✅ Good
key1, key2 = split(key, num=2)
noise1 = add_noise(grads1, key=key1)
noise2 = add_noise(grads2, key=key2)  # Independent noise

# ❌ Bad
noise1 = add_noise(grads1, key=key)
noise2 = add_noise(grads2, key=key)  # Correlated noise!
```

**3. Thread keys through your code explicitly**
```python
# ✅ Good: Explicit key threading
def train_step(model, batch, key: RngKey):
    key_noise, key_dropout = split(key, num=2)
    grads = compute_grads(model, batch, key=key_dropout)
    noisy_grads = add_noise(grads, key=key_noise)
    return update_model(model, noisy_grads)
```

**4. Use `fold_in()` for step-dependent randomness**
```python
# ✅ Good: Deterministic per-step keys
base_key = RngKey(42)
for step in range(1000):
    step_key = fold_in(base_key, step)
    train_step(model, batch, key=step_key)
```

**5. Document key usage in function signatures**
```python
# ✅ Good: Clear key requirements
def train_model(
    model,
    data,
    key_sampling: RngKey,  # For data sampling
    key_noise: RngKey,     # For DP noise
    key_init: RngKey,      # For initialization
) -> None:
    ...
```

### ❌ Don't Do This

**1. Don't mix RngKey with global seeds**
```python
# ❌ Bad: Mixing paradigms
torch.manual_seed(42)  # Global seed
key = RngKey(100)  # RngKey
noise = torch.randn(100)  # Which seed controls this?
```

**2. Don't reuse keys**
```python
# ❌ Bad: Key reuse creates correlation
key = RngKey(42)
noise1 = add_noise(grads1, key=key)  # First use
noise2 = add_noise(grads2, key=key)  # SAME KEY = CORRELATED!
# This violates DP independence assumptions
```

**3. Don't create keys inside loops**
```python
# ❌ Bad: Non-reproducible
for step in range(100):
    key = RngKey(secrets.randbits(32))  # Different each run!
    train_step(model, batch, key=key)

# ✅ Good: Reproducible
key = RngKey(42)
for step in range(100):
    key, step_key = split(key, num=2)
    train_step(model, batch, key=step_key)
```

**4. Don't share keys across ranks without thinking**
```python
# ❌ Bad: All ranks get same noise (violates DP!)
key = RngKey(42)  # Same on all ranks
noise_fn = gaussian_noise(key=key)

# ✅ Good: Split keys per rank
global_key = RngKey(42)
rank_keys = split(global_key, num=world_size)
my_key = rank_keys[rank]
noise_fn = gaussian_noise(key=my_key)
```

**5. Don't modify RngKey objects**
```python
# ❌ Bad: RngKey is immutable
key = RngKey(42)
key.seed = 100  # AttributeError!

# ✅ Good: Create new key
key = RngKey(100)
```

---

## Troubleshooting

### "My results aren't reproducible!"

**Symptoms**: Same seed gives different results across runs.

**Causes**:
1. **Using global seeds instead of RngKey**
   - Solution: Use RngKey consistently for all Opaque operations
   
2. **Mixing RngKey with PyTorch's global RNG**
   - Solution: Set PyTorch's global seed separately if using non-Opaque operations:
     ```python
     torch.manual_seed(42)  # For PyTorch operations
     key = RngKey(42)  # For Opaque operations
     ```

3. **Non-deterministic PyTorch backend operations**
   - Solution: Enable deterministic mode:
     ```python
     torch.use_deterministic_algorithms(True)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     ```

4. **Unavoidable framework randomness** (e.g., dropout in transformers)
   - Solution: Set global seed before model creation:
     ```python
     torch.manual_seed(42)
     model = TransformerModel(...)  # Dropout uses global seed
     ```

### "My distributed training has correlated noise!"

**Symptoms**: DP privacy budget is wrong, or ranks have identical noise.

**Cause**: All ranks using the same key for noise generation.

**Solution**: Split keys per rank:
```python
from opaque.random import RngKey, split
import torch.distributed as dist

world_size = dist.get_world_size()
rank = dist.get_rank()

# Split key per rank
global_key = RngKey(42)
rank_keys = split(global_key, num=world_size)
my_noise_key = rank_keys[rank]  # Each rank gets unique key

noise_fn = gaussian_noise(key=my_noise_key)
```

### "How do I restart training from a checkpoint?"

**Problem**: Need reproducible randomness when resuming.

**Solution**: Use `fold_in()` to derive checkpoint-specific keys:
```python
from opaque.random import RngKey, fold_in

# Original training
base_key = RngKey(42)
for step in range(1000):
    step_key = fold_in(base_key, step)
    train_step(model, batch, key=step_key)
    if step == 500:
        save_checkpoint(model, step=500)

# Resume from checkpoint
checkpoint_step = 500
model = load_checkpoint("checkpoint_500.pt")
resume_key = fold_in(base_key, checkpoint_step)

# Continue training with correct keys
for step in range(checkpoint_step, 1000):
    step_key = fold_in(base_key, step)
    train_step(model, batch, key=step_key)
```

### "I'm getting TypeError: 'RngKey' object is not iterable"

**Cause**: Trying to unpack a single RngKey.

**Solution**: Check `split()` return value:
```python
# ❌ Bad
key1, key2 = RngKey(42)  # RngKey is not iterable

# ✅ Good
key1, key2 = split(RngKey(42), num=2)
```

### "My keys have the same seed - why are they different?"

**Observation**: After splitting, subkeys have different seeds.

**Explanation**: This is expected! `split()` creates **independent** keys with **different** seeds:

```python
key = RngKey(42)
key1, key2 = split(key, num=2)

print(key1.seed)  # e.g., 1847392013 (derived from 42)
print(key2.seed)  # e.g., 3927583921 (different, also derived from 42)
```

The seeds look random, but are **deterministically derived** from the parent key. Same parent key → same subkey seeds every time.

---

## Migration Guide: From Global Seeds to RngKey

### Pattern 1: Basic Random Operations

**Before** (global seed):
```python
torch.manual_seed(42)
noise = torch.randn(100)
```

**After** (RngKey):
```python
from opaque.random import RngKey, generator_from_key

key = RngKey(42)
generator = generator_from_key(key)
noise = torch.randn(100, generator=generator)
```

### Pattern 2: Training Loop

**Before** (global seed):
```python
torch.manual_seed(42)
for epoch in range(num_epochs):
    for batch in dataloader:
        # Randomness from global state
        noisy_grads = add_dp_noise(grads)
        optimizer.step()
```

**After** (RngKey):
```python
from opaque.random import RngKey, split

key = RngKey(42)
for epoch in range(num_epochs):
    for batch in dataloader:
        # Explicit key splitting
        key, step_key = split(key, num=2)
        noisy_grads = add_dp_noise(grads, key=step_key)
        optimizer.step()
```

### Pattern 3: Distributed Training

**Before** (manual seed offsets):
```python
rank = torch.distributed.get_rank()
torch.manual_seed(42 + rank)  # Hope this doesn't overlap!
```

**After** (RngKey):
```python
from opaque.random import RngKey, split
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

global_key = RngKey(42)
rank_keys = split(global_key, num=world_size)
my_key = rank_keys[rank]  # Guaranteed non-overlapping
```

### Pattern 4: Multiple Random Components

**Before** (manual seed management):
```python
torch.manual_seed(42)
sampler = setup_sampler()
torch.manual_seed(43)
noise = setup_noise()
torch.manual_seed(44)
dropout = setup_dropout()
```

**After** (RngKey):
```python
from opaque.random import RngKey, split

key = RngKey(42)
key_sample, key_noise, key_dropout = split(key, num=3)

sampler = setup_sampler(key=key_sample)
noise = setup_noise(key=key_noise)
dropout = setup_dropout(key=key_dropout)
```

---

## Summary

**RngKey provides**:
- ✅ **Explicit randomness**: No hidden global state
- ✅ **Reproducibility**: Same key → same random numbers, always
- ✅ **Composability**: Split keys to create independent streams
- ✅ **Distributed-safe**: Easy to ensure non-overlapping randomness per rank
- ✅ **Debuggability**: Track exactly where randomness comes from

**Key patterns**:
1. Create key: `key = RngKey(42)`
2. Split key: `key1, key2 = split(key, num=2)`
3. Step-based keys: `step_key = fold_in(base_key, step)`
4. Distributed keys: `my_key = split(global_key, num=world_size)[rank]`

**Golden rule**: Never reuse keys. Always split to create independent randomness.

---

## Related Documentation

- [Noise APIs](noise.md) - Using RngKey with DP noise
- [Sampling](sampling.md) - Using RngKey with Poisson sampling
- [Auditing](auditing.md) - Using RngKey with privacy auditing
- [Distributed Training](distributed.md) - RngKey in DDP/FSDP setups
- [API Reference](../api/functional_utils.md) - Complete RngKey API
