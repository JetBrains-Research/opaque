# Random Number Generation API

Opaque provides immutable, JAX-style RNG key semantics with explicit key threading for deterministic and reproducible DP training.

## Overview

The `opaque.random` module provides:

- **Core primitives**: `RngKey`, `split()`, `fold_in()` - Functional RNG with immutable keys
- **Convenience helpers**:
  - `key()` - Create RngKey from integer seed
  - `random_key()` - Non-deterministic key from system entropy
  - `training_key()` - Deterministic keys for training loops with proper derivation order
  - `set_reproducible_pytorch_seed()` - Configure PyTorch/CUDNN reproducibility
- **Bridge function**: `generator_from_key()` - Create torch.Generator from RngKey

## Quick Reference

### Essential Imports

```python
# Core primitives
from opaque.random import RngKey, split, fold_in, key, generator_from_key

# Convenience helpers
from opaque.random import random_key, training_key, set_reproducible_pytorch_seed
```

### Creating Keys

```python
# From integer seed (reproducible)
k = key(42)

# Non-deterministic (for prototyping)
k = random_key()  # Uses system entropy

# From torch.Generator
gen = torch.Generator().manual_seed(42)
k = RngKey(seed=42)  # Equivalent for reproducible use
```

### Splitting Keys

```python
# Binary split (most common)
k1, k2 = split(key, num=2)

# Multi-way split
keys = split(key, num=10)  # Returns tuple

# Loop pattern - split and advance
for step in range(100):
    key, step_key = split(key, num=2)
    result = do_something_random(step_key)
```

### Per-Step Derivation

```python
# Using fold_in for step-based keys
base_key = key(42)
step_key = fold_in(base_key, step=0)

# Or use training_key helper
step_key = training_key(base_seed=42, step=0)
```

### PyTorch Reproducibility

```python
# Set all PyTorch/CUDNN seeds from RngKey
from opaque.random import set_reproducible_pytorch_seed

set_reproducible_pytorch_seed(key(42))

# Then use training_key for per-step DP randomness
for step in range(num_steps):
    step_key = training_key(base_seed=42, step=step)
    # ... training step ...
```

## API Reference

### Classes

#### RngKey

```python
@dataclass(frozen=True)
class RngKey:
    seed: int
    impl: str = "opaque_threefry_like"
```

Immutable RNG key. Thread explicitly through functions for deterministic randomness.

**Attributes:**
- `seed`: Integer seed value (main key material)
- `impl`: Implementation identifier (default: "opaque_threefry_like")

### Functions

#### key(seed: int) → RngKey

Create an RngKey from an integer seed.

```python
from opaque.random import key

k = key(42)
```

**Args:**
- `seed`: Integer seed value

**Returns:** RngKey

**Raises:** TypeError if seed is not an integer

---

#### random_key() → RngKey

Create a non-deterministic RngKey using system entropy.

```python
from opaque.random import random_key

k = random_key()  # Each call returns different key
```

Useful for prototyping and experiments. For reproducible training, use `key()` or `training_key()`.

**Returns:** RngKey with seed from `secrets.randbits(64)`

---

#### split(rng_key: RngKey, num: int = 2) → tuple[RngKey, ...]

Split a key into `num` independent child keys.

```python
from opaque.random import split, key

k = key(42)
k1, k2 = split(k, num=2)

# Multi-way split
keys = split(k, num=10)
```

**Args:**
- `rng_key`: Key to split
- `num`: Number of child keys (default: 2)

**Returns:** Tuple of `num` independent RngKeys

**Raises:** ValueError if `num < 1`

**Golden Rule:** Never reuse keys. Always split to create independent randomness.

---

#### fold_in(rng_key: RngKey, data: int | str) → RngKey

Deterministically derive a new key from additional data (typically a counter).

```python
from opaque.random import fold_in, key

base_key = key(42)

# Derive step-specific keys
step_0_key = fold_in(base_key, 0)
step_1_key = fold_in(base_key, 1)

# or with string data
key_v2 = fold_in(base_key, "v2")
```

**Args:**
- `rng_key`: Base key
- `data`: Integer or string to fold in

**Returns:** New RngKey derived from `rng_key` and `data`

**Raises:** TypeError if `data` is not int or str

**Use Cases:**
- Step counters in loops
- Checkpointing (deterministic resume key)
- Debugging specific iterations
- Versioning DP mechanisms

---

#### training_key(base_seed: int, step: int, rank: int | None = None, worker_id: int | None = None, synchronized: bool | Literal["auto"] | None = None) → RngKey

Deterministic training loop key with proper derivation order.

```python
from opaque.random import training_key

# Simple training loop
for step in range(100):
    k = training_key(base_seed=42, step=step)
    # Use k for this step's randomness

# Distributed training - per-rank noise
import torch.distributed as dist
rank = dist.get_rank()
k = training_key(
    base_seed=42,
    step=step,
    rank=rank,
    synchronized=False,  # Different key per rank
)

# Distributed training - synchronized noise
k = training_key(
    base_seed=42,
    step=step,
    rank=rank,
    synchronized=True,  # Same key on all ranks
)
```

**Args:**
- `base_seed` (int): Reproducible seed for entire training run
- `step` (int): Training step counter (folded first)
- `rank` (int | None): Distributed rank (folded after step if unsynchronized)
- `worker_id` (int | None): DataLoader worker ID (folded last)
- `synchronized` (bool | "auto" | None, default: None):
  - `True`: Same key for all ranks (centralized DP-SGD)
  - `False`: Different keys per rank (per-rank noise)
  - `"auto"`: Synchronized if `rank is None`, unsynchronized if rank provided
  - `None`: Must not pass `rank` (raises ValueError)

**Returns:** RngKey following step → rank → worker_id derivation order

**Raises:**
- ValueError if `rank` is passed without specifying `synchronized`
- ValueError if `synchronized` has invalid value

**Key Pattern:** Always use `step → rank → worker_id` derivation order for consistent and correct distributed training behavior.

---

#### generator_from_key(rng_key: RngKey) → torch.Generator

Create a deterministic `torch.Generator` from an RngKey.

```python
from opaque.random import generator_from_key, key
import torch

k = key(42)
gen = generator_from_key(k)

# Use with torch operations
tensor = torch.randn(10, generator=gen)
```

**Args:**
- `rng_key`: RngKey to convert

**Returns:** torch.Generator seeded with rng_key.seed

**Note:** The Generator is deterministic but independent of Opaque's DP noise. Use for framework-level randomness (e.g., dropout, model initialization) separate from DP operations.

---

#### set_reproducible_pytorch_seed(key_val: RngKey) → None

Configure PyTorch and CUDNN for reproducible training from a single RngKey.

```python
from opaque.random import key, training_key, set_reproducible_pytorch_seed

# At training start
set_reproducible_pytorch_seed(key(42))

# Then use training_key for per-step DP
for step in range(num_steps):
    step_key = training_key(base_seed=42, step=step)
    # ... training with deterministic framework + DP randomness
```

Sets:
- `torch.manual_seed()` for CPU operations
- `torch.cuda.manual_seed_all()` for GPU operations
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)` (if available)
- `CUBLAS_WORKSPACE_CONFIG` environment variable for cuBLAS determinism

**Args:**
- `key_val`: RngKey (typically from `key()`)

**Returns:** None (side effects only)

**Performance Impact:** Typically 10-30% slower with full determinism.

**Example:**

```python
from opaque.random import key, training_key, set_reproducible_pytorch_seed
from opaque.noise import gaussian_noise
from opaque.sampling import PoissonSampler

# Setup framework reproducibility once
set_reproducible_pytorch_seed(key(42))

# Training loop with per-step DP randomness
for step in range(1000):
    # Deterministic per-step key
    step_key = training_key(base_seed=42, step=step)
    step_key, noise_key = split(step_key, num=2)
    
    # Use for DP
    noise_fn, state = gaussian_noise(1.1, key=noise_key)
    grads = compute_dp_grads(model, batch)
    noisy_grads, state = noise_fn(grads, state)
    optimizer.step(noisy_grads)
```

**See Also:**
- torch.use_deterministic_algorithms - PyTorch determinism documentation
- torch.backends.cudnn - CUDNN configuration

## Common Patterns

### Component Splitting

Split master key for different components:

```python
from opaque.random import split, key

master = key(42)
sampling_key, noise_key, init_key = split(master, num=3)

sampler = PoissonSampler(..., key=sampling_key)
noise_fn, _ = gaussian_noise(..., key=noise_key)
model = initialize_model(init_key)  # If using jax
```

### Sequential Splitting (Loop Pattern)

Thread key through loop, splitting at each step:

```python
from opaque.random import split, key

k = key(42)
for step in range(100):
    k, step_k = split(k, num=2)
    result = train_step(model, batch, key=step_k)
```

### Distributed Training

**Using training_key()** (recommended):

```python
import torch.distributed as dist
from opaque.random import training_key

rank = dist.get_rank()

for step in range(steps):
    # Sampling: synchronized across ranks
    sample_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=True,
    )
    
    # Noise: independent per rank
    noise_key = training_key(
        base_seed=42,
        step=step,
        rank=rank,
        synchronized=False,
    )
```

**Manual approach** (for reference):

```python
from opaque.random import split, key

master = key(42)
sampling_key, noise_master = split(master, num=2)

# Noise: per-rank keys
rank_keys = split(noise_master, num=world_size)
my_noise_key = rank_keys[rank]

noise_fn = gaussian_noise(..., key=my_noise_key)
```

## Troubleshooting

### Results not reproducible?

Ensure all randomness uses RngKey:

```python
# Correct: Use RngKey throughout
from opaque.random import set_reproducible_pytorch_seed, training_key

set_reproducible_pytorch_seed(key(42))  # Framework
for step in range(n):
    k = training_key(42, step=step)  # DP operations
    # ... training ...

# Incorrect: Mixing with global seeds
torch.manual_seed(42)  # Framework
noise_fn = gaussian_noise(..., key=some_key)  # DP
# Different RNG sources can cause issues
```

### Distributed ranks have correlated noise?

Ensure rank-specific keys:

```python
# Incorrect: All ranks share the same key
k = key(42)
noise_fn = gaussian_noise(..., key=k)  # All ranks get identical noise

# Correct: Per-rank keys
k_base = key(42)
rank_keys = split(k_base, num=world_size)
noise_fn = gaussian_noise(..., key=rank_keys[rank])
```

### Need to resume from checkpoint?

Use `fold_in()` for deterministic resume:

```python
from opaque.random import fold_in, key

base_key = key(42)

# Save checkpoint
checkpoint = {"step": 500, ...}

# Resume
for step in range(checkpoint["step"], total_steps):
    step_key = fold_in(base_key, step)
    # ... training ...
```

## Further Reading

- [RngKey User Guide](../user-guide/rng-key.md) - Comprehensive conceptual guide
- [Noise APIs](noise.md) - Using keys with noise injection
- [Sampling](sampling.md) - Using keys with samplers
- [Distributed Training](distributed.md) - DDP/FSDP patterns with keys
