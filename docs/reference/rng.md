# Random Number Generation API

Opaque provides immutable, JAX-style RNG key semantics with explicit key
threading for deterministic and reproducible DP training.

## Overview

The backend-neutral `opaque.random` module provides:

- **Core primitives**: `RngKey`, `split()`, `fold_in()` — Functional RNG with immutable keys
- **Sampling**: `normal(rng_key, shape, *, dtype=None, like=None)` — Backend-native
  normal sampling determined solely by an immutable key
- **Convenience helpers**:
  - `key()` — Create an RngKey from an integer seed
  - `random_key()` — Nondeterministic key from system entropy

The `opaque-torch` provider adds `opaque.torch.random` helpers:

- `set_reproducible_pytorch_seed()` — Configure PyTorch/cuDNN reproducibility
- `generator_from_key()` — Create a `torch.Generator` from an RngKey for
  Torch compatibility APIs

## Contract and portability

`RngKey` is immutable. `key`, `split`, and `fold_in` have stable,
backend-neutral derivation semantics, while `normal()` is native sampling:
reusing the same key and arguments replays within the active backend without
consuming framework global RNG state. Torch, JAX, and MLX are not required to
return equal sample values for the same key.

`normal()` honors requested shape and dtype; providers use `like` for supported
placement according to their native device policy. Unsupported dtype/device
requests retain the provider's native error behavior; Opaque does not silently
move sampling to another provider. Global model/framework randomness is outside
this contract.

`random_key()` is the explicit system-entropy convenience boundary.
Deterministic APIs do not obtain entropy implicitly. Host-side sampling such as
sampler index selection and auditing uses a private, explicitly keyed NumPy
generator; it is deterministic for the installed NumPy implementation but is
not a cross-provider or cross-NumPy-version bitstream promise.

Noise mechanisms domain-separate pytree leaves by their current flattening
index, and adaptive per-group clipping uses sorted group order. Preserve that
structure across a checkpoint: reordering or inserting leaves/groups can
change the substream assigned to an existing leaf/group.

## Quick Reference

### Essential Imports

```python
# Core primitives
from opaque.random import fold_in, key, random_key, split
from opaque.random.types import RngKey

# Torch-specific helpers (provided by opaque-torch)
from opaque.torch.random import generator_from_key, set_reproducible_pytorch_seed
```

### Creating Keys

```python
# From integer seed (reproducible)
k = key(42)

# Nondeterministic (for prototyping)
k = random_key()  # Uses system entropy

# From torch.Generator
gen = torch.Generator().manual_seed(42)
k = RngKey(seed=42)  # Equivalent for reproducible use
```

### Splitting Keys

```python
# Binary split (most common)
k1, k2 = split(k, num=2)

# Multi-way split
keys = split(k, num=10)  # Returns tuple

# Loop pattern — split and advance
for step in range(100):
    k, step_key = split(k, num=2)
    result = do_something_random(step_key)
```

### Per-Step Derivation

```python
# Using fold_in for step-based keys
base_key = key(42)
step_key = fold_in(base_key, 0)

# Multiple values (variadic)
step_rank_key = fold_in(base_key, step, rank)
```

### PyTorch Reproducibility

```python
# Set all PyTorch/CUDNN seeds from RngKey
from opaque.random import fold_in, key
from opaque.torch.random import set_reproducible_pytorch_seed

set_reproducible_pytorch_seed(key(42))

# Then use fold_in for per-step DP randomness
base = key(42)
for step in range(num_steps):
    step_key = fold_in(base, step)
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

Create a nondeterministic RngKey using system entropy.

```python
from opaque.random import random_key

k = random_key()  # Each call returns different key
```

Useful for prototyping and experiments. This is the explicit system-entropy
boundary; for reproducible training, use `key()` with an explicit seed.

**Returns:** RngKey with seed from `secrets.randbits(64)`

---

#### split(rng_key: RngKey, num: int = 2) → tuple of RngKey values

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

#### fold_in(rng_key: RngKey, *data: int | str) → RngKey

Deterministically derive a new key by folding in one or more values.

Accepts a variable number of int/str arguments. Each value is folded
sequentially, so `fold_in(k, a, b)` equals `fold_in(fold_in(k, a), b)`.

```python
from opaque.random import fold_in, key

base_key = key(42)

# Single value
step_key = fold_in(base_key, 0)

# Multiple values (variadic)
step_rank_key = fold_in(base_key, step, rank)

# Full derivation chain
full_key = fold_in(base_key, step, rank, worker_id)

# String data
key_v2 = fold_in(base_key, "v2")
```

**Args:**
- `rng_key`: Base key
- `*data`: One or more int or str values to fold in sequentially

**Returns:** New RngKey derived from `rng_key` and all folded values

**Raises:**
- TypeError if any value is not int or str
- ValueError if no data values are provided

**Use Cases:**
- Step counters in loops: `fold_in(base, step)`
- Distributed training: `fold_in(base, step, rank)`
- DataLoader workers: `fold_in(base, step, rank, worker_id)`
- Checkpointing (deterministic resume key)
- Versioning DP mechanisms

---

#### normal(rng_key: RngKey, shape, *, dtype=None, like=None) → native array

Draw a backend-native standard-normal array from an immutable key. Reusing the
same key and arguments returns the same result within the active backend and
does not advance the backend's global generator.

`dtype` selects the output dtype; when omitted, `like.dtype` or the provider
default is used. `like` also supplies supported provider placement. Native
algorithms and device support differ across providers, so this function does
not promise cross-backend equality or support for every dtype/device pair.

---

#### generator_from_key(rng_key: RngKey) → torch.Generator

Create a deterministic `torch.Generator` from an RngKey.

```python
import torch
from opaque.random import key
from opaque.torch.random import generator_from_key

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

Configure PyTorch and cuDNN for reproducible training from a single RngKey.

```python
from opaque.random import fold_in, key
from opaque.torch.random import set_reproducible_pytorch_seed

# At training start
set_reproducible_pytorch_seed(key(42))

# Then use fold_in for per-step DP
base = key(42)
for step in range(num_steps):
    step_key = fold_in(base, step)
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

**Performance impact:** Deterministic algorithms may be slower; measure on
your workload.

**Example:**

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import fold_in, key, split
from opaque.torch.random import set_reproducible_pytorch_seed

# Setup framework reproducibility once
set_reproducible_pytorch_seed(key(42))

# Training loop with per-step DP randomness
base = key(42)
for step in range(1000):
    # Deterministic per-step key
    step_key = fold_in(base, step)
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

**Using fold_in()** (recommended):

```python
from opaque.distributed import get_rank
from opaque.random import key, fold_in

rank = get_rank()
base = key(42)

for step in range(steps):
    # Synchronized noise — same key on all ranks (no rank folded in)
    noise_key = fold_in(base, step)

    # Independent local sampling — derive a separate rank domain.
    sample_key = fold_in(base, "sampler", rank, step)
```

Use the synchronized key for mechanisms that must add identical noise on every
rank. Fold in the rank only for independently randomized components, such as
per-rank sampling. Split component keys before either derivation.

**Manual approach** (for reference):

```python
from opaque.random import split, key

master = key(42)
sampling_key, _noise_key = split(master, num=2)

# Independent local component: per-rank keys
rank_keys = split(sampling_key, num=world_size)
my_sampler_key = rank_keys[rank]

sampler = PoissonSampler(..., key=my_sampler_key)
```

## Troubleshooting

### Results not reproducible?

Opaque keyed operations are isolated from global RNG draws. To reproduce an
application, also explicitly configure randomness in user models and framework
transforms:

```python
# Correct: Use RngKey throughout
from opaque.random import fold_in, key
from opaque.torch.random import set_reproducible_pytorch_seed

set_reproducible_pytorch_seed(key(42))  # Framework
base = key(42)
for step in range(n):
    k = fold_in(base, step)  # DP operations
    # ... training ...

# Keyed DP noise remains independent of framework-global draws.
noise_fn = gaussian_noise(..., key=some_key)
```

### Distributed ranks have correlated noise?

That is correct for centralized DP mechanisms that must add the same noise on
every rank. Use a rank-derived key only for components that require independent
local streams:

```python
# Synchronized centralized noise
k = key(42)
noise_fn = gaussian_noise(..., key=k)  # All ranks get identical noise

# Independent local sampler stream
sampler = PoissonSampler(..., key=fold_in(key(42), "sampler", rank))

# Also valid: split a dedicated sampler component first.
sampler_root, _ = split(key(42))
sampler = PoissonSampler(..., key=split(sampler_root, num=world_size)[rank])
```

### Need to resume from checkpoint?

Save and restore the complete functional state rather than recreating a
stateful mechanism from a step-derived key:

```python
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict

# Save the state returned by the latest noise call.
checkpoint = state_dict(noise_state)

# Rebuild the same mechanism configuration and restore its key, cursor, and
# any streaming state from the snapshot.
noise_fn, template_state = gaussian_noise(..., key=key(0))
noise_state = from_state_dict(template_state, checkpoint)

# The next noise_fn call matches the uninterrupted sequence in this backend.
```

## Further Reading

- [RngKey User Guide](../user-guide/rng-key.md) - Conceptual guide
- [Noise APIs](noise.md) - Using keys with noise injection
- [Sampling](sampling.md) - Using keys with samplers
- [Distributed Training](distributed.md) - DDP patterns with keys
