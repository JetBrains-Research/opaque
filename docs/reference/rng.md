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

`RngKey` is immutable, and `key`, `split`, and `fold_in` derive from it with
backend-neutral semantics. `normal()` is native sampling: the same key and
arguments replay within the active provider, without consuming the framework's
global RNG state, but providers are not required to agree on the values. It
honors the requested shape and dtype and takes placement from `like`;
unsupported dtype or device requests raise the provider's own error rather than
moving the draw elsewhere.

`random_key()` is the explicit system-entropy boundary — deterministic APIs
never reach for entropy implicitly. Host-side sampling (sampler index selection,
auditing) uses a private NumPy generator built from an explicit key, which
replays for the installed NumPy but is not a bitstream promise across versions.

Noise mechanisms domain-separate pytree leaves by their current flattening
index, and adaptive per-group clipping uses sorted group order. Preserve that
structure across a checkpoint: reordering or inserting leaves/groups can
change the substream assigned to an existing leaf/group.

## Domain separation

`fold_in` hashes integers and strings down disjoint paths: `fold_in(k, 1)` and
`fold_in(k, "1")` are different keys, and no chain of integer folds can reach a
key derived through a string fold. Opaque divides the two.

**Integers are the caller's** — steps, ranks, epochs, leaf and group indices,
and every key `split` returns, since `split(k, n)` *is*
`fold_in(k, i) for i in range(n)`.

**Strings root a mechanism.** Anything that draws randomness folds one unique,
namespaced string into the key it was handed, once, and derives everything else
beneath that tag:

```python
from opaque.random import fold_in, normal

MY_STREAM = "mylab.rare_events.noise"   # yours, and nobody else's


def my_noise(grads, *, key, step):
    step_key = fold_in(key, MY_STREAM, step)
    return [
        leaf + normal(fold_in(step_key, i), leaf.shape, like=leaf)
        for i, leaf in enumerate(grads)
    ]
```

Without a root, the derivation you would write is `fold_in(key, step)` — and so
is everyone else's. Two mechanisms handed the same base key then draw the same
numbers, and nothing reports it: not a test, not an error, not an accountant.

Keys are pure inputs, so two further rules follow:

- **Every distinct draw needs a distinct key.** Calling `normal` twice with one
  key replays the same values; it does not continue a stream.
- **One key, two shapes are not independent.** `normal(k, (4,))` is a prefix of
  `normal(k, (8,))`. Fold before changing shape.

### Tags Opaque already occupies

Each shipped mechanism roots its own key space. Do not reuse these tags; any
other namespaced string is free.

| Tag | Mechanism |
| --- | --- |
| `opaque.dpsgd.gaussian` | `opaque.dpsgd.noise.gaussian_noise` (both streams) |
| `opaque.dpsgd.adaptive_clipping` | `opaque.dpsgd.clipping` adaptive threshold noise |
| `opaque.dpftrl.mf_gaussian` | `opaque.dpftrl.noise.mf_gaussian_noise` |
| `opaque.dpftrl.second_moment.first` / `.second` | paired MF release with private second moments |
| `opaque.paired.first` / `opaque.paired.second` | paired first/second-moment streams |
| `opaque.auditing.canary_selection` / `opaque.auditing.coin_flip` | `opaque.auditing.coin_flip` |

Sub-derivations beneath a root — a step counter, a leaf index, `"leaf"`,
`"column"` — need no namespace of their own.

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
```

`key()` and `random_key()` are the only supported ways to make a key; the
`RngKey` dataclass is the type they return, for annotations and `isinstance`.
To hand a key to Torch, convert in that direction with
[`generator_from_key`](#generator_from_keyrng_key-rngkey--torchgenerator).

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
# Root your mechanism once, then derive integers beneath it
base_key = key(42)
stream = fold_in(base_key, "mylab.rare_events.noise")
step_key = fold_in(stream, 0)

# Multiple values (variadic)
step_rank_key = fold_in(stream, step, rank)
```

Passing `base_key` straight to a shipped mechanism is fine — `gaussian_noise`
and friends root themselves. The root above is for randomness *you* draw.

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
- TypeError if `rng_key` is not an `RngKey`, or any value is not int or str
  (`bool` is rejected even though it subclasses `int`)
- ValueError if no data values are provided

Integers and strings are hashed down disjoint paths, which is what lets a
string tag root a mechanism's key space — see
[Domain separation](#domain-separation) for the convention and
[Tags Opaque already occupies](#tags-opaque-already-occupies) for the tags
that are taken.

**Use Cases:**
- Rooting a mechanism you wrote: `fold_in(base, "mylab.my_mechanism")` —
  see [Domain separation](#domain-separation)
- Step counters in loops: `fold_in(stream, step)`
- Distributed training: `fold_in(stream, step, rank)`
- DataLoader workers: `fold_in(stream, step, rank, worker_id)`
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
model = initialize_model(init_key)  # keyed model initialization
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
