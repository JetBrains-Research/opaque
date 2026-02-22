# Random Number Generation

Opaque uses explicit RNG keys instead of global random state. Every
function that involves randomness -- noise injection, sampling, adaptive
clipping, auditing -- takes a `key` parameter. This follows the JAX PRNG
model: no hidden state, fully deterministic, and safe in distributed
settings.

## Why explicit keys

Global random state (`torch.manual_seed`, `np.random.seed`) has two
problems for DP-SGD:

1. **Reproducibility is fragile.** Any library call that consumes random
   numbers silently advances the global counter, making results depend on
   import order, PyTorch version, and GPU scheduling.
2. **Distributed training is error-prone.** You must carefully manage which
   ranks share state and which diverge. A subtle bug can cause all ranks to
   generate the same noise (correct for centralized DP-SGD) or different
   noise (incorrect for centralized DP-SGD) without any visible error.

Explicit keys solve both problems. The key is a value, not hidden state.
You pass it explicitly, split it when you need independent randomness, and
fold in additional data (step counter, rank) to derive new keys
deterministically. The same key always produces the same output, regardless
of what other code has run.

## Core primitives

All primitives are in `opaque.random`:

### `key(seed)`

Create a key from an integer seed. This is the entry point for all
randomness in Opaque.

```python
from opaque.random import key

k = key(42)
# RngKey(seed=42, impl='opaque_threefry_like')
```

The returned `RngKey` is a frozen (immutable) dataclass with a `seed`
field (uint64) and an `impl` field identifying the hash function.

### `split(k, num=2)`

Derive `num` independent child keys from a parent key. Use this when you
need multiple independent sources of randomness.

```python
from opaque.random import key, split

k = key(42)
k_noise, k_sample = split(k)
k1, k2, k3 = split(k, num=3)
```

Each child is deterministically derived via `fold_in(k, i)` for index `i`.
The children are statistically independent: using one does not affect the
others.

**Golden rule:** never reuse a key for two different purposes. Always split
first.

```python
# Wrong: reusing the same key
noise_fn, ns = gaussian_noise(stddev=1.0, key=k)
sampler = PoissonSampler(dataset, sample_rate=0.01, key=k)  # correlated

# Right: split first
k_noise, k_sample = split(k)
noise_fn, ns = gaussian_noise(stddev=1.0, key=k_noise)
sampler = PoissonSampler(dataset, sample_rate=0.01, key=k_sample)
```

### `fold_in(k, data)`

Mix additional data into a key to derive a new key deterministically.
Accepts integers or strings.

```python
from opaque.random import key, fold_in

k = key(42)
step_key = fold_in(k, step)           # int: step counter
rank_key = fold_in(k, f"rank:{r}")    # str: domain separation
noise_key = fold_in(k, "noise")       # str: label
```

`fold_in` uses BLAKE2b hashing internally. Different data values produce
different keys:

```python
fold_in(k, 0).seed != fold_in(k, 1).seed
fold_in(k, "noise").seed != fold_in(k, "sampling").seed
fold_in(k, 42).seed != fold_in(k, "42").seed  # int vs str distinguished
```

Use `fold_in` for structured key derivation: step counters, rank indices,
domain labels, version strings.

### `generator_from_key(k)`

Convert an `RngKey` to a `torch.Generator` for use with PyTorch operations
that require one.

```python
from opaque.random import key, generator_from_key

gen = generator_from_key(key(42))
tensor = torch.randn(10, generator=gen)
```

This bridges Opaque's immutable keys with PyTorch's generator API. Use it
for non-DP operations (dropout, weight initialization) where you want
determinism but don't need Opaque's key management.

### `random_key()`

Create a non-deterministic key using system entropy. Each call returns a
different key.

```python
from opaque.random import random_key

k = random_key()  # different every time
```

Use for prototyping and exploration when reproducibility is not needed.
For production training, always use `key(seed)` with a fixed seed.

## Using keys with Opaque components

### Noise

```python
from opaque import gaussian_noise
from opaque.random import key

noise_fn, state = gaussian_noise(stddev=1.1, key=key(42))
noisy_grads, state = noise_fn(grads, state)
```

The noise function manages its own step counter internally. You provide a
key at construction time; each call to `noise_fn` derives a new key from
the base key and the current step counter, then increments the counter.

### Sampling

```python
from opaque import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))
```

### Adaptive clipping

```python
from opaque import adaptive_clipped_grad
from opaque.random import key

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn, initial_clip_norm=1.0, key=key(7),
)
```

### Auditing

```python
import opaque.auditing as auditing
from opaque.random import key

experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))
```

## Training loop patterns

### Simple (single device)

In the common case, you create keys once at the start and thread state
through the loop:

```python
from opaque import clipped_grad, gaussian_noise
from opaque.random import key, split

k_noise, k_sample = split(key(42))

grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=k_noise)

for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

### With `training_key`

`training_key` is a convenience function that implements a canonical
derivation chain for training loops: `base_seed -> step -> rank -> worker`.

```python
from opaque.random import training_key

for step in range(num_steps):
    k = training_key(base_seed=42, step=step)
    noise_fn, noise_state = gaussian_noise(stddev=1.1, key=k)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_seed` | `int` | required | Seed for the entire training run |
| `step` | `int` | required | Training step counter |
| `rank` | `int` or `None` | `None` | Distributed rank |
| `worker_id` | `int` or `None` | `None` | DataLoader worker ID |
| `synchronized` | `bool`, `"auto"`, or `None` | `None` | Noise synchronization policy |

The `synchronized` parameter controls whether the derived key is the same
across ranks or different:

| Value | Behavior | Use case |
|-------|----------|----------|
| `True` | Same key on all ranks (rank ignored) | Centralized DP-SGD (synchronized noise) |
| `False` | Different key per rank (`fold_in(k, rank)`) | Independent noise per device |
| `"auto"` | Synchronized if `rank is None`, else unsynchronized | Sensible default |
| `None` | Raises `ValueError` if `rank` is provided | Prevents accidental misuse |

The derivation order is: `key(base_seed) -> fold_in(step) -> fold_in(rank)
(if unsynchronized) -> fold_in(worker_id)`.

### Distributed training

In centralized DP-SGD, all ranks must add the **same** noise so that
models stay in sync. This requires the same key on every rank.

`gaussian_noise` handles this automatically: with the default
`synchronized="auto"`, it detects whether `torch.distributed` is
initialized and uses synchronized noise if so.

```python
# Same key on all ranks -> same noise -> models stay in sync
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))
```

For explicit control:

```python
from opaque.random import training_key
import torch.distributed as dist

rank = dist.get_rank()

# Synchronized: same key regardless of rank
k = training_key(base_seed=42, step=step, rank=rank, synchronized=True)

# Unsynchronized: different key per rank
k = training_key(base_seed=42, step=step, rank=rank, synchronized=False)
```

## Reproducibility

Same key produces identical output across runs, platforms, and devices:

```python
from opaque import gaussian_noise
from opaque.random import key
import torch

grads = {"w": torch.randn(100)}

noise_fn, s1 = gaussian_noise(stddev=1.0, key=key(42))
noisy1, _ = noise_fn(grads, s1)

noise_fn, s2 = gaussian_noise(stddev=1.0, key=key(42))
noisy2, _ = noise_fn(grads, s2)

assert torch.equal(noisy1["w"], noisy2["w"])
```

Opaque controls only its own randomness. PyTorch operations like
`torch.randn` or `torch.nn.Dropout` use PyTorch's global state and may
vary across platforms. Use `set_reproducible_pytorch_seed` to configure
framework-level determinism:

```python
from opaque.random import set_reproducible_pytorch_seed, key

set_reproducible_pytorch_seed(key(42))
# Sets torch.manual_seed, torch.cuda.manual_seed_all
# Enables torch.backends.cudnn.deterministic
# Disables torch.backends.cudnn.benchmark
```

This has a 10-30% performance cost due to deterministic algorithm
selection. Call it once at training startup if you need full reproducibility.

## Checkpoint and resume

Because keys are values (not stateful objects), checkpointing is
straightforward. Save the noise state and restore it:

```python
# Save
state = {"noise_state": noise_state, "step": step, ...}
torch.save(state, "checkpoint.pt")

# Resume
state = torch.load("checkpoint.pt")
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))
# Advance to the saved step by setting the internal counter
noise_state = state["noise_state"]
```

Alternatively, reconstruct the key from the step counter using
`training_key`:

```python
# Resume from step 500
k = training_key(base_seed=42, step=500)
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=k)
```

## API reference

See [Random API Reference](../api/rng.md) for complete function signatures
and return types.
