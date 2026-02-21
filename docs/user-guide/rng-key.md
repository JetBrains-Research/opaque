# Random Number Generation with RngKey

## Overview

Opaque uses **explicit random keys** instead of global seeds.  Every function
that involves randomness (noise, sampling, auditing) takes a `key` parameter
created with `key(seed)`.  This follows the JAX PRNG model: no hidden state,
deterministic, and safe in distributed settings.

```python
from opaque.random import key

k = key(42)  # deterministic
```

## Core Primitives

All primitives live in `opaque.random`:

| Function | Purpose |
|----------|---------|
| `key(seed)` | Create a key from an integer seed |
| `split(k, num=2)` | Derive `num` independent child keys |
| `fold_in(k, data)` | Mix a key with an integer or string |
| `generator_from_key(k)` | Convert to `torch.Generator` |
| `random_key()` | Non-deterministic key (uses system entropy) |
| `training_key(...)` | Ergonomic key derivation for training loops |

### `key(seed)`

```python
from opaque.random import key

k = key(42)
# RngKey(seed=42, impl='opaque_threefry_like')
```

### `split(k, num=2)`

Create independent child keys:

```python
from opaque.random import key, split

k = key(42)
k1, k2 = split(k)
noise_key, sample_key, clip_key = split(k, num=3)
```

### `fold_in(k, data)`

Mix additional data into a key:

```python
from opaque.random import key, fold_in

k = key(42)
step_key = fold_in(k, step)        # int
rank_key = fold_in(k, f"rank:{r}")  # str
```

## Using Keys with Opaque

### Noise

```python
from opaque import gaussian_noise
from opaque.random import key

noise_fn, state = gaussian_noise(stddev=1.1, key=key(42))
noisy_grads, state = noise_fn(grads, state)
```

### Sampling

```python
from opaque import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))
```

### Adaptive Clipping

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

## Training Loop Patterns

### Simple (Single Device)

The noise function manages its own key progression internally via a step
counter.  You only need to provide a key at construction time:

```python
from opaque import clipped_grad, gaussian_noise
from opaque.random import key

grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))

for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

### With `training_key()` (Per-Step Keys)

For explicit control over per-step randomness:

```python
from opaque.random import training_key

for step in range(num_steps):
    k = training_key(base_seed=42, step=step)
    noise_fn, noise_state = gaussian_noise(stddev=1.1, key=k)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
```

### Distributed Training

In DDP, noise must be **synchronized** (same key on all devices).
`gaussian_noise` handles this automatically with `synchronized="auto"`:

```python
# Same key on all ranks → same noise → models stay in sync
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))
```

For explicit rank control:

```python
from opaque.random import training_key
import torch.distributed as dist

rank = dist.get_rank()

# Synchronized noise (same key regardless of rank)
k = training_key(base_seed=42, step=step, rank=rank, synchronized=True)

# Independent noise per rank
k = training_key(base_seed=42, step=step, rank=rank, synchronized=False)
```

## Reproducibility

Same key → same output, across runs, platforms, and devices:

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

!!! note
    Opaque controls only its own randomness.  PyTorch operations like
    `torch.randn` may vary across platforms.

## Migration from Global Seeds

If you're coming from Opacus or plain PyTorch:

```python
# ❌ Old pattern
torch.manual_seed(42)
noise = torch.randn_like(grads) * sigma

# ✅ Opaque pattern
from opaque import gaussian_noise
from opaque.random import key

noise_fn, state = gaussian_noise(stddev=sigma, key=key(42))
noisy_grads, state = noise_fn(grads, state)
```

## See Also

- [Noise Addition](noise.md) — using keys with noise functions
- [Distributed Training](distributed.md) — synchronized vs. independent keys
- [API Reference](../api/rng.md) — full `opaque.random` API
