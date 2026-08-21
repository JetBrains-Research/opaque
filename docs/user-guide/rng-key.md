# Random Number Generation

Opaque uses explicit RNG keys instead of global random state. Stochastic
library APIs receive a `key` directly or state that retains one. This follows
the JAX PRNG model: no hidden state is consumed, results replay within an
active backend, and key derivation is explicit in distributed settings.

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

| Function | Purpose |
|----------|---------|
| `key(seed)` | Create a key from an integer seed |
| `split(k, num=2)` | Derive `num` independent child keys |
| `fold_in(k, *data)` | Mix a key with one or more int/str values |
| `normal(k, shape, *, dtype=None, like=None)` | Draw backend-native keyed normal samples |
| `opaque.torch.random.generator_from_key(k)` | Convert to `torch.Generator` |
| `random_key()` | Non-deterministic key (uses system entropy) |

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
noise_fn, ns = gaussian_noise(noise_multiplier=1.0, key=k)
sampler = PoissonSampler(dataset, sample_rate=0.01, key=k)  # correlated

# Right: split first
k_noise, k_sample = split(k)
noise_fn, ns = gaussian_noise(noise_multiplier=1.0, key=k_noise)
sampler = PoissonSampler(dataset, sample_rate=0.01, key=k_sample)
```

### `fold_in(k, *data)`

Mix additional data into a key. Variadic — accepts multiple values:

```python
from opaque.random import key, fold_in

k = key(42)
step_key = fold_in(k, step)              # single int
rank_key = fold_in(k, f"rank:{r}")        # single str
combined = fold_in(k, step, rank)         # multiple values
full     = fold_in(k, step, rank, worker) # step → rank → worker
```

`fold_in(k, a, b)` is equivalent to `fold_in(fold_in(k, a), b)`.

`fold_in` uses BLAKE2b hashing internally. Different data values produce
different keys, and `int` versus `str` values are distinguished:

```python
fold_in(k, 0).seed != fold_in(k, 1).seed
fold_in(k, 42).seed != fold_in(k, "42").seed  # int vs str distinguished
```

That last line is more than a curiosity: integers and strings are hashed down
disjoint paths, and Opaque divides them. Integers belong to the caller — steps,
ranks, epochs, leaf and group indices, and every key `split` returns. A unique,
namespaced string tag roots a mechanism; code that draws its own randomness
folds one in, then derives everything beneath it.

```python
stream = fold_in(k, "mylab.rare_events.noise")   # yours, and nobody else's
step_key = fold_in(stream, step)
```

Skip the root and you write `fold_in(k, step)`, which is what every mechanism
writes: two of them handed the same base key draw byte-identical noise, and no
test, error, or accountant reports it. Passing a bare `key(42)` to
`gaussian_noise` or a sampler is fine — those root themselves. See
[Domain separation](../reference/rng.md#domain-separation) for the rule and the
tags already taken.

### Backend semantics and limits

`key`, `split`, and `fold_in` are backend-neutral: the same inputs derive the
same `RngKey` whichever provider is active. Array-valued sampling is
deliberately different — `normal()` hands the key to the active provider's
native implementation, so the same arguments replay within one provider but are
not promised to agree across providers. It takes dtype and placement from
`dtype` or `like`, and an unsupported dtype or device raises the provider's own
error rather than falling back. Keyed sampling never consumes the framework's
global random state; randomness inside your model does not become keyed by
being called from Opaque.

### `generator_from_key(k)`

Convert an `RngKey` to a `torch.Generator` for use with PyTorch operations
that require one.

```python
from opaque.random import key
from opaque.torch.random import generator_from_key

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
`random_key()` is the explicit system-entropy boundary; deterministic Opaque
APIs never call it as an implicit fallback. For production training, always
use `key(seed)` with a fixed seed and record that seed with the checkpoint.

## Using keys with Opaque components

### Noise

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=key(42))
noisy_grads, state = noise_fn(grads, state)
```

The noise function manages its own step counter internally. You provide a
key at construction time; each call to `noise_fn` derives a new key from
the base key and the current step counter, then increments the counter.

### Sampling

```python
from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))
```

Samplers and auditing utilities select host-side indices with a private
`numpy.random.default_rng` built from an explicit, domain-separated key — never
NumPy's global RNG. Those streams replay for the installed NumPy, but are not a
bitstream promise across versions.

### Adaptive clipping

```python
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.random import key

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn, initial_clipping_norm=1.0, key=key(7),
)
```

### Auditing

`coin_flip()` derives separate audit subkeys for canary selection and inclusion
coins. Training mechanisms should use their own RNG domains so their
randomness remains independent from auditing, even with a shared reproducible
root seed.

```python
import opaque.auditing as auditing
from opaque.random import key

cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
```

## Training loop patterns

### Simple (single device)

In the common case, you create keys once at the start and thread state
through the loop:

```python
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key, split

k_noise, k_sample = split(key(42))

grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=1.0, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=k_noise)

for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

### With `fold_in()` (per-step keys)

Use `fold_in` to derive per-step keys from a base key. This gives
explicit control over the derivation chain:
`key(seed) → fold_in(step) → fold_in(rank) → fold_in(worker)`.

```python
from opaque.random import key, fold_in

base = key(42)
for step in range(num_steps):
    step_key = fold_in(base, step)
    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=step_key)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
```

### Distributed training

In centralized DP-SGD, all ranks must add the **same** noise so that
models stay in sync. Pass the same `key(seed)` on every rank:

```python
# Same key on all ranks → same noise → models stay in sync
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))
```

For per-rank key control, use `fold_in()`:

```python
from opaque.distributed import get_rank
from opaque.random import key, fold_in

rank = get_rank()

# Synchronized noise — same key, no rank folded in
step_key = fold_in(key(42), step)

# Independent local component — use a separate named domain and rank
sample_key = fold_in(key(42), "sampler", rank, step)

# Equivalent explicit chaining
sample_key = fold_in(fold_in(key(42), "sampler"), rank, step)
```

Use the same noise key on every rank only when the mechanism requires
synchronized noise, such as centralized DP-SGD. For independent local
components, derive a component key first and then fold in a named rank domain.

### Pytree and group streams

Gaussian and matrix-factorization noise derive one stream per pytree leaf from
the leaf's flattening index, and adaptive clipping derives per-group streams
from sorted group order. Reordering or inserting leaves or groups therefore
changes the substream an otherwise unchanged leaf gets, so treat that ordering
as part of the mechanism's checkpointed configuration.

## Reproducibility

Within one backend and supported dtype/device configuration, the same key and
arguments produce identical output across runs:

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
import torch

grads = {"w": torch.randn(100)}

noise_fn, s1 = gaussian_noise(noise_multiplier=1.0, key=key(42))
noisy1, _ = noise_fn(grads, s1)

noise_fn, s2 = gaussian_noise(noise_multiplier=1.0, key=key(42))
noisy2, _ = noise_fn(grads, s2)

assert torch.equal(noisy1["w"], noisy2["w"])
```

Opaque promises semantic portability — stable key derivation and keyed native
sampling — not a shared cross-backend bitstream. Provider-native device
determinism can also be limited by the framework and hardware.

Opaque controls only its own randomness. PyTorch operations like
`torch.randn` or `torch.nn.Dropout` use PyTorch's global state and may
vary across platforms. The `opaque-torch` provider exposes
`set_reproducible_pytorch_seed` for framework-level determinism:

```python
from opaque.random import key
from opaque.torch.random import set_reproducible_pytorch_seed

set_reproducible_pytorch_seed(key(42))
# Sets torch.manual_seed, torch.cuda.manual_seed_all
# Enables torch.backends.cudnn.deterministic
# Disables torch.backends.cudnn.benchmark
```

This has a 10-30% performance cost due to deterministic algorithm selection.
It is an explicit framework-global setting for model code, not a requirement
for Opaque's keyed sampling. Call it once at training startup if user-model
reproducibility needs it.

## Checkpoint and resume

Because keys are values, noise state contains the base key and cursor needed
to continue. Save that state through Opaque's serializer and restore it into a
fresh state template built with the same mechanism configuration:

```python
from opaque.serialization import from_state_dict, state_dict

# Save after any number of noise_fn calls.
checkpoint = state_dict(noise_state)
torch.save(checkpoint, "checkpoint.pt")

# Resume. The template's key is replaced by the saved key.
noise_fn, template_state = gaussian_noise(noise_multiplier=1.1, key=key(0))
noise_state = from_state_dict(template_state, torch.load("checkpoint.pt"))
```

The next noise result matches an uninterrupted run. Constructing a new noise
function from `fold_in(base_key, step)` is not a substitute: that starts a fresh
mechanism state at step zero. Restore samplers the same way, with
`from_state_dict(template_sampler, snapshot)` — the template supplies the
dataset, the snapshot the key and cursor.

## API reference

See [Random API Reference](../reference/rng.md) for complete function signatures
and return types.
