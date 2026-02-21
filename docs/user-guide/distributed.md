# Distributed Training with DDP

Opaque supports multi-GPU training via PyTorch
**DistributedDataParallel (DDP)**.  FSDP, TP, and PP are not supported.

## How DP-SGD Works in DDP

Each device runs the same training loop on a different data shard.  Three
things must happen in the right order:

1. **Clip** — each device computes per-example clipped gradients on its local batch.
2. **Aggregate** — an AllReduce SUM collects the clipped gradients from every device.
3. **Noise** — every device independently adds the *same* Gaussian noise (same key → same noise → models stay in sync).

```
Device 0:  clip(local_batch) ──┐
Device 1:  clip(local_batch) ──┼── AllReduce SUM ── + noise(key) ── update
Device 2:  clip(local_batch) ──┘
```

!!! danger "Noise must be added on **every** device with the **same** key"

    If devices generate different noise the models diverge.  Never add noise
    on rank 0 and broadcast.  Pass the same `key=key(42)` everywhere, or use
    `synchronized="auto"` (the default), which picks a shared seed
    automatically.

## Minimal Example

A complete single-step DP-SGD loop with DDP:

```python
import torch
import torch.distributed as dist
from opaque import clipped_grad, gaussian_noise, make_functional, PoissonSampler
import opaque.distributed as dist_utils
from opaque.random import key

# Distributed setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Model → functional form
model = MyModel().to(device)
fmodel, params = make_functional(model)

def loss_fn(params, x, y):
    return ((fmodel(params, x) - y) ** 2).sum()

# DP components
grad_fn, clip_state = clipped_grad(
    loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2),
)
noise_fn, noise_state = gaussian_noise(
    stddev=1.1 * clip_state.sensitivity(), key=key(42),
)

# Poisson sampler (auto-sharded in DDP)
sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(0))
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

# Training loop
for batch_x, batch_y in loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # 1. Clip (local)
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    # 2. Aggregate (AllReduce SUM)
    grads = dist_utils.sum_gradients(grads)

    # 3. Noise (identical on every device)
    noisy_grads, noise_state = noise_fn(grads, noise_state)

    # 4. Update
    params = {k: params[k] - 0.01 * noisy_grads[k] for k in params}

dist.destroy_process_group()
```

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py
```

## Noise Synchronization

The `synchronized` parameter on all noise constructors (`gaussian_noise`,
`band_mf_noise`, etc.) controls whether devices generate identical or
independent noise:

| Value | Behaviour |
|-------|-----------|
| `"auto"` (default) | Synchronized if `torch.distributed` is initialized, independent otherwise |
| `True` | Force synchronized — all ranks use the same key |
| `False` | Independent — key is folded with rank via `fold_in(key, rank)` |

In the common centralized DP-SGD pattern you want synchronized noise so
that all devices stay in sync after each update.  The default `"auto"`
handles this automatically.

## Adaptive Clipping

`adaptive_clipped_grad` computes per-example clipped gradients and tracks
a per-step clipping rate.  The core function is **local-only** — it does
not perform any communication.  To keep the clip norm consistent across
ranks you must explicitly synchronize the state after each step:

```python
from opaque import adaptive_clipped_grad
from opaque.clipping import sync_adaptive_clip_state
from opaque.random import key

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    batch_argnums=(1, 2),
    initial_clip_norm=1.0,
    key=key(7),
)

# In the training loop:
grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
clip_state = sync_adaptive_clip_state(clip_state)   # ← REQUIRED in DDP
grads = dist_utils.sum_gradients(grads)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

`sync_adaptive_clip_state` aggregates `num_clipped` and `total` across
ranks (sum), recomputes the global clipping rate, adds quantile noise,
and updates `clip_norm`.  After the call, `clip_state.clip_norm` is
identical on every device.

For fixed clipping (`clipped_grad`), the state is already deterministic
and does not need synchronization.  You can optionally validate with
`sync_clip_state(state)`, which asserts that `l2_norm_bound` matches
across ranks and raises `RuntimeError` if it doesn't.

## Poisson Sampling

`PoissonSampler` auto-detects DDP and switches to **SHARDED** mode:
each rank samples from a disjoint partition of the dataset.
The key is automatically diversified per rank via `fold_in(key, rank)`.

```python
from opaque import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))
loader = DataLoader(dataset, batch_sampler=sampler)
```

No manual rank handling is needed.

## Privacy Accounting

Privacy accounting is the same as single-device training.
The effective sample rate is the *global* sample rate across all devices:

```python
import opaque.accounting as acc

noise_multiplier = 1.1
sample_rate = batch_size_per_device * world_size / dataset_size

step = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)
training = step * num_steps
epsilon = training.epsilon_at(delta=1e-5)
```

## Matrix Factorization Noise

For correlated noise mechanisms (`band_mf_noise`, `blt_mf_noise`), each
device must generate the *same* correlated noise stream.  Pass the same
`key` and use `synchronized="auto"` (the default).  See
[Matrix Factorization](matrix-factorization.md) for details.

## API Reference

### `opaque.distributed`

Core distributed primitives.  All functions are no-ops (or return input
unchanged) when `torch.distributed` is not initialized, so the same
training code works on a single device without changes.

| Function | Purpose |
|----------|---------|
| `is_distributed()` | `True` if `torch.distributed` is initialized |
| `get_rank()` | Current rank (0 if not distributed) |
| `get_world_size()` | Number of devices (1 if not distributed) |
| `sum_gradients(grads)` | AllReduce **SUM** on a PyTree of tensors |
| `reduce_pytree(pytree, op)` | AllReduce on a PyTree (op: `"sum"`, `"mean"`, `"max"`, `"min"`) |
| `reduce_scalar(value, op)` | Reduce a Python float across ranks (default op: `"mean"`) |
| `all_reduce(tensor, op)` | In-place AllReduce on a single tensor |
| `gather_tensors(tensor, dim)` | Gather variable-size tensors from all ranks and concatenate |
| `gather_pytree(pytree)` | Gather + concatenate tensor leaves of a PyTree |
| `sync_state(state, field_ops)` | Synchronize scalar fields of a dataclass by per-field reduction |
| `assert_scalar_equal(v, name)` | Raise `RuntimeError` if a scalar differs across ranks |
| `barrier()` | Blocking barrier across all ranks |

### `opaque.clipping.distributed`

Clipping-specific sync helpers that understand `FixedClipState` and
`AdaptiveClipState`.

| Function | Purpose |
|----------|---------|
| `sync_clip_state(state)` | Assert `FixedClipState.l2_norm_bound` matches across ranks |
| `sync_adaptive_clip_state(state)` | Aggregate counts, recompute global clipping rate, update `clip_norm` |
| `sync_adaptive_clipped_grad_aux(aux)` | Gather auxiliary tensors (`grad_norms`, `loss_values`, etc.) |

### `opaque.noise.distributed`

Noise-specific validation helpers (called automatically by noise
functions when `synchronized=True`).

| Function | Purpose |
|----------|---------|
| `sync_gaussian_noise_state(state)` | Assert seed and step counter match across ranks |
| `sync_mf_noise_state(state)` | Assert seed and step counter match for MF noise |

See [API Reference](../api/distributed.md) for full docstrings.

## Limitations

- **DDP only** — FSDP, Tensor Parallel, and Pipeline Parallel are not supported.
- **Single-node only** — Multi-node DDP should work but is not extensively tested.
- **NCCL recommended** — Other backends (Gloo, MPI) are not tested.

## See Also

- [Distributed example](https://github.com/JetBrains-Research/opaque/blob/main/examples/distributed_dp_training.py)
- [Parallelism Compatibility](../development/parallelism_compatibility.md)
- [Known Limitations](../limitations.md)
