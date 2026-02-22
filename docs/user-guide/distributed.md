# Distributed Training

Opaque supports multi-GPU training via PyTorch DistributedDataParallel
(DDP). FSDP, Tensor Parallel, and Pipeline Parallel are not supported.

## How DP-SGD works with DDP

Each device runs the same training loop on a different data shard. Three
operations happen in sequence:

1. **Clip** -- each device computes per-example clipped gradients on its
   local batch.
2. **Aggregate** -- an AllReduce SUM collects the clipped gradient sums
   from every device.
3. **Noise** -- every device independently adds the *same* Gaussian noise.
   Because the noise key is identical on all ranks, the noise is identical,
   and models stay in sync.

```
Device 0:  clip(local_batch) --+
Device 1:  clip(local_batch) --+-- AllReduce SUM -- + noise(key) -- update
Device 2:  clip(local_batch) --+
```

There are two valid approaches to noise in distributed DP-SGD:

- **Independent generation (recommended):** Every rank generates the same
  noise using the same key. No communication is needed — the noise is
  identical because the RNG state is identical. This is simpler and avoids
  an extra broadcast.
- **Rank-0 broadcast:** Rank 0 generates the noise and broadcasts it to
  all other ranks. This is correct but adds a communication step and
  requires special-casing rank 0.

## Minimal example

```python
import torch
import torch.distributed as dist
import torchopt
from opaque import clipped_grad, gaussian_noise, make_functional, PoissonSampler
import opaque.distributed as dist_utils
from opaque.random import key, fold_in
from opaque.sampling.distributed import local_shard

# Distributed setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Model -> functional form
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

# Poisson sampler (shard dataset)
shard = local_shard(dataset, rank=rank, world_size=dist.get_world_size())
sampler = PoissonSampler(
    shard, sample_rate=0.01, key=fold_in(key(0), rank),
)
loader = torch.utils.data.DataLoader(shard, batch_sampler=sampler)

# Optimizer
optimizer = torchopt.sgd(lr=0.01)
opt_state = optimizer.init(params)

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
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = torchopt.apply_updates(params, updates)

dist.destroy_process_group()
```

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py
```

## Noise synchronization

Noise constructors take a `key` parameter. Pass the **same key** on every
rank to get identical noise (centralized DP-SGD). Use `fold_in(key, rank)`
to get independent per-rank noise streams when needed:

```python
from opaque.random import key, fold_in

# Synchronized noise — same key on all ranks
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(42))

# Independent noise — different key per rank
noise_fn, noise_state = gaussian_noise(stddev=1.1, key=fold_in(key(42), rank))
```

In the common centralized DP-SGD pattern, pass the same `key(seed)` on
every rank so that all devices produce identical noise and models stay
in sync.

## Gradient aggregation

`sum_gradients` performs an AllReduce SUM on a PyTree of tensors:

```python
import opaque.distributed as dist_utils

grads = dist_utils.sum_gradients(grads)
```

This sums the clipped gradient contributions from all devices. After this
call, every rank holds the same total clipped gradient sum.

For more general reductions, use `reduce_pytree`:

```python
grads = dist_utils.reduce_pytree(grads, op="mean")
```

## Adaptive clipping

`adaptive_clipped_grad` is local-only -- it does not communicate. To keep
the clip norm consistent across ranks, explicitly synchronize the state
after each step:

```python
from opaque import adaptive_clipped_grad
from opaque.distributed import sync
from opaque.random import key

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    batch_argnums=(1, 2),
    initial_clip_norm=1.0,
    key=key(7),
)

# In the training loop:
grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
clip_state = sync(clip_state)   # required in DDP
grads = dist_utils.sum_gradients(grads)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

`sync()` auto-dispatches based on the type of the state object. For
`AdaptiveClipState`, it aggregates `num_clipped` and `total` across ranks
(sum), recomputes the global clipping rate, and updates `clip_norm`. After
the call, `clip_state.clip_norm` is identical on every device.

For fixed clipping (`clipped_grad`), the state is deterministic and does
not need synchronization. You can optionally validate with
`sync(clip_state)`, which asserts that `l2_norm_bound` matches across
ranks and raises `RuntimeError` if it does not.

## Poisson sampling

Shard the dataset using `local_shard()`, then create a `PoissonSampler`
on the shard. Derive a per-rank key via `fold_in(key, rank)`.

```python
import torch.distributed as dist
from opaque import PoissonSampler
from opaque.random import key, fold_in
from opaque.sampling.distributed import local_shard

rank = dist.get_rank()
world_size = dist.get_world_size()

shard = local_shard(dataset, rank=rank, world_size=world_size)
sampler = PoissonSampler(
    shard,
    sample_rate=0.01,
    key=fold_in(key(42), rank),
)
loader = DataLoader(shard, batch_sampler=sampler)
```

## Privacy accounting

Privacy accounting is the same as single-device training. The effective
sample rate is the global sample rate across all devices:

```python
import opaque.accounting as acc

global_sample_rate = batch_size_per_device * world_size / dataset_size
step = acc.poisson(acc.gaussian(noise_multiplier), global_sample_rate)
training = step * num_steps
epsilon = training.epsilon_at(delta=1e-5)
```

## Optimizer state synchronization

For correlated noise mechanisms (`band_mf_noise`, `blt_mf_noise`), each
device must generate the same correlated noise stream. Pass the same
`key=key(seed)` on every rank.

Functional optimizers (TorchOpt) stay synchronized automatically because:

1. After `sum_gradients`, all ranks hold the same gradient sum.
2. After adding synchronized noise, all ranks hold the same noisy gradient.
3. `optimizer.update` is a pure function: same inputs produce same outputs.

So optimizer states evolve identically on all devices without explicit
synchronization.

## `opaque.distributed` API summary

All functions are no-ops (or return input unchanged) when
`torch.distributed` is not initialized, so the same training code works on
a single device without changes.

| Function | Purpose |
|----------|---------|
| `is_distributed()` | `True` if `torch.distributed` is initialized |
| `get_rank()` | Current rank (0 if not distributed) |
| `get_world_size()` | Number of devices (1 if not distributed) |
| `sum_gradients(grads)` | AllReduce SUM on a PyTree of tensors |
| `reduce_pytree(pytree, op)` | AllReduce on a PyTree (op: `"sum"`, `"mean"`, `"max"`, `"min"`, `"product"`) |
| `reduce_scalar(value, op)` | Reduce a Python float across ranks |
| `all_reduce(tensor, op)` | In-place AllReduce on a single tensor |
| `gather_tensors(tensor, dim)` | Gather variable-size tensors from all ranks and concatenate |
| `gather_pytree(pytree)` | Gather and concatenate tensor leaves of a PyTree |
| `assert_pytree_equal(pytree, name)` | Assert a PyTree is identical across ranks (fingerprint check) |
| `sync(state)` | Dispatch to the right sync function for any state/aux type |
| `sync_object(state, field_ops)` | Synchronize scalar fields of a dataclass across ranks |
| `assert_scalar_equal(v, name)` | Raise `RuntimeError` if a scalar differs across ranks |
| `barrier()` | Blocking barrier across all ranks |

### Type-specific sync behavior

`sync()` auto-dispatches based on the type of the object passed. The
following types are registered:

| Type | Behavior |
|------|----------|
| `FixedClipState` | Assert `l2_norm_bound` matches across ranks |
| `AdaptiveClipState` | Aggregate counts, recompute global clipping rate, update `clip_norm` |
| `ClippedFunAux`, `ClippedGradAux`, `AdaptiveClippedGradAux` | Gather aux tensors across ranks |
| `GaussianNoiseState` | Assert seed and step counter match across ranks |
| `MFNoiseState` | Assert seed and step counter match for MF noise |

!!! note "Bounded and rectified noise"
    `rectified_gaussian_noise` and `bounded_gaussian_noise` both return
    `GaussianNoiseState`, so `sync()` handles them automatically. Explicit
    helpers `sync_rectified_noise_state()` and `sync_bounded_noise_state()`
    are available in `opaque.noise.distributed` for readability.

See [API Reference](../api/distributed.md) for full docstrings.

## Limitations

- **DDP only.** FSDP, Tensor Parallel, and Pipeline Parallel are not
  supported.
- **NCCL recommended.** Other backends (Gloo, MPI) are not tested.
- **Single-node primarily.** Multi-node DDP should work but is not
  extensively tested.

## API reference

See [Distributed API Reference](../api/distributed.md) for complete
function signatures and return types.
