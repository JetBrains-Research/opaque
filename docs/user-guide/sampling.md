# Sampling & Microbatching

Sampling determines how training examples are selected for each step.
In DP-SGD, the sampling mechanism directly affects the privacy guarantee:
Poisson subsampling provides privacy amplification, meaning you need less
noise for the same epsilon when each example is included independently with
small probability.

Opaque provides three sampler classes, all designed to work with PyTorch's
`DataLoader` via the `batch_sampler` parameter.

## Poisson sampling

### Why it matters for privacy

If each example is independently included with probability `q`, the privacy
cost of one step is approximately `q` times the cost without subsampling.
This is the *privacy amplification by subsampling* effect. The smaller the
sample rate, the stronger the amplification:

```python
import opaque.accounting as acc

# Without subsampling: full dataset
full = acc.gaussian(1.0) * 1000
print(full.epsilon_at(1e-5))  # large

# With Poisson subsampling: sample_rate = 0.01
subsampled = acc.poisson(acc.gaussian(1.0), sample_rate=0.01) * 1000
print(subsampled.epsilon_at(1e-5))  # much smaller
```

The sample rate is typically `batch_size / dataset_size`:

```python
dataset_size = 50_000
batch_size = 256
sample_rate = batch_size / dataset_size  # 0.00512
```

### `PoissonSampler`

The standard sampler. Each example is included independently with probability
`sample_rate`, producing variable-size batches.

```python
from opaque.sampling import PoissonSampler
from opaque.random import key
import torch.utils.data as data

dataset = data.TensorDataset(X, y)
sampler = PoissonSampler(
    dataset,
    sample_rate=0.01,
    num_iterations=10,
    key=key(42),
)
loader = data.DataLoader(dataset, batch_sampler=sampler)

for batch in loader:
    # batch size varies around dataset_size * sample_rate
    ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_source` | any with `__len__` | The dataset |
| `sample_rate` | `float` in (0, 1] | Inclusion probability per example |
| `num_iterations` | `int` or `None` | Number of batches to yield (default: `None` = infinite) |
| `key` | `RngKey` | RNG key for reproducibility |

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `expected_batch_size` | `float` | `len(data_source) * sample_rate` |
| `batch_size_variance` | `float` | `len(data_source) * sample_rate * (1 - sample_rate)` |

Batch sizes follow a Binomial distribution. For large datasets and small
sample rates, the standard deviation is roughly `sqrt(expected_batch_size)`.

### `TruncatedPoissonSampler`

Poisson sampling with an upper bound on batch size. When a Poisson sample
exceeds `max_batch_size`, it is randomly subsampled down. This gives
tighter privacy bounds than standard Poisson (up to 20% improvement in
epsilon) while preventing memory spikes from unusually large batches.

```python
from opaque.sampling import TruncatedPoissonSampler
from opaque.random import key

sampler = TruncatedPoissonSampler(
    dataset,
    sample_rate=batch_size / dataset_size,
    max_batch_size=batch_size,
    num_iterations=num_steps,
    key=key(42),
)
loader = data.DataLoader(dataset, batch_sampler=sampler)
```

**Additional parameter:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_batch_size` | `int` | Upper bound on batch size |

Privacy accounting uses `acc.truncated_poisson` to match:

```python
step = acc.truncated_poisson(
    acc.gaussian(noise_multiplier),
    sample_rate=batch_size / dataset_size,
    batch_size_cap=batch_size,
    dataset_size=dataset_size,
)
training = step * num_steps
```

### `CyclicPoissonSampler`

Partitions the dataset into `cycle_length` groups and cycles through them,
sampling from each group with probability `sampling_prob`. This sampler
is required for matrix-factorization correlated noise mechanisms
(BandMF via `mf_noise`) which need a fixed participation pattern.

```python
from opaque.sampling import CyclicPoissonSampler, PartitionType
from opaque.random import key

sampler = CyclicPoissonSampler(
    dataset,
    sampling_prob=0.5,
    cycle_length=5,
    iterations=500,
    partition_type=PartitionType.EQUAL_SPLIT,
    key=key(42),
)
loader = data.DataLoader(dataset, batch_sampler=sampler)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | any with `__len__` | required | The dataset |
| `sampling_prob` | `float` in (0, 1] | required | Inclusion probability within each group |
| `cycle_length` | `int` | 1 | Number of groups (1 = standard Poisson) |
| `iterations` | `int` or `None` | `None` (= 1) | Total batches to yield |
| `truncated_batch_size` | `int` or `None` | `None` | Optional upper bound on batch size |
| `partition_type` | `PartitionType` | `EQUAL_SPLIT` | How examples are assigned to groups |
| `key` | `RngKey` | required | RNG key |

**Partition types:**

- `PartitionType.EQUAL_SPLIT` -- shuffle the dataset, then split into groups
  of equal size. Deterministic group sizes.
- `PartitionType.INDEPENDENT` -- assign each example to a random group
  (multinomial). Group sizes vary.

At iteration `i`, the sampler draws from group `i % cycle_length`. With
`cycle_length=1` it reduces to standard Poisson sampling.

See [Noise Addition](noise.md#matrix-factorization-noise-dp-ftrl) for how cyclic
sampling integrates with correlated noise mechanisms.

## DataLoader integration

All three samplers are PyTorch `Sampler` subclasses that yield lists of
indices (i.e., they are batch samplers). Pass them to `DataLoader` via the
`batch_sampler` parameter:

```python
loader = data.DataLoader(dataset, batch_sampler=sampler)
```

Do not pass `batch_size`, `shuffle`, or `sampler` when using
`batch_sampler` -- PyTorch does not allow combining these parameters.

Because Poisson sampling produces variable-size batches, your training code
should handle batches of different sizes. In practice this is rarely a
problem since `clipped_grad` and `gaussian_noise` work with any batch size.

## Distributed sampling

In distributed training, shard the dataset explicitly using `local_shard()`
and derive a per-rank key via `fold_in(key, rank)`:

1. The dataset is partitioned across ranks. Rank `r` owns indices
   `[r * shard_size, (r+1) * shard_size)`, with the last rank receiving
   any remainder.
2. The RNG key is diversified per rank via `fold_in(key, rank)`, so
   different ranks sample different subsets of their shard.
3. No communication is needed -- each rank samples independently from its
   own partition.

```python
from opaque.sampling.distributed import local_shard
from opaque.random import key, fold_in
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

shard = local_shard(dataset, rank=rank, world_size=world_size)
sampler = PoissonSampler(shard, sample_rate=0.01, key=fold_in(key(42), rank))
loader = data.DataLoader(shard, batch_sampler=sampler)
```

**Privacy accounting in distributed mode** uses the global sample rate:

```python
global_sample_rate = batch_size_per_device * world_size / dataset_size
step = acc.poisson(acc.gaussian(noise_multiplier), global_sample_rate)
```

### Distributed helpers

The `opaque.sampling.distributed` submodule provides two utilities used
internally by the samplers:

| Function | Description |
|----------|-------------|
| `local_shard_bounds(dataset_size)` | Returns `(start, end)` index range for the current rank |
| `rank_key(key)` | Returns `fold_in(key, rank)` for rank > 0, unchanged for rank 0 |

These are available for advanced use cases but most users do not need them
directly.

## Microbatching

Microbatching is a memory optimization, not a sampling strategy. When
per-example gradient computation via `vmap` exceeds GPU memory,
microbatching processes the batch in smaller chunks and accumulates the
clipped gradient sums. The result is mathematically identical to processing
the full batch.

```python
from opaque import clipped_grad

grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    batch_argnums=1,
    microbatch_size=16,  # process 16 examples at a time
)
```

Each microbatch of 16 examples is vmapped, per-example gradients are
clipped, and the clipped gradients are summed. The partial sums are
accumulated in-place, so peak memory is proportional to
`microbatch_size * model_parameters` rather than
`batch_size * model_parameters`.

### Choosing microbatch size

Use `TrainingProfiler` to compare a few candidate microbatch sizes and select
the largest stable value for your device:

```python
from opaque.profiling import StepTimer, TrainingProfiler, reset_peak_memory

profiler = TrainingProfiler(device)
for optimal in [64, 32, 16, 8, 4, 2, 1]:
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        clipping_norm=1.0,
        batch_argnums=(1, 2),
        microbatch_size=optimal,
    )

    reset_peak_memory(device)
    timer = StepTimer(device, batch_size=batch_size)
    with timer:
        grads, aux = grad_fn(params, batch_x, batch_y, state=clip_state)
    profiler = profiler.add_step(timer)

    print(optimal, profiler.current_metrics()["memory_peak_gb"])

grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    batch_argnums=(1, 2),
    microbatch_size=optimal,
)
```

See [Memory Optimizations](memory-optimizations.md) for more details on memory
analysis tools.

### Privacy equivalence

Microbatching does not change the privacy guarantee. The clipped gradient
sum is identical whether the batch is processed in one shot or in chunks:

```python
# These produce the same result:
# Full batch: vmap over 256 examples, clip, sum
grads, state = grad_fn(params, batch_256, state=state)

# Microbatched: vmap over 16 at a time, clip each, sum partials
grad_fn_mb, state_mb = clipped_grad(loss_fn, clipping_norm=1.0,
                                     batch_argnums=1, microbatch_size=16)
grads_mb, state_mb = grad_fn_mb(params, batch_256, state=state_mb)
# grads == grads_mb
```

## Choosing a sampler

| Sampler | Batch size | Privacy | Use case |
|---------|-----------|---------|----------|
| `PoissonSampler` | Variable | Standard amplification | Research, general use |
| `TruncatedPoissonSampler` | Bounded above | Tighter (up to 20%) | Production, memory-constrained |
| `CyclicPoissonSampler` | Cyclic groups | Depends on mechanism | Matrix-factorization noise |

For most DP-SGD workloads, `PoissonSampler` is sufficient.
`TruncatedPoissonSampler` is a reasonable upgrade when you want tighter
privacy bounds or need predictable batch sizes. `CyclicPoissonSampler` is
only needed with correlated noise mechanisms.

## API reference

See [Sampling API Reference](../api/sampling.md) for complete function
signatures and return types.
