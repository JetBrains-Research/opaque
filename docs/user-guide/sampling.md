# Sampling & Microbatching

Sampling determines how training examples are selected for each step.
In DP-SGD, the sampling mechanism directly affects the privacy guarantee:
Poisson subsampling provides privacy amplification, meaning you need less
noise for the same epsilon when each example is included independently with
small probability.

Opaque provides sampler classes designed to work with PyTorch's
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
full = dpsgd_acc.gaussian(1.0) * 1000
print(full.epsilon_at(1e-5))  # large

# With Poisson subsampling: sample_rate = 0.01
subsampled = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), sample_rate=0.01) * 1000
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
from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import key
import torch.utils.data as data

dataset = data.TensorDataset(X, y)
sampler = PoissonSampler(
    dataset,
    sample_rate=0.01,
    n_steps=10,
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
| `n_steps` | `int` or `None` | Number of batches to yield (default: `None` = infinite) |
| `key` | `RngKey` | RNG key for reproducibility |

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `expected_batch_size` | `float` | `len(data_source) * sample_rate` |
| `batch_size_variance` | `float` | `len(data_source) * sample_rate * (1 - sample_rate)` |

Batch sizes follow a Binomial distribution. For large datasets and small
sample rates, the standard deviation is roughly `sqrt(expected_batch_size)`.

### Truncated Poisson (batch cap)

Use :class:`~opaque.dpsgd.sampling.PoissonSampler` with
``truncated_batch_size`` set. When a Poisson draw exceeds that cap, a
uniform random subset of the selected indices is kept. That caps batch
size (more stable training and memory) but **weakens** privacy relative to
plain Poisson at the same ``sample_rate``—account with
``dpsgd_acc.poisson(..., truncated_batch_size=..., dataset_size=...)`` so the
PLD matches the cap.

```python
from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(
    dataset,
    sample_rate=batch_size / dataset_size,
    truncated_batch_size=batch_size,
    n_steps=num_steps,
    key=key(42),
)
loader = data.DataLoader(dataset, batch_sampler=sampler)
```

**Additional parameter:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `truncated_batch_size` | `int` | Upper bound on batch size |

Privacy accounting (truncated form of ``poisson``):

```python
step = dpsgd_acc.poisson(
    dpsgd_acc.gaussian(noise_multiplier),
    batch_size / dataset_size,
    truncated_batch_size=batch_size,
    dataset_size=dataset_size,
)
training = step * num_steps
```

### `CyclicPoissonSampler` (DP-FTRL)

The dataset is split into ``bands`` disjoint groups (``partition_type`` fixes
how, when ``bands > 1``).  Step ``i`` draws only from group ``i % bands``: each
example in that group is kept independently with probability ``sample_rate``,
so the within-step batch size is Binomial.  Training steps therefore advance a
fixed rotation over which group is active, while inclusion inside the active
group stays Poisson-style.

For an identity MF baseline (``identity_strategy`` / ``identity_mf``), use
``bands=1`` so the lone group is the full dataset and every step is plain
Poisson on all examples, matching whole-process ``dpftrl_acc.poisson`` with an
``IdentityMf`` inner.  For BandMF, set ``bands`` to the same count as in
``band_mf_strategy`` / ``BandMf`` so participation matches correlated
``mf_gaussian_noise``.

```python
from opaque.dpftrl.sampling import CyclicPoissonSampler
from opaque.dpftrl.sampling.types import PartitionType
from opaque.random import key

sampler = CyclicPoissonSampler(
    dataset,
    sample_rate=0.5,
    bands=5,
    n_steps=500,
    partition_type=PartitionType.EQUAL_SPLIT,
    key=key(42),
)
loader = data.DataLoader(dataset, batch_sampler=sampler)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | any with `__len__` | required | The dataset |
| `sample_rate` | `float` in (0, 1] | required | Inclusion probability within each active group |
| `bands` | `int` | 1 | Number of groups in the cycle (``1`` = full dataset each step; match strategy for BandMF). |
| `n_steps` | `int` | 1 | Total batches to yield |
| `partition_type` | `PartitionType` | `EQUAL_SPLIT` | How examples are assigned to groups |
| `key` | `RngKey` | required | RNG key |

**Partition types:**

- `PartitionType.EQUAL_SPLIT` -- shuffle the dataset, then split into groups
  of equal size. Deterministic group sizes.
- `PartitionType.INDEPENDENT` -- assign each example to a random group
  (multinomial). Group sizes vary.

See [Noise Addition](noise.md#matrix-factorization-noise-dp-ftrl) for how this
sampler pairs with matrix-factorization noise.

### `BMinSepSampler`

Warm-start **b-min-sep** subsampling for BandMF (Dong & Ganesh, arXiv:2602.09338).
Each step includes each *eligible* example independently with probability `p`
(the paper’s $p$). Eligibility excludes any example that appeared in one of the
previous `bands - 1` batches. Initial per-example cooldowns are drawn from the
stationary distribution so expected batch size is roughly stable from step 0.

Use `p = p_0 / (1 - p_0 * (bands - 1))` when matching a target per-example rate
`p_0 = expected_batch_size / dataset_size` (for `bands == 1`, `p = p_0`).
Pair with `opaque.accounting.b_min_sep` for privacy accounting.

```python
from opaque.dpftrl.sampling import BMinSepSampler
from opaque.random import key

p0 = batch_size / len(dataset)
bands = 8
p = p0 / (1.0 - p0 * (bands - 1))
sampler = BMinSepSampler(
    dataset,
    bands=bands,
    sampling_prob=p,
    iterations=num_steps,
    key=key(42),
)
```

## Balls-in-Bins sampling

### `BallsInBinsSampler`

Each example is independently assigned to one of `num_bins` bins (Binomial
bin sizes; some bins may be empty). The assignment is **fixed once at init**
and **reused across all epochs** — this is required by the dominating-pair
BnB privacy accounting. Used with DP-λCGD, BISR, BSR, and BLT mechanisms,
**and with the plain Gaussian mechanism for DP-SGD**: `acc.balls_in_bins`
accepts both Gaussian and MF inners (see [Mechanisms — Cross-cutting
amplification](../mechanisms/index.md)).

```python
from opaque.dpftrl.sampling import BallsInBinsSampler
from opaque.random import key

sampler = BallsInBinsSampler(
    dataset,
    num_bins=dataset_size // batch_size,
    num_epochs=8,
    key=key(42),
)
loader = data.DataLoader(dataset, batch_sampler=sampler)
```

`acc.balls_in_bins(mechanism, num_bins, num_epochs)` returns the
**total** multi-epoch privacy cost — do not compose further with
`* num_epochs`.  In a training loop, book the cost once before training
begins; the per-step accumulator does not compose for BnB.

`BallsInBinsSampler` is incompatible with parallel-Poisson DDP
(`--no-shard`): each rank must work a disjoint shard so every example
participates exactly once per epoch globally.  Pass `--shard` (or its
DDP equivalent) when using BnB across multiple ranks.

## Sequential batch sampling

### `SequentialBatchSampler`

Iterates through the dataset in fixed-size contiguous batches with no
randomness. The last incomplete batch is dropped. This is the only
sampler that does not require an RNG key — it is fully deterministic.
Pre-shuffle the dataset once before constructing the sampler.

Used by the BLT mechanism, which requires deterministic batch order with
fixed separation between participations.

```python
from opaque.dpftrl.sampling import SequentialBatchSampler
import torch.utils.data as data

sampler = SequentialBatchSampler(
    dataset,
    batch_size=256,
)
loader = data.DataLoader(dataset, batch_sampler=sampler)

for batch in loader:
    # batch size is always exactly 256
    ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_source` | any with `__len__` | The dataset (must not be empty) |
| `batch_size` | `int` (≥ 1) | Exact number of examples per batch |

**Properties:**

| Property | Returns | Description |
|----------|---------|-------------|
| `expected_batch_size` | `float` | Always equals `batch_size` (exact, not statistical) |

## DataLoader integration

All samplers are PyTorch `Sampler` subclasses that yield lists of
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
from opaque.distributed import local_shard
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
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(noise_multiplier), global_sample_rate)
```

### Distributed helpers

The `opaque.distributed` submodule provides two utilities used
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
from opaque.dpsgd.clipping import clipped_grad

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
from opaque.profiling import reset_peak_memory
from opaque.profiling import StepTimer, TrainingProfiler

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
| `PoissonSampler` + ``truncated_batch_size`` | Bounded above | Weaker than plain Poisson (same ``sample_rate``) | Production, stable batch sizes / memory |
| `CyclicPoissonSampler` (``opaque.dpftrl``) | Variable | ``dpftrl_acc.poisson`` | DP-FTRL; identity MF → ``bands=1``; BandMF → ``bands`` = strategy |
| `BallsInBinsSampler` | Fixed (deterministic) | Balls-in-bins amplification | λCGD, BISR, BLT |
| `SequentialBatchSampler` | Fixed (deterministic) | No amplification | BLT (pre-shuffled dataset) |

For most DP-SGD workloads, `PoissonSampler` is sufficient.
Use ``truncated_batch_size`` when you need **capped** batch sizes; expect
**worse** privacy than plain Poisson at the same ``sample_rate`` unless you
recalibrate noise. For DP-FTRL, use
``opaque.dpftrl.sampling.CyclicPoissonSampler`` as in the section above.
`BallsInBinsSampler` and
`SequentialBatchSampler` are used with matrix-factorization mechanisms
that require fixed batch sizes.

## API reference

See [Sampling API Reference](../reference/sampling.md) for complete function
signatures and return types.
