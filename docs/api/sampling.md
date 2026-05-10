# Sampling

Sampling primitives live in `opaque.dpsgd.sampling` (Poisson) and
`opaque.dpftrl.sampling` (Poisson with optional ``bands``, b-min-sep, balls-in-bins, sequential).
Distributed shard helpers live in `opaque.distributed`. They provide
privacy-amplifying sampling mechanisms for DP-SGD and DP-FTRL.

## Overview

Privacy amplification through sampling is a key technique in DP-SGD: training
on randomly selected subsets provides stronger privacy than training on the
full dataset.

Opaque provides these sampling strategies:

1. **Poisson Sampling — DP-SGD** (`opaque.dpsgd.sampling.PoissonSubsampler`):
   each example is sampled independently with probability `sample_rate`.
   Optional `truncated_batch_size` caps per-step batch size for more stable
   batches and memory; accounting must use the truncated-Poisson PLD (weaker
   than plain Poisson at the same `sample_rate`).

2. **Poisson Sampling — DP-FTRL** (`opaque.dpftrl.sampling.PoissonSampler`):
   iteration ``i`` draws from group ``i % bands`` with probability
   ``sample_rate``. ``bands=1`` is plain Poisson on the full dataset; larger
   ``bands`` give cyclic participation for correlated matrix-factorization
   noise (e.g. BandMF).

3. **Balls-in-Bins Sampling** (`BallsInBinsSampler`): each example is
   independently assigned to a bin once at init; the assignment is **fixed
   across epochs** (required for BnB accounting). Bin sizes are variable
   (Binomial); some bins may be empty. Used with DP-λCGD, BISR, BSR, and
   BLT mechanisms.

4. **Sequential Batch Sampling** (`SequentialBatchSampler`): iterates
   through the dataset in fixed-size contiguous batches with no randomness.
   The dataset should be pre-shuffled once before constructing the sampler.
   Used with the BLT mechanism.

5. **b-min-sep** (`BMinSepSampler`): warm-start minimum-separation Poisson
   subsampling for BandMF (arXiv:2602.09338). Use with `ftrl_acc.b_min_sep`.

**See also**: [Sampling & Microbatching User Guide](../user-guide/sampling.md)

## PoissonSubsampler (DP-SGD)

```python
from opaque.dpsgd.sampling import PoissonSubsampler
from opaque.random import key

sampler = PoissonSubsampler(
    data_source,
    sample_rate=batch_size / len(data_source),
    n_steps=None,
    truncated_batch_size=None,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sample_rate` | `float` | required | Probability of including each example, in (0, 1] |
| `n_steps` | `int` or `None` | `None` | Number of batches to yield. `None` = infinite |
| `truncated_batch_size` | `int` or `None` | `None` | Optional per-step batch-size cap. When set, the sampler emits batches truncated to this many examples (uniform random subset of the Poisson draw). |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

Plain Poisson — account with
`dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate)`.

Truncated Poisson (when `truncated_batch_size` is set) — account with
`dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate,
truncated_batch_size=batch, dataset_size=n)` to use the matching
truncated-Poisson PLD.

## BallsInBinsSampler

```python
from opaque.dpftrl.sampling import BallsInBinsSampler
from opaque.random import key

sampler = BallsInBinsSampler(
    data_source,
    num_bins=dataset_size // batch_size,
    n_steps=8 * (dataset_size // batch_size),
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `num_bins` | `int` | required | Number of bins per epoch (≥ 2). Typically `dataset_size / batch_size` |
| `n_steps` | `int` or `None` | `None` | Total number of batches to yield. Must be a positive multiple of `num_bins` (per-bin participation count is `n_steps // num_bins`). `None` = infinite |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

Bin sizes are variable (Binomial distribution). Assignments are **fixed
across epochs** (required for BnB dominating-pair accounting). Empty bins
are skipped.

Account with `ftrl_acc.balls_in_bins(mechanism, num_bins, n_steps)` where
`mechanism` is `ftrl_acc.lambda_cgd(...)`, `ftrl_acc.bisr(...)`,
`ftrl_acc.blt(...)`, or `ftrl_acc.mf_identity(...)`.

## PoissonSampler (DP-FTRL)

```python
from opaque.dpftrl.sampling import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(
    data_source,
    sample_rate=0.5,
    bands=4,
    n_steps=1000,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sample_rate` | `float` | required | Probability of including each eligible example, in (0, 1] |
| `bands` | `int` | `1` | Number of cyclic groups (band width). `1` collapses to plain Poisson |
| `n_steps` | `int` | `1` | Total batches to yield |
| `partition_type` | `PartitionType` | `EQUAL_SPLIT` | How to partition: `EQUAL_SPLIT` (only used when `bands > 1`) or `INDEPENDENT` |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

In distributed training, shard the dataset with `local_shard()` and pass
a per-rank key via `fold_in(key, rank)`. Best used with `mf_noise`
for correlated noise (DP-FTRL); account with
`ftrl_acc.poisson(mechanism, sample_rate, n_steps=...)`.  There is no
batch-size cap on this sampler; ``ftrl_acc.poisson`` matches uncapped
Poisson draws only.

## Distributed Helpers

### `local_shard`

Partition a dataset for DDP training. Returns a `Subset` containing the
contiguous shard for the given rank.

```python
from opaque.distributed import local_shard
import torch.distributed as dist

shard = local_shard(
    dataset,
    rank=dist.get_rank(),
    world_size=dist.get_world_size(),
)
sampler = PoissonSubsampler(shard, sample_rate=0.01, key=fold_in(key(42), rank))
loader = DataLoader(shard, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | dataset with `len()` | required | The full training dataset |
| `rank` | `int` | `0` | Current device rank |
| `world_size` | `int` | `1` | Total number of devices |

**Returns:** `torch.utils.data.Subset` containing the local shard.

::: opaque.distributed.local_shard
    options:
      show_source: true
      heading_level: 3

## API Documentation

::: opaque.dpsgd.sampling.PoissonSubsampler
    options:
      show_source: true
      heading_level: 3

::: opaque.dpftrl.sampling.PoissonSampler
    options:
      show_source: true
      heading_level: 3

::: opaque.dpftrl.sampling.BMinSepSampler
    options:
      show_source: true
      heading_level: 3

::: opaque.dpftrl.sampling.BallsInBinsSampler
    options:
      show_source: true
      heading_level: 3

## SequentialBatchSampler

```python
from opaque.dpftrl.sampling import SequentialBatchSampler

sampler = SequentialBatchSampler(
    data_source,
    batch_size=256,
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset (must not be empty) |
| `batch_size` | `int` | required | Exact number of examples per batch (≥ 1) |

Batch size is deterministic and fixed. The last incomplete batch is
dropped (like `drop_last=True`). This sampler has no RNG key — it is
fully deterministic. Pre-shuffle the dataset before constructing the
sampler.

Used with the BLT mechanism, which requires deterministic batch order
with fixed separation between participations.

::: opaque.dpftrl.sampling.SequentialBatchSampler
    options:
      show_source: true
      heading_level: 3
