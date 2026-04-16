# Sampling

The `opaque.sampling` module provides privacy-amplifying sampling mechanisms
for DP-SGD training.

## Overview

Privacy amplification through sampling is a key technique in DP-SGD: training
on randomly selected subsets provides stronger privacy than training on the
full dataset.

Opaque provides these sampling strategies:

1. **Poisson Sampling** (`PoissonSampler`): Each example sampled independently
   with probability `sample_rate`. Variable batch sizes, strong privacy
   amplification.

2. **Truncated Poisson Sampling** (`TruncatedPoissonSampler`): Poisson
   sampling with a maximum batch size cap. Predictable memory usage, tighter
   privacy bounds than fixed-batch sampling.

3. **Cyclic Poisson Sampling** (`CyclicPoissonSampler`): Partitions the
   dataset into groups and cycles through them. Designed for matrix-
   factorization noise mechanisms (BandMF) where predictable sampling
   structure enables correlated noise.

4. **Balls-in-Bins Sampling** (`BallsInBinsSampler`): Each example is
   independently assigned to a bin once at init; the assignment is **fixed
   across epochs** (required for BnB accounting). Bin sizes are variable
   (Binomial); some bins may be empty. Used with DP-λCGD, BISR, BSR, and
   BLT mechanisms.

5. **Sequential Batch Sampling** (`SequentialBatchSampler`): Iterates
   through the dataset in fixed-size contiguous batches with no randomness.
   The dataset should be pre-shuffled once before constructing the sampler.
   Used with the BLT mechanism.

6. **b-min-sep** (`BMinSepSampler`): Warm-start minimum-separation Poisson
   subsampling for BandMF (arXiv:2602.09338). Use with `acc.b_min_sep`.

**See also**: [Sampling & Microbatching User Guide](../user-guide/sampling.md)

## PoissonSampler

```python
from opaque import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(
    data_source,
    sample_rate=batch_size / len(data_source),
    num_iterations=None,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sample_rate` | `float` | required | Probability of including each example, in (0, 1] |
| `num_iterations` | `int` or `None` | `None` | Number of batches to yield. `None` = infinite |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

Account with `acc.poisson(acc.gaussian(nm), sample_rate)`.

## TruncatedPoissonSampler

```python
from opaque import TruncatedPoissonSampler
from opaque.random import key

sampler = TruncatedPoissonSampler(
    data_source,
    sample_rate=batch_size / len(data_source),
    max_batch_size=max_batch,
    num_iterations=None,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sample_rate` | `float` | required | Expected sampling rate, in (0, 1] |
| `max_batch_size` | `int` | required | Maximum batch size cap |
| `num_iterations` | `int` or `None` | `None` | Number of batches to yield. `None` = infinite |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

Account with `acc.truncated_poisson(acc.gaussian(nm), sample_rate,
batch_size_cap, dataset_size)`.

## BallsInBinsSampler

```python
from opaque import BallsInBinsSampler
from opaque.random import key

sampler = BallsInBinsSampler(
    data_source,
    num_bins=dataset_size // batch_size,
    num_epochs=8,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `num_bins` | `int` | required | Number of bins per epoch (≥ 2). Typically `dataset_size / batch_size` |
| `num_epochs` | `int` or `None` | `None` | Number of epochs. `None` = infinite |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

Bin sizes are variable (Binomial distribution). Assignments are **fixed
across epochs** (required for BnB dominating-pair accounting). Empty bins
are skipped.

Account with `acc.balls_in_bins(mechanism, num_bins, num_epochs)` where
`mechanism` is `acc.lambda_cgd(...)`, `acc.bisr(...)`, `acc.blt(...)`, or
`acc.gaussian(...)`.

## CyclicPoissonSampler

```python
from opaque import CyclicPoissonSampler
from opaque.random import key

sampler = CyclicPoissonSampler(
    data_source,
    sampling_prob=0.5,
    cycle_length=4,
    iterations=1000,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sampling_prob` | `float` | required | Probability of including each eligible example, in (0, 1] |
| `cycle_length` | `int` | `1` | Number of groups to partition into. 1 = standard Poisson |
| `iterations` | `int \| None` | `None` | Total batches to yield. None = 1 epoch |
| `truncated_batch_size` | `int \| None` | `None` | Maximum batch size cap |
| `partition_type` | `PartitionType` | `EQUAL_SPLIT` | How to partition: `EQUAL_SPLIT` or `INDEPENDENT` |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

In distributed training, shard the dataset with `local_shard()` and pass
a per-rank key via `fold_in(key, rank)`. Best used with `mf_noise`
for correlated noise (DP-FTRL).

## Distributed Helpers

### `local_shard`

Partition a dataset for DDP training. Returns a `Subset` containing the
contiguous shard for the given rank.

```python
from opaque.sampling.distributed import local_shard
import torch.distributed as dist

shard = local_shard(
    dataset,
    rank=dist.get_rank(),
    world_size=dist.get_world_size(),
)
sampler = PoissonSampler(shard, sample_rate=0.01, key=fold_in(key(42), rank))
loader = DataLoader(shard, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | dataset with `len()` | required | The full training dataset |
| `rank` | `int` | `0` | Current device rank |
| `world_size` | `int` | `1` | Total number of devices |

**Returns:** `torch.utils.data.Subset` containing the local shard.

::: opaque.sampling.distributed.local_shard
    options:
      show_source: true
      heading_level: 3

## API Documentation

::: opaque.sampling.poisson.PoissonSampler
    options:
      show_source: true
      heading_level: 3

::: opaque.sampling.truncated_poisson.TruncatedPoissonSampler
    options:
      show_source: true
      heading_level: 3

::: opaque.sampling.cyclic_poisson.CyclicPoissonSampler
    options:
      show_source: true
      heading_level: 3

::: opaque.sampling.b_min_sep.BMinSepSampler
    options:
      show_source: true
      heading_level: 3

::: opaque.sampling.balls_in_bins.BallsInBinsSampler
    options:
      show_source: true
      heading_level: 3

## SequentialBatchSampler

```python
from opaque import SequentialBatchSampler

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

::: opaque.sampling.sequential.SequentialBatchSampler
    options:
      show_source: true
      heading_level: 3
