# Sampling

The `opaque.sampling` module provides privacy-amplifying sampling mechanisms
for DP-SGD training.

## Overview

Privacy amplification through sampling is a key technique in DP-SGD: training
on randomly selected subsets provides stronger privacy than training on the
full dataset.

Opaque provides three sampling strategies:

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

**See also**: [Sampling & Microbatching User Guide](../user-guide/sampling.md)

## PoissonSampler

```python
from opaque import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(
    data_source,
    sample_rate=batch_size / len(data_source),
    num_epochs=1,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sample_rate` | `float` | required | Probability of including each example, in (0, 1] |
| `num_epochs` | `int` | `1` | Number of epochs to iterate |
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
    num_epochs=1,
    key=key(42),
)
loader = DataLoader(dataset, batch_sampler=sampler)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_source` | dataset with `len()` | required | The training dataset |
| `sample_rate` | `float` | required | Expected sampling rate, in (0, 1] |
| `max_batch_size` | `int` | required | Maximum batch size cap |
| `num_epochs` | `int` | `1` | Number of epochs to iterate |
| `key` | `RngKey` | required | RNG key for reproducible sampling |

Account with `acc.truncated_poisson(acc.gaussian(nm), sample_rate,
batch_size_cap, dataset_size)`.

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

Auto-detects distributed training: uses SHARDED mode (each rank cycles
through its shard) when `torch.distributed` is initialized.

Best used with `band_mf_noise` or `blt_mf_noise` for correlated noise.

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
