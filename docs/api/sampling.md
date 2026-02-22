# Sampling

The `opaque.sampling` module provides privacy-amplifying sampling mechanisms for DP-SGD training.

## Overview

**Privacy amplification through sampling** is a key technique in DP-SGD: training on randomly selected subsets provides
stronger privacy than training on the full dataset.

Opaque provides two sampling strategies:

1. **Poisson Sampling**: Each example sampled independently with probability `sample_rate`
  - Variable batch sizes
  - Strong privacy amplification
  - Standard in DP research

2. **Truncated Poisson Sampling**: Poisson sampling with bounded batch sizes
  - Bounded batch sizes (predictable memory usage)
  - Tighter privacy bounds than fixed-batch sampling
  - Well suited for production workloads

**See also**: [Poisson Sampling & Microbatching User Guide](../user-guide/sampling.md)

## Poisson Sampler

::: opaque.sampling.poisson
options:
members:
- PoissonSampler

## Truncated Poisson Sampler

::: opaque.sampling.poisson
options:
members:
- TruncatedPoissonSampler
