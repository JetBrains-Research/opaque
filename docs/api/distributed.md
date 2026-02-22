# opaque.distributed

Distributed training utilities for differential privacy with DDP.

## Overview

The `opaque.distributed` module provides composable primitives for multi-GPU
DP training:

- **Core**: `is_distributed()`, `get_rank()`, `get_world_size()`
- **Gradient aggregation**: `sum_gradients()` (AllReduce SUM on PyTrees)
- **State sync**: `sync()`, `sync_object()`, `reduce_scalar()`, `gather_tensors()`

DDP is the only supported parallelism strategy.
See [User Guide: Distributed Training](../user-guide/distributed.md) for usage.

## Core Utilities

::: opaque.distributed.is_distributed
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.get_rank
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.get_world_size
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.all_reduce
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.barrier
    options:
        show_source: true
        heading_level: 3

## Gradient Aggregation

::: opaque.distributed.sum_gradients
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.reduce_pytree
    options:
        show_source: true
        heading_level: 3

## State Synchronization

::: opaque.distributed.reduce_scalar
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.sync
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.sync_object
    options:
        show_source: true
        heading_level: 3

## Tensor Gathering

::: opaque.distributed.gather_tensors
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.gather_pytree
    options:
        show_source: true
        heading_level: 3

::: opaque.distributed.assert_scalar_equal
    options:
        show_source: true
        heading_level: 3

## Privacy Ordering

The order of operations matters for DP guarantees:

```python
grads, clip_state = grad_fn(params, x, y, state=clip_state)  # 1. Clip
grads = dist_utils.sum_gradients(grads)                       # 2. Aggregate
noisy_grads, noise_state = noise_fn(grads, noise_state)       # 3. Noise
```

See [examples/distributed_dp_training.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/distributed_dp_training.py)
for a complete working script.
