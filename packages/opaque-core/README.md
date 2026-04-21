# opaque-core

Core primitives for the Opaque differential-privacy ecosystem. Ships
algorithm-agnostic modules in both `opaque.core.*` (internal primitives) and
the user-facing top-level `opaque.*` (hoisted conveniences).

## Install

```bash
pip install opaque-core
```

## Contents

Top-level (user-facing):

- `opaque.functional` — `make_functional`, `with_batch_dim` (PyTorch <->
  functional API bridges)
- `opaque.distributed` — DDP plumbing (`is_distributed`, `get_rank`,
  `all_reduce`, `reduce_pytree`, `sum_gradients`, `gather_pytree`,
  `reduce_scalar`, `sync_object`, `sync`, `local_shard`). Submodules:
  `collectives`, `gradients`, `state`, `shard`.

Internal primitives (`opaque.core.*`):

- `opaque.random` — JAX-style RNG keys, PyTorch generator bridge
- `opaque.core.pytree` — `tree_map`, `tree_leaves`, `partition`, `merge`,
  `global_norm`
- `opaque.clipping` — per-example / per-group clipping primitives
  (`clipped_grad`, `clipped_fun`, `clip_pytree`, `ClipState`,
  `FixedClipState`, `PerGroup`, `per_group`). Distributed sync available
  via `opaque.clipping.distributed`.
- `opaque.functional` — `empty_collate` wrapper for variable-size /
  empty batches (shared by all Poisson-style samplers)
- `opaque.core.noise` — `NoiseState` base class

## Usage

```python
from opaque.random import key
from opaque.clipping import clipped_grad
from opaque.dpsgd.sampling import PoissonSampler

rng = key(0)
# ... use clipped_grad + PoissonSampler in a DP training loop
```

## Partition policy

`opaque-core` holds only algorithm-agnostic primitives. DP-SGD-specific
mechanisms (Gaussian noise, adaptive/auto clipping, truncated + standard
Poisson samplers) live in `opaque-dpsgd`. DP-FTRL mechanisms (BLT / Toeplitz
/ BSR / BiSR / JME / λ-CGD noise, b-min-sep / cyclic-Poisson / balls-in-bins
/ sequential samplers) live in `opaque-dpftrl`.
