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
- `opaque.optimizers` — Opaque-built functional optimizer factories
  with DP-aware paths: `adamw`, `lion`, `ademamix`, `adafactor`, and
  the `schedule_free` wrapper.  All return `torchopt`-compatible
  `GradientTransformation`s; DP-aware modes are selected at `update()`
  time via optional `noise_stddev` (DP-AdamW-BC) and
  `noisy_squared_grads` (private second-moment stream) kwargs.  Vanilla SGD /
  Adam / RMSprop / etc. are not re-exported here — import them from
  torchopt directly when needed (`from torchopt import sgd`).
  Less-common building blocks live in submodules:
  `opaque.optimizers.serialization` (`state_dict` / `load_state_dict`
  for checkpoint round-tripping) and
  `opaque.optimizers.schedule_free` (`get_eval_params` for the
  published `x` weights).  See
  [`docs/api/optimizers.md`](../../docs/api/optimizers.md) for the
  full reference.
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
/ BSR / BiSR / λ-CGD noise, private second-moment streams, b-min-sep /
cyclic-Poisson / balls-in-bins / sequential samplers) live in `opaque-dpftrl`.
