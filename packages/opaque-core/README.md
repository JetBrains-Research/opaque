# opaque-core

Core primitives for the Opaque differential-privacy ecosystem. Ships
algorithm-agnostic modules under the user-facing top-level `opaque.*`
namespace.

## Install

```bash
pip install opaque-core
```

## Contents

Top-level (user-facing):

- `opaque.functional` — `make_functional`, `with_batch_dim` (PyTorch <->
  functional API bridges)
- `opaque.serialization` — flat ``state_dict`` / ``from_state_dict`` for
  functional runtime state (optimizers, clipping, noise state, …).
- `opaque.optimizers` — Opaque-built functional optimizer factories
  with a wrapper-aware update surface: `sgd`, `adam`, `adamw`, `lion`,
  `ademamix`, `adafactor`, `rmsprop`, `adagrad`, and the `schedule_free`
  wrapper. All return `torchopt`-compatible
  `GradientTransformation`s; DP-aware modes read `NoisyPytree` metadata
  (DP-AdamW-BC) or private second-moment streams at `update()` time. A small
  set of vanilla TorchOpt primitives (`adadelta`, `radam`) is re-exported for
  convenience.
  Schedule-free's published weights are reachable as ``opt_state.x``.  See
  [`docs/api/optimizers.md`](../../docs/api/optimizers.md) for the
  full reference.
- `opaque.distributed` — DDP plumbing (`is_distributed`, `get_rank`,
  `all_reduce`, `reduce_pytree`, `sum_gradients`, `gather_pytree`,
  `reduce_scalar`, `sync_object`, `sync`, `local_shard`). Submodules:
  `collectives`, `gradients`, `state`, `shard`.

Additional primitives:

- `opaque.random` — JAX-style RNG keys, PyTorch generator bridge
- `opaque.pytree` — `tree_map`, `tree_leaves`, `partition`, `merge`,
  `global_norm`
- `opaque._clipping` — internal implementation of fixed + AUTO-S clipping
  (import via `opaque.dpsgd.clipping` or `opaque.dpftrl.clipping` in
  application code). Distributed sync registers on import of those modules.
- `opaque.profiling` — `TrainingProfiler`, `StepTimer`, memory diagnostics
- `opaque.types` — wrapper-pytree types (`ClippedPytree`, `NoisedPytree`,
  `PerGroup`, `ClipState`, `NoiseState`, …)

## Usage

```python
from opaque.random import key
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.sampling import PoissonSubsampler

rng = key(0)
# ... use clipped_grad + PoissonSubsampler in a DP training loop
```

## Partition policy

`opaque-core` holds only algorithm-agnostic primitives. DP-SGD-specific
mechanisms (Gaussian noise, adaptive/auto clipping, truncated + standard
Poisson samplers) live in `opaque-dpsgd`. DP-FTRL mechanisms (BLT / Toeplitz
/ BSR / BiSR / λ-CGD noise, private second-moment streams, b-min-sep /
cyclic-Poisson / balls-in-bins / sequential samplers) live in `opaque-dpftrl`.
