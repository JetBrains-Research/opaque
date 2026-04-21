# opaque-core

Core primitives for the Opaque differential-privacy ecosystem. All public
API lives under the real package `opaque.core.*`.

## Install

```bash
pip install opaque-core
```

## Contents

- `opaque.core.random` — functional JAX-style RNG keys, PyTorch generator bridge
- `opaque.core.utils` — pytree ops, `make_functional`, `PerGroup`
- `opaque.core.clipping` — per-example / per-group clipping primitives
- `opaque.core.sampling` — Poisson sampler, `poisson_collate`, distributed shards
- `opaque.core.distributed` — DDP plumbing, `sum_gradients`, `gather_pytree`
- `opaque.core.profiling` — memory / step-timer profiler
- `opaque.core.noise.types` — the generic `NoiseState` base class

## Usage

```python
from opaque.core.random import key
from opaque.core.clipping import clipped_grad
from opaque.core.sampling import PoissonSampler

rng = key(0)
# ... use clipped_grad + PoissonSampler in a DP training loop
```

## Partition policy

`opaque-core` holds only algorithm-agnostic primitives. DP-SGD-specific
mechanisms (Gaussian noise, adaptive/auto clipping, truncated Poisson)
live in `opaque-dpsgd`. Matrix-factorization mechanisms (BLT/Toeplitz/BSR
noise, b-min-sep sampling, cyclic Poisson) live in `opaque-mf`.
