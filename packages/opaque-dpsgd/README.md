# opaque-dpsgd

Differentially Private SGD mechanisms for Opaque: Gaussian /
truncated-Gaussian noise, adaptive clipping, and truncated Poisson
sampling. Fixed and AUTO-S clipping (algorithm-agnostic — they have a
constant per-record sensitivity bound and therefore compose with both
this package's Gaussian mechanism and DP-FTRL's matrix-factorization
mechanisms) live in
[`opaque.clipping`](../opaque-core/README.md). Functional optimizers
(including the universal ``adamw`` with optional DP bias-correction) live
in [`opaque.optimizers`](../opaque-core/README.md).

## Install

```bash
pip install opaque
```

This package is an internal implementation package in the `opaque.*`
namespace. Use `opaque` as the public installation target.

## Quick start

```python
from opaque.clipping import auto_clipped_grad, clipped_grad
from opaque.random import key
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import PoissonSampler
```

## Layout

- `opaque.dpsgd.noise` — `gaussian_noise`, `truncated_gaussian_noise`
- `opaque.dpsgd.clipping` — `adaptive_clipped_grad` (re-exports `auto_clipped_grad` / `auto_clipped_fun` for backward compatibility; canonical home is `opaque.clipping`)
- `opaque.dpsgd.sampling` — `PoissonSampler` (with optional ``truncated_batch_size``)

All algorithm-agnostic primitives (fixed and AUTO-S clipping, Poisson
sampling, RNG keys, pytree / distributed / profiling helpers) live in
[`opaque-core`](../opaque-core/README.md).
