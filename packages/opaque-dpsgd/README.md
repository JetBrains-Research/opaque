# opaque-dpsgd

Differentially Private SGD mechanisms for Opaque: Gaussian /
truncated-Gaussian noise, adaptive and AUTO-S clipping, and truncated
Poisson sampling. Functional optimizers (including the universal
``adamw`` with optional DP bias-correction) live in
[`opaque.optimizers`](../opaque-core/README.md).

## Install

```bash
pip install opaque
```

This package is an internal implementation package in the `opaque.*`
namespace. Use `opaque` as the public installation target.

## Quick start

```python
from opaque.clipping import clipped_grad
from opaque.random import key
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import TruncatedPoissonSampler
```

## Layout

- `opaque.dpsgd.noise` — `gaussian_noise`, `truncated_gaussian_noise`, `per_group_noise_stddev`
- `opaque.dpsgd.clipping` — `adaptive_clipped_grad`, `auto_clipped_grad`, `auto_clipped_fun`
- `opaque.dpsgd.sampling` — `TruncatedPoissonSampler`

All algorithm-agnostic primitives (fixed clipping, Poisson sampling,
RNG keys, pytree / distributed / profiling helpers) live in
[`opaque.core`](../opaque-core/README.md).
