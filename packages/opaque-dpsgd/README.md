# opaque-dpsgd

Differentially Private SGD mechanisms for Opaque: Gaussian /
truncated-Gaussian noise, adaptive clipping, and truncated Poisson
subsampling. Fixed and AUTO-S clipping live in
[`opaque.dpsgd.clipping`](../opaque-core/README.md) (implemented in
`opaque._clipping` inside `opaque-core`). Functional optimizers
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
from opaque.dpsgd.clipping import auto_clipped_grad, clipped_grad
from opaque.random import key
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import PoissonSubsampler
```

## Layout

- `opaque.dpsgd.noise` — `gaussian_noise`, `truncated_gaussian_noise`
- `opaque.dpsgd.clipping` — `clipped_grad`, `auto_clipped_grad`, `per_group`, `adaptive_clipped_grad`, `.types`, `.fun`
- `opaque.dpsgd.sampling` — `PoissonSubsampler` (optional ``truncated_batch_size``)

Shared low-level clipping code ships in `opaque-core` as `opaque._clipping`;
RNG keys, pytree helpers, distributed plumbing, and serialization also live
in [`opaque-core`](../opaque-core/README.md).
