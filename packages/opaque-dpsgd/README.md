# opaque-dpsgd

DP-SGD mechanisms for Opaque: Gaussian noise (optionally
bounded — [Chen and Hale, 2024](https://arxiv.org/abs/2211.17230)), clipping
(fixed, AUTO-S, adaptive), and Poisson subsampling. Functional optimizers
(including the universal `adamw` with optional DP bias correction) live in
[`opaque.optimizers`](../opaque-optimizers/README.md).

## Install

Install the root package as described in the [repository installation guide](https://github.com/JetBrains-Research/opaque#installation).
`opaque-dpsgd` is included in the default `opaque` package set.

For a standalone installation, install `opaque-dpsgd` with the provider that
owns your native arrays:

```bash
pip install opaque-dpsgd opaque-torch
```

## Backends

The mechanisms operate on native arrays and select the matching installed
provider from their first array-bearing call. Applications that need a provider
before any array is available can call `opaque.backend.set_backend("torch")`.

## Quick start

```python
from opaque.dpsgd.clipping import (
    adaptive_clipped_grad,
    auto_clipped_grad,
    clipped_grad,
)
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import key
```

## Layout

- `opaque.dpsgd.noise` — `gaussian_noise` (optional `bound` for the bounded Gaussian mechanism)
- `opaque.dpsgd.clipping` — `clipped_grad`, `auto_clipped_grad`, `per_group`, `adaptive_clipped_grad`, `.types`, `.fun`
- `opaque.dpsgd.sampling` — `PoissonSampler` (optional `truncated_batch_size`), `KOutOfTSampler`, `RandomAllocationSampler`

RNG keys, pytree helpers, distributed plumbing, and serialization live in
[`opaque-engine`](../opaque-engine/README.md).
