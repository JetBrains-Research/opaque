# opaque-dpftrl

Matrix-factorization noise mechanisms for Opaque: BLT, BSR, BiSR,
band-MF, lambda-CGD, identity — plus the MF-specific participation
samplers (b-min-separation, Poisson, balls-in-bins, sequential
batches). Functional optimizers (including the universal ``adamw``
that consumes private ``noisy_squared_grads`` streams) live in
[`opaque.optimizers`](../opaque-core/README.md).

## Install

```bash
pip install "opaque[dpftrl]"
```

This package is an internal implementation package in the `opaque.*`
namespace. Use `opaque` (with the `dpftrl` extra) as the public
installation target.

## Quick start

```python
from opaque.random import key
from opaque.dpftrl.noise import mf_noise, blt_strategy
from opaque.dpftrl.sampling import BMinSepSampler
```

## Layout

- `opaque.dpftrl.noise` — strategies (band-MF, BLT, BSR, BiSR, identity, lambda-CGD) + dispatchers
- `opaque.dpftrl.clipping` — MF-safe `clipped_grad`, `auto_clipped_grad`, `per_group`
- `opaque.dpftrl.sampling` — `BMinSepSampler`, `CyclicPoissonSampler`, `BallsInBinsSampler`, `SequentialBatchSampler`

Shared clipping implementation and other cross-cutting primitives live in
[`opaque-core`](../opaque-core/README.md).
