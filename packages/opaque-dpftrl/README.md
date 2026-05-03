# opaque-dpftrl

Matrix-factorization noise mechanisms for Opaque: BLT, BSR, BiSR,
band-MF, JME, lambda-CGD, identity — plus the MF-specific participation
samplers (b-min-separation, cyclic Poisson, balls-in-bins, sequential
batches). Functional optimizers (including the universal ``adamw``
that consumes ``noisy_squared_grads`` from JME) live in
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

- `opaque.dpftrl.noise` — strategies (band-MF, BLT, BSR, BiSR, identity, JME, lambda-CGD) + dispatchers
- `opaque.dpftrl.sampling` — `BMinSepSampler`, `CyclicPoissonSampler`, `BallsInBinsSampler`, `SequentialBatchSampler`

All algorithm-agnostic primitives (Poisson sampling, fixed clipping,
RNG keys, pytree / distributed / profiling helpers) live in
[`opaque.core`](../opaque-core/README.md).
