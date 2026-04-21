# opaque-mf

Matrix-factorization noise mechanisms for Opaque: BLT, BSR, BiSR,
band-MF, JME, lambda-CGD, identity — plus the ``AdamW-JME`` optimizer
and the MF-specific participation samplers (b-min-separation,
cyclic Poisson, balls-in-bins, sequential batches).

## Install

```bash
pip install opaque-mf                 # core MF mechanisms
pip install "opaque-mf[optimizers]"   # + AdamW-JME (torchopt)
```

`opaque-mf` depends on `opaque-core` and `opaque-accounting`; both
install automatically. The native accounting extension is loaded
lazily — `import opaque.mf` works without it, as long as no
calibration helper (e.g. `bisr_strategy`) is called.

## Quick start

```python
from opaque.core.random import key
from opaque.mf.noise import mf_noise, blt_strategy
from opaque.mf.sampling import BMinSepSampler
```

## Layout

- `opaque.mf.noise` — strategies (band-MF, BLT, BSR, BiSR, identity, JME, lambda-CGD) + dispatchers
- `opaque.mf.optimizers` — `adamw_jme` (requires the `optimizers` extra)
- `opaque.mf.sampling` — `BMinSepSampler`, `CyclicPoissonSampler`, `BallsInBinsSampler`, `SequentialBatchSampler`

All algorithm-agnostic primitives (Poisson sampling, fixed clipping,
RNG keys, pytree / distributed / profiling helpers) live in
[`opaque.core`](../opaque-core/README.md).
