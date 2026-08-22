# opaque-dpftrl

Matrix-factorization noise mechanisms for Opaque: BLT, BSR, BiSR,
band-MF, lambda-CGD, identity — plus the MF-specific participation
samplers (b-min-separation, Poisson, balls-in-bins, sequential
batches). Functional optimizers (including the universal `adamw`
that consumes private `noisy_squared_grads` streams) live in
[`opaque.optimizers`](../opaque-optimizers/README.md).

The noise and clipping paths execute eagerly on the active provider. They
accept and return provider-native arrays, retain provider-native streaming
state, and support scalar or `PerGroup` clipping bounds and optional private
second-moment streams. Strategy construction and accounting stay on the host:
strategy coefficients are NumPy arrays and are projected into immutable
provider-independent execution plans before native array computation begins.

## Install

Install the root package as described in the [repository installation guide](https://github.com/JetBrains-Research/opaque#installation),
using its `dpftrl` extra to include this component:

```bash
pip install "opaque[dpftrl]"      # includes the Torch provider
```

Libraries that do not want the default bundle can install the mechanism wheel
with only their provider:

```bash
pip install opaque-dpftrl opaque-torch
```

The first gradient template or clipped pytree activates the matching installed
provider. Applications can instead select one explicitly with
`opaque.backend.set_backend("torch")` before constructing the mechanism.

## Quick start

```python
from opaque.random import key
from opaque.dpftrl.noise import mf_gaussian_noise, blt_strategy
from opaque.dpftrl.sampling import BMinSepSampler
```

`mf_gaussian_noise(..., compute_dtype=None)` resolves internal sampling and
linear-combination arithmetic to the active provider's `float32`; outputs are
cast back to each input leaf's dtype. A fixed `RngKey` and the same state replay
deterministically within one provider; providers use their own native random
implementations, so samples are not promised to agree across them.

Checkpoint `MFNoiseState` or `SecondMomentMFNoiseState` through
`opaque.serialization.state_dict` and restore it against a freshly constructed
state template with `from_state_dict`. The next eager call continues the saved
provider-native stream, including its key, step counter, and correlation
buffers.

Hugging Face Transformers integration is provided separately by
`opaque-transformers` and remains Torch-only.

## Layout

- `opaque.dpftrl.noise` — strategies (band-MF, BLT, BSR, BiSR, identity, lambda-CGD) + dispatchers
- `opaque.dpftrl.clipping` — MF-safe `clipped_grad`, `auto_clipped_grad`, `per_group`
- `opaque.dpftrl.sampling` — `BMinSepSampler`, `CyclicPoissonSampler`, `BallsInBinsSampler`, `SequentialBatchSampler`

Shared clipping implementation and other cross-cutting primitives live in
[`opaque-engine`](../opaque-engine/README.md).
