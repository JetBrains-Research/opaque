# Backends

Opaque's mechanisms are backend-neutral. `opaque-engine` holds the algorithms —
clipping, noise, samplers, optimizer rules, accounting — written against a
portable set of primitives (`opaque.ops`, `opaque.pytree`, `opaque.autodiff`,
`opaque.random`). A **backend provider** supplies implementations of those
primitives for one array framework. `opaque-torch` is the default provider and
`opaque-mlx` supports MLX on Apple Silicon; `KnownBackend` also reserves the
`jax` name.

Mechanism code is written once and runs on whichever provider is active, so
training code normally never mentions this module.

## Automatic selection

Passing a native array or model to any Opaque call selects the matching
provider:

```python
import mlx.core as mx
from opaque import ops

ops.sum(mx.ones((3,)))    # selects the MLX provider
```

Selection is context-local and sticky: it stays active for the rest of the
context, so later calls that carry no array — `ops.zeros(...)`, sampling noise
from an `RngKey` — dispatch to the same provider. Selecting a provider also
registers its serialization handlers, so `state_dict` round-trips native
arrays.

## Explicit selection

Select a provider up front when setup runs before any native array exists:

```python
from opaque.backend import set_backend

set_backend("torch")
# or, on Apple Silicon with opaque-mlx installed:
set_backend("mlx")
```

`use_backend` scopes a selection, `active_backend` queries it, and
`clear_backend` returns to inference:

```python
from opaque.backend import active_backend, clear_backend, use_backend

with use_backend("torch"):
    ...
active_backend()      # the active backend, or None
clear_backend()
```

## Errors

| Condition | Error |
|-----------|-------|
| Nothing active and no array in the arguments | `BackendNotSelectedError` |
| A named provider whose wheel is not installed | `BackendProviderError`, naming the wheel |
| One call carrying arrays from two providers | `MixedBackendError` |
| An array from a provider other than the active one | `BackendMismatchError` |

## Provider-specific surfaces

Some features are inherently framework-shaped. `opaque-torch` owns in-place
DDP collectives, Hugging Face patches, and Triton kernels; `opaque-mlx` owns
MLX module functionalization, device helpers, and explicit MLX-group lifecycle.
They are documented under [`opaque.torch`](../reference/torch.md) and
[`opaque.mlx`](../reference/mlx.md).

See [`opaque.backend`](../reference/backend.md) for the API and
[Installation](../getting-started/installation.md) for the wheel layout.
