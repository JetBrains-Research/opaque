# Backends

Opaque's mechanisms are backend-neutral. `opaque-engine` holds the algorithms —
clipping, noise, samplers, optimizer rules, accounting — written against a
portable set of primitives (`opaque.ops`, `opaque.pytree`, `opaque.autodiff`,
`opaque.random`). A **backend provider** supplies implementations of those
primitives for one array framework. `opaque-torch` is the provider shipped
today; `KnownBackend` also reserves the `jax` and `mlx` names.

Mechanism code is written once and runs on whichever provider is active, so
training code normally never mentions this module.

## Automatic selection

Passing a native array or model to any Opaque call selects the matching
provider:

```python
import torch
from opaque import ops

ops.sum(torch.ones(3))    # selects the Torch provider
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

A few operations have a correct answer with no provider at all — annotating a
trace when nothing can record it, asking whether a value is a native array when
nothing is active to own one. Those are declared *neutral* and answer instead
of raising in the first row's case; every other row still raises. Calling code
therefore does not select or probe a backend to use them. See
[declaring an operation](../reference/backend.md#declaring-an-operation).

## Torch-only surfaces

Some features are inherently framework-shaped and live in `opaque-torch` rather
than the engine: `make_functional`, the in-place DDP collectives, the Hugging
Face model patches, and the Triton kernels. They are documented under
[`opaque.torch`](../reference/torch.md).

See [`opaque.backend`](../reference/backend.md) for the API and
[Installation](../getting-started/installation.md) for the wheel layout.
