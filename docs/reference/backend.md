# opaque.backend

Backend selection and activation for the dispatched engine. Most code never
touches this module: passing a native array to any Opaque call activates the
matching provider by inference. The explicit API covers cold-process
activation, scoped overrides, and error handling. See
[Backends](../user-guide/backends.md) for the concepts.

```python
from opaque.backend import set_backend, use_backend, active_backend

set_backend("torch")            # activate by name from a cold process
with use_backend("torch"):      # scoped selection
    ...
backend = active_backend()      # the active backend, or None
```

## Selection

::: opaque.backend.set_backend
    options:
        show_source: true
        heading_level: 3

::: opaque.backend.use_backend
    options:
        show_source: true
        heading_level: 3

::: opaque.backend.active_backend
    options:
        show_source: true
        heading_level: 3

::: opaque.backend.clear_backend
    options:
        show_source: true
        heading_level: 3

::: opaque.backend.ensure_backend
    options:
        show_source: true
        heading_level: 3

## Identity and errors

::: opaque.backend.KnownBackend
    options:
        show_source: true
        heading_level: 3

::: opaque.backend.BackendError
    options:
        heading_level: 3

::: opaque.backend.BackendNotSelectedError
    options:
        heading_level: 3

::: opaque.backend.BackendMismatchError
    options:
        heading_level: 3

::: opaque.backend.MixedBackendError
    options:
        heading_level: 3

::: opaque.backend.BackendProviderError
    options:
        heading_level: 3

## Declaring an operation

For an operation [`opaque.ops`](ops.md) does not provide, declare a primitive
and register an implementation per backend. The declaring code stays neutral;
dispatch resolves the implementation from the active backend.

```python
from opaque.primitive import BackendProvider, primitive


@primitive
def selective_log_softmax(logits: object, indices: object) -> object:
    raise NotImplementedError


_MINE = BackendProvider("torch")


@_MINE.implements(selective_log_softmax)
def _torch_impl(logits, indices):
    import torch

    return torch.log_softmax(logits, dim=-1).gather(-1, indices)
```

Guard a call where an implementation may be missing, and provide a fallback:

```python
if selective_log_softmax.supports("torch"):
    scores = selective_log_softmax(logits, indices)
else:
    scores = ops.sum(ops.multiply(logits, mask), axis=-1)
```

Declarations made through this façade are `PrimitiveTier.OPTIONAL` — the
default, and the only tier valid outside the engine. `PrimitiveTier.CORE` is
the profile every provider must implement in full before it may activate, so a
`CORE` declaration made in user code appends to that profile and makes every
shipped provider incomplete: `set_backend` then raises
`IncompleteBackendError` for the rest of the process, including for code that
never touches the extension. The core-profile machinery stays at
`opaque.api.engine.primitive`, where providers built inside this repository
work.

::: opaque.primitive.primitive
    options:
        show_source: true
        heading_level: 3

::: opaque.primitive.Primitive
    options:
        show_source: false
        heading_level: 3

::: opaque.primitive.PrimitiveTier
    options:
        heading_level: 3

::: opaque.primitive.BackendProvider
    options:
        show_source: false
        heading_level: 3

::: opaque.primitive.supports
    options:
        heading_level: 3

::: opaque.primitive.registered_backends
    options:
        heading_level: 3

::: opaque.primitive.PrimitiveError
    options:
        heading_level: 3

::: opaque.primitive.UnsupportedPrimitiveError
    options:
        heading_level: 3

::: opaque.primitive.IncompleteBackendError
    options:
        heading_level: 3

::: opaque.primitive.DuplicatePrimitiveRegistrationError
    options:
        heading_level: 3

::: opaque.primitive.InvalidPrimitiveRegistrationError
    options:
        heading_level: 3
