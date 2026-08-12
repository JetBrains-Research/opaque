# Backend providers and primitives

Opaque extensions own their operations as decorated primitives. On the first
backend-bearing call, Opaque recognizes Torch, JAX, or MLX arguments, lazily
loads the corresponding provider wheel, and selects its implementation. The
selection remains active in the current context for later calls.

## Declare an extension primitive

Declare a primitive once, in the package that owns the operation. The
decorator preserves the declaration's name, signature, and docstring while
turning it into the canonical object providers register against.

```python
from opaque.primitive import PrimitiveTier, primitive

@primitive(tier=PrimitiveTier.CORE)
def selective_log_softmax(logits: object, indices: object) -> object:
    """Select indexed log probabilities."""
    raise NotImplementedError
```

Use `PrimitiveTier.OPTIONAL` for capabilities that are not required for every
provider. Primitive names are derived from the declaration by default; pass
`name="package.operation"` when an explicit identity is useful.

## Implement a provider

Create one `BackendProvider` identity and bind implementations with its
decorator during provider import:

```python
from opaque.primitive import BackendProvider

_PROVIDER = BackendProvider("example")

@_PROVIDER.implements(selective_log_softmax)
def selective_log_softmax_impl(logits, indices):
    ...
```

Registration uses primitive objects rather than repeated string names.
Registering a second implementation for the same primitive and backend raises
`DuplicatePrimitiveRegistrationError`; use `replace=True` only for a
deliberate override, such as an isolated test.
`selective_log_softmax.supports("jax")`,
`selective_log_softmax.registered_backends()`, and the module-level
`supports()` / `registered_backends()` helpers provide diagnostics without
resolving lazy targets.

When a backend has no registration, calling the primitive raises
`UnsupportedPrimitiveError`. Its `primitive_name` and `backend_name`
attributes identify the missing contract.

## Automatic selection

No backend is selected at import time. A primitive or executable transform
inspects its invocation arguments recursively, including common containers,
dataclasses, arrays, and model type hierarchies. The first recognized Torch,
JAX, or MLX value loads `opaque-torch`, `opaque-jax`, or `opaque-mlx` and makes
that backend sticky:

```python
import torch
from opaque.backend import active_backend
from opaque.ops import square

result = square(torch.tensor([2.0]))
assert active_backend().name == "torch"
```

Arguments from multiple recognized frameworks raise `MixedBackendError`.
Arguments that conflict with the sticky backend raise `BackendMismatchError`;
call `clear_backend()` before changing the sticky selection. A call containing
only neutral values requires an already active or explicitly selected backend.
If a recognized provider wheel is absent, the error names the package to
install.

## Explicit lifecycle

Applications normally rely on inference. Tests and uncommon multi-backend
applications can select or switch providers explicitly:

```python
from opaque.backend import clear_backend, set_backend, use_backend
from opaque.jax import jax_backend
from opaque.torch import torch_backend

set_backend(torch_backend())

with use_backend(jax_backend()):
    result = selective_log_softmax(jax_logits, jax_indices)

# The previous Torch selection is restored here.
clear_backend()
```

`use_backend()` is nested, exception-safe, and context-local. `set_backend()`
persists in the current context, while `clear_backend()` returns it to the
unselected state. Third-party providers are explicitly selectable; automatic
loading is deterministic and limited to the first-party providers.

## Implement the portable core

The public authoring modules define the portable core:

- `opaque.ops` for native-array inspection, creation, math, reductions, dtype
  handling, and lifecycle operations;
- `opaque.autodiff` for `grad_and_value` and `vmap`;
- `opaque.pytree` for dispatched tree operations and normalized `ParamPath`;
- `opaque.random` for immutable keys and keyed `normal` sampling.

Implementations receive and return native arrays, dtypes, and device values;
Opaque does not wrap them. `grad_and_value` returns `(grads, value)`. `vmap`
must support `randomness="error"`; it must either implement `"same"` and
`"different"` explicitly or reject them. `normal(rng_key, shape, ...)` must
derive its result from the immutable key without mutating hidden generator
state.

Automatic activation, `set_backend()`, and `use_backend()` validate the
versioned portable core profile. `core_profile()` exposes the version and
required primitives, and `validate_core_primitives()` is available to provider
tests. A missing core registration raises `IncompleteBackendError` before the
backend becomes active.

## Optional capabilities

Distributed execution, device probing, profiling, model functionalization,
and provider-specific serialization are optional capabilities. Register such
operations as ordinary optional primitives, for example under a package-owned
`example.runtime.*` name. Do not add them to the portable-core profile.

An optional capability is checked at the call site. If it is unavailable,
allow `UnsupportedPrimitiveError` to reach the caller rather than silently
falling back or ignoring an option. A provider that implements only the
portable core can therefore be activated and can run portable algorithms.

## Provider registration checklist

1. Give the provider a unique, stable `BackendProvider` identity.
2. Register every primitive in `core_profile().primitives` with
   `@provider.implements(...)`.
3. Keep registration idempotent when a provider factory can be called more
   than once.
4. Add optional capability registrations only for behavior the provider
   actually supports.
5. Exercise activation, unsupported optional calls, keyed randomness, and
   `vmap(randomness="error")` in the provider's conformance tests.