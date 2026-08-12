# Backend providers and primitives

Opaque extensions own their operations as named primitives. A primitive is a
canonical operation whose implementation is selected by the active backend's
stable `name`; it is not selected from the type of an input array. This keeps
mechanisms portable without requiring changes to Opaque's backend interface.

## Declare an extension primitive

Declare a qualified name once, in the package that owns the operation. A
provider can register an eager callable or a lazy `"module:attribute"` target.
Lazy targets are imported when first called and are then cached.

```python
from opaque.primitive import Primitive, lazy_implementation

selective_log_softmax = Primitive("example.alignment.selective_log_softmax")
selective_log_softmax.register_many({
    "torch": lazy_implementation("example._torch:selective_log_softmax"),
    "jax": lazy_implementation("example._jax:selective_log_softmax"),
    "mlx": lazy_implementation("example._mlx:selective_log_softmax"),
})
```

Primitive names are globally canonical. Registering a second implementation
for the same name and backend raises `DuplicatePrimitiveRegistrationError`;
use `replace=True` only for a deliberate override, such as an isolated test.
`primitive.supports("jax")`, `primitive.registered_backends()`, and the
module-level `supports()` / `registered_backends()` helpers provide diagnostics
without resolving lazy targets.

When a backend has no registration, calling the primitive raises
`UnsupportedPrimitiveError`. Its `primitive_name` and `backend_name`
attributes identify the missing contract.

## Activate a provider

A provider is any object with a stable `name` attribute. Register its core
primitive implementations before activating it:

```python
from opaque.backend import use_backend

with use_backend(jax_backend()):
    result = selective_log_softmax(logits, indices)
```

`use_backend()` is nested, exception-safe, and context-local. `set_backend()`
selects a backend for the current context until replaced. The bundled Torch
provider is active by default, so applications that do not choose a provider
continue to use Torch.

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

`set_backend()` and `use_backend()` validate the versioned portable core
profile. `core_profile()` exposes the version and required primitives, and
`validate_core_primitives()` is available to provider tests. A missing core
registration raises `IncompleteBackendError` before the backend becomes
active.

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

1. Give the provider a unique, stable `name`.
2. Register every primitive in `core_profile().primitives` under that name.
3. Keep registration idempotent when a provider factory can be called more
   than once.
4. Add optional capability registrations only for behavior the provider
   actually supports.
5. Exercise activation, unsupported optional calls, keyed randomness, and
   `vmap(randomness="error")` in the provider's conformance tests.