# Backend providers and primitives

Opaque extensions own their operations as decorated primitives. On the first
backend-bearing call, Opaque recognizes Torch, JAX, or MLX arguments, lazily
loads the corresponding provider wheel, and selects its implementation. The
selection remains active in the current context for later calls.

Install `opaque-engine` together with the provider wheel an application uses,
or install `opaque` for the default Torch bundle and `opaque[jax]` or
`opaque[mlx]` for the additional first-party providers.

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
def selective_log_softmax_impl(logits, indices): ...
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
accepts only `fn`, `in_axes`, and `out_axes`; providers own any framework-specific
hidden-RNG policy. Torch uses its native default and rejects hidden random
operations, while JAX and MLX use their native transforms. Deterministic
functions are the portable conformance surface; Opaque does not define cross-provider semantics
for hidden model RNG. `normal(rng_key, shape, ...)` must derive its result from
the immutable key without mutating hidden generator state.

Automatic activation, `set_backend()`, and `use_backend()` validate the
versioned portable core profile. `core_profile()` exposes the version and
required primitives, and `validate_core_primitives()` is available to provider
tests. A missing core registration raises `IncompleteBackendError` before the
backend becomes active.

## Optional capabilities

Runtime integration uses named profiles to advertise groups of related
primitives. `RuntimeProfile.DISTRIBUTED` covers eager process-level rank,
size, barriers, return-based reductions, native-array gathering, and object
gathering. `RuntimeProfile.OBSERVABILITY` covers synchronization and
normalized memory observations. `ExecutionProfile` covers optional
execution transforms: `COMPILATION` (`compile`), `CHECKPOINTING`
(`checkpoint`), and `SAVED_ACTIVATIONS` (`optimize_saved_activations`).

Use `RuntimeProfile.DISTRIBUTED.supports(backend)`,
`RuntimeProfile.OBSERVABILITY.supports(backend)`, or
`ExecutionProfile.COMPILATION.supports(backend)` to discover complete profile
support; use each primitive's `.supports(backend)` method for finer-grained
capabilities.

The first-party capability matrix is:

| Integration | Torch | JAX | MLX |
|---|---:|---:|---:|
| Portable core | yes | yes | yes |
| Distributed profile | yes | yes | yes |
| Observability profile | yes | yes | yes |
| Execution profile: compilation | yes | yes | yes |
| Execution profile: checkpointing | yes | yes | yes |
| Execution profile: saved activations | yes | yes | yes [^1] |
| Native array serialization | `Tensor` + `Parameter` | `jax.Array` | `mlx.core.array` |
| Allocator cache clear | yes | no | yes |
| Peak-memory reset | yes | no | yes |
| Trace annotation | yes | yes | no |

[^1]: On MLX `optimize_saved_activations` is an identity transform and emits
a one-time warning: unified memory removes the separate host/device placement
problem, but total activation storage is not reduced.

Memory fields remain `None` when the selected device cannot expose them. In
particular, JAX reports only fields supplied by `Device.memory_stats()`, and
MLX does not currently expose device capacity. Provider-level support for an
allocator operation does not imply that every device type implements it.

An optional capability is checked at the call site. If it is unavailable,
allow `UnsupportedPrimitiveError` to reach the caller rather than silently
falling back or ignoring an option. A provider that implements only the
portable core can therefore be activated and can run portable algorithms.

## Organize provider integrations

First-party providers keep registrations in modules under their backend
package: `_core.py` for the portable compute profile, `_runtime.py` for
distributed and observability operations, `_serialization.py` for native
array handlers, and `_execution.py` for optional execution transforms such as
`compile`, `checkpoint`, and `optimize_saved_activations`. The public provider
factory imports all of these areas and registers serialization handlers
idempotently.

Provider loading is also the registration boundary for native serialization.
If serialization is the first Opaque operation, activate the provider first:

```python
from opaque.jax import jax_backend
from opaque.serialization import state_dict

jax_backend()
checkpoint = state_dict(params)
```

An earlier backend-bearing primitive call performs the same activation
automatically. `opaque-base` deliberately does not import provider wheels.

## Provider registration checklist

1. Give the provider a unique, stable `BackendProvider` identity.
2. Register every primitive in `core_profile().primitives` with
   `@provider.implements(...)`.
3. Keep registration idempotent when a provider factory can be called more
   than once.
4. Add optional capability registrations only for behavior the provider
   actually supports.
5. Exercise activation, unsupported optional calls, keyed randomness, and
   deterministic `vmap` behavior in the provider's conformance tests.