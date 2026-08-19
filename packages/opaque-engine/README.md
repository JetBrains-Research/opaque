# opaque-engine

Backend-neutral execution substrate for Opaque. Install it with a provider
wheel — `opaque-torch` is the one shipped today. `opaque-engine` itself neither
installs nor imports a framework.

The engine owns portable primitive declarations and algorithms: `opaque.ops`,
`opaque.autodiff`, `opaque.pytree`, `opaque.random`, and the backend lifecycle
under `opaque.backend`. Its runtime contract exposes distributed and
observability profiles whose supported operations are discoverable at runtime.
Providers receive and return native arrays, dtypes, and device values; they
also register their native serialization handlers when activated.

Shared algorithms and state containers remain in the engine, including
clipping, noise allocation, schedules, generic functional/batch helpers, and
portable profiling interfaces. Framework-specific conveniences belong to their
provider; for example, PyTorch module conversion is
`opaque.torch.functional.make_functional`.

Use the public façades `opaque.types`, `opaque.pytree`, `opaque.random`,
`opaque.distributed`, `opaque.functional`, `opaque.scheduling`, and
`opaque.profiling`. Clipping is exposed by stack façades
(`opaque.dpsgd.clipping`, `opaque.dpftrl.clipping`). `opaque.backend` covers
provider selection and activation.
