# opaque.backend

Backend selection and activation for the dispatched engine. Most code never
touches this module: passing native arrays (a `torch.Tensor`, say) to any
Opaque primitive activates the matching provider by inference. The explicit
API exists for cold-process activation, scoped overrides, and error handling.

```python
from opaque.backend import set_backend, use_backend, active_backend

set_backend("torch")            # activate by name from a cold process
with use_backend("torch"):      # scoped selection
    ...
backend = active_backend()      # the currently selected backend, or None
```

Selection is context-local and sticky: once selected (explicitly or by
inference), the backend stays active for the current context. The registry
reserves the provider names in `KnownBackend`; selecting a name whose
provider wheel is not installed raises a guided install error.

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
