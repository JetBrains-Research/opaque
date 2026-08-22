# opaque.execution

Optional, backend-dispatched execution transforms. They are not part of the
portable core profile: a provider registers them separately, and callers
discover support through `opaque.execution.types.ExecutionProfile`.

```python
from opaque.execution import compile, checkpoint, optimize_saved_activations

compiled = compile(fn)
checkpointed = checkpoint(block)
offloaded = optimize_saved_activations(block)
```

A transform is constructed from the target callable alone and needs no active
provider; it binds to the backend of its first invocation and caches one
transformed callable per backend. See
[Memory Optimizations](../user-guide/memory-optimizations.md) for semantics,
composition order, and examples.

## Execution transforms

::: opaque.execution.compile
    options:
        show_source: true
        heading_level: 3

::: opaque.execution.checkpoint
    options:
        show_source: true
        heading_level: 3

::: opaque.execution.optimize_saved_activations
    options:
        show_source: true
        heading_level: 3

## Capability discovery

::: opaque.execution.types.ExecutionProfile
    options:
        show_source: true
        heading_level: 3

::: opaque.execution.types.ExecutionProfileSnapshot
    options:
        show_source: true
        heading_level: 3

::: opaque.execution.supports_profile
    options:
        show_source: true
        heading_level: 3

::: opaque.execution.profile_primitives
    options:
        show_source: true
        heading_level: 3

`opaque.execution.types.EXECUTION_PROFILE_VERSION` is the current version of the
profile contract and is carried in every `ExecutionProfileSnapshot`.
