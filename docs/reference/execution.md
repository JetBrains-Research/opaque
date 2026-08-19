# opaque.execution

Optional, backend-dispatched execution transforms. These transforms are not
part of Opaque's portable core profile; a provider registers them separately
and callers discover support through `ExecutionProfile`.

A transform is constructed with only the target callable:

```python
from opaque.execution import compile, checkpoint, optimize_saved_activations

compiled = compile(fn)
checkpointed = checkpoint(block)
offloaded = optimize_saved_activations(block)
```

The returned callable binds lazily to the backend of its first invocation
and caches one transformed callable per backend. The wrappers themselves are
backend-neutral, so construction requires no active provider.

See [Memory Optimizations](../user-guide/memory-optimizations.md) for
transform semantics, composition order, and examples.

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

::: opaque.execution.ExecutionProfile
    options:
        show_source: true
        heading_level: 3

::: opaque.execution.ExecutionProfileSnapshot
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

`EXECUTION_PROFILE_VERSION` is the current version of the execution profile
contract. It is included in `ExecutionProfileSnapshot` and can be used by
provider tests to validate that the optional profile primitives they register
match the engine declaration.
