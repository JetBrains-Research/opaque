# Distributed sync registry

`opaque.distributed.sync(state)` is the public dispatch entry point
for cross-rank synchronisation of clip / noise / aux state objects.
Like the serialisation registry, it dispatches by exact type.

## The registry

```python
from opaque.api.engine.distributed._registry import (
    register_sync_handler,
    lookup_sync_handler,
)
from opaque.api.engine.distributed._types import SyncHandler  # Protocol
```

`register_sync_handler(type_, handler)` wires a callable that takes a
state object and returns a synced state object (typically a
`dataclasses.replace`-style update with cross-rank-reduced fields).

The Opaque-built handlers live alongside their state types and
register on import:

- `opaque.api.engine.clipping._distributed` → `FixedClipState`,
  `AutoClipState`, `ClippedGradAux`, `AutoClippedGradAux`,
  `ClipPytreeAux`.
- `opaque.api.dpsgd.clipping._distributed` → `AdaptiveClipState`,
  `AdaptiveClippedGradAux`.
- `opaque.api.dpsgd.noise._distributed` → `GaussianNoiseState`.
- `opaque.api.dpftrl.noise._distributed` → MF state classes.

Once registered, calling `opaque.distributed.sync(state)` looks up the
handler and runs it; from rank 0 you see the cross-rank reduction
applied. Outside DDP (`is_distributed() is False`) the handler is
expected to no-op or pass through.

## Handler shape

```python
def sync_my_state(state: MyState) -> MyState:
    """Aggregate any cross-rank reductions into ``state`` and return a synced copy."""
    if not is_distributed():
        return state
    # ... cross-rank logic ...
    return replace(state, my_field=new_field)
```

If your handler also needs to sync an *aux* output (a separate
dataclass returned by your operation alongside the new state),
register a second handler keyed by the aux type.

## Worked example: a custom adaptive clip-state

```python
from dataclasses import replace

from opaque.api.engine.distributed._registry import register_sync_handler
from opaque.distributed import is_distributed
from opaque.distributed.collectives import all_reduce


@dataclasses.dataclass
class MyDriftedClipState:
    threshold: float
    drift_rate: float


def _sync_my_state(state: MyDriftedClipState) -> MyDriftedClipState:
    if not is_distributed():
        return state
    # E.g. average the threshold across ranks.
    averaged = all_reduce(
        torch.tensor(state.threshold), op="mean"
    ).item()
    return replace(state, threshold=averaged)


register_sync_handler(MyDriftedClipState, _sync_my_state)
```

Now `opaque.distributed.sync(state)` dispatches to your handler.

## Where to register

Side-effect import on the module that defines the state class. The
Opaque convention: a `_distributed.py` sibling of the state file that
imports the state class and calls `register_sync_handler` at module
load. The state-class-defining module then does
`from . import _distributed  # noqa: F401` so any consumer that loads
the state class also loads the sync registration.

## See also

- [`opaque.distributed` reference](../reference/distributed.md)
- [Adding a new mechanism family](new-mechanism.md)
