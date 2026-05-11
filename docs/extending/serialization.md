# Serialization registry

Opaque's checkpointing pipeline is a **type-keyed registry** mapping
exact Python types to a `(dump_fn, load_fn)` pair. The user-facing
`opaque.serialization.state_dict` / `from_state_dict` walk an object
graph and consult the registry per-node; whatever isn't registered
falls through to a generic walker that handles dataclasses, NamedTuples,
tuples, lists, dicts, and primitives.

This page documents how to register handlers for your own types —
state objects, custom data classes with non-standard semantics, opaque
artefacts that you want to round-trip.

## The registry

The entry points you'll use, all re-exported on
`opaque.api.base.serialization`:

```python
from opaque.api.base.serialization import (
    register_serializer,
    lookup_serializer,
    Serializer,            # Protocol — defines the (dump, load) shape
    SerializedState,       # alias for dict[str, Any]
)
```

`register_serializer` is also re-exported on the public
`opaque.serialization` façade — both names refer to the same function.

## Handler shape

A serializer is a pair of callables:

```python
def dump(obj) -> dict[str, Any]:
    """Return a flat dict of relative-key paths."""

def load(template, sd: Mapping[str, Any]):
    """Reconstruct the object using ``template`` for shape and ``sd`` for content."""
```

- **`dump`** returns a `{relative_path: leaf_value}` dict. The
  dispatcher rewrites those keys to absolute paths (joining with the
  prefix the parent walked). A leaf value uses the empty key
  `""`; sub-fields use either dotted (`".foo"`) or bracketed
  (`"[0]"`) segments.
- **`load`** receives a *template* (a freshly-initialised object of
  the same shape as at save time) and the *sub-dict* of the flat
  state dict whose keys are relative to this object's root. Missing
  paths keep the template's value (forward compatibility for new
  fields).

## Worked example: registering a custom DP process

```python
from dataclasses import dataclass
from typing import Any, Mapping

from opaque.api.accounting.core._base import DpProcess
from opaque.api.base.serialization import register_serializer


@dataclass
class TrimmedGaussian(DpProcess):
    """Custom process: Gaussian truncated to a Mahalanobis ball."""
    noise_multiplier: float
    radius: float


def _dump_trimmed_gaussian(obj: TrimmedGaussian) -> dict[str, Any]:
    return {
        "type": "TrimmedGaussian/1",
        "noise_multiplier": obj.noise_multiplier,
        "radius": obj.radius,
    }


def _load_trimmed_gaussian(_template, sd: Mapping[str, Any]) -> TrimmedGaussian:
    assert sd["type"] == "TrimmedGaussian/1", sd["type"]
    return TrimmedGaussian(
        noise_multiplier=sd["noise_multiplier"],
        radius=sd["radius"],
    )


register_serializer(
    TrimmedGaussian, _dump_trimmed_gaussian, _load_trimmed_gaussian
)
```

After import, `state_dict` and `from_state_dict` round-trip the
`TrimmedGaussian` instances through the unified
`opaque.serialization` API.

## Versioning

Tag dumps with a version string (`"TrimmedGaussian/1"`) so future-you
can grow the schema without breaking old checkpoints — the loader
inspects the tag and dispatches to the right migration path. The
built-in DP process serializers under `opaque.api.accounting.core._serialization`
follow this pattern.

## Where to put the registration

Side-effect imports register handlers at module-load time. The Opaque
codebase has the convention of registering on the impl module's first
load; for example, `opaque.api.engine.serialization._structural`
registers `torch.Tensor` and `numpy.ndarray` handlers when
`opaque-engine` is imported. Follow the same pattern for your own
type — register at module import, fail loudly if registration fails.

## See also

- [`opaque.serialization` reference](../reference/serialization.md)
- [Adding a new mechanism family](new-mechanism.md) — full contributor
  path including custom state + serializer + sync handler.
- [Composition](composition.md) — the types your state interacts with.
- [Contract tests at a glance](index.md#contract-tests-at-a-glance) —
  the CI gates an extension PR typically trips.
