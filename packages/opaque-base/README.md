# opaque-base

Pure-Python foundation for the Opaque library. Ships the serialization
registry and dispatcher that every other `opaque-*` wheel registers
handlers against:

- `opaque.api.base.serialization._registry` — `register_serializer`,
  `lookup_serializer`.
- `opaque.api.base.serialization._dispatch` — `state_dict`,
  `from_state_dict`.
- `opaque.api.base.serialization._structural` — generic Python container
  walker (dataclass, NamedTuple, tuple, list, mappings, primitives). Torch
  tensors and NumPy arrays are registered as exact-type handlers from
  `opaque-engine` and `opaque-accounting` independently.

`opaque-base` has **no third-party dependencies** — only the Python
stdlib. This is what lets `opaque-accounting` ship as a torch-free
standalone wheel: it depends only on `opaque-base`.

User-facing API lives at the `opaque.serialization` façade (shipped by
this wheel).
