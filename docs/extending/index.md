# Extending Opaque

This section documents the **contributor-facing** surface for plugging
new mechanism families, custom state types, distributed-sync handlers,
and low-level clipping helpers into Opaque.

If you're using Opaque to train a model, you don't need anything here —
the [User Guide](../user-guide/index.md) and [API Reference](../reference/index.md)
cover everything users normally touch.

## When to extend

You're in the right place if you want to:

- **Register a custom state type** for serialisation
  (e.g. checkpoint a non-tensor state object). See
  [Serialization registry](serialization.md).
- **Register a sync handler** for a custom clip-state or noise-state
  so that `opaque.distributed.sync(my_state)` does the right thing.
  See [Distributed sync](distributed-sync.md).
- **Use the low-level clipping helpers** (`clipped_fun`,
  `auto_clipped_fun`, `clip_pytree`) to clip something other than a
  per-example loss gradient. See [Clipping `fun`](clipping-fun.md).
- **Plug in a new mechanism family** — a new noise mechanism, a
  custom DP process for accounting, a new clipping rule. See
  [Adding a new mechanism family](new-mechanism.md).

## The `opaque.api.*` namespace

Opaque's public façades live at `opaque.<concern>` (e.g.
`opaque.serialization`, `opaque.types`, `opaque.dpsgd.clipping`). The
implementation lives parallel to that under `opaque.api.*` —
`opaque.api.base.serialization`, `opaque.api.engine.clipping`,
`opaque.api.dpsgd.noise`, and so on. The user-facing façades
re-export selected names from the impl tree.

`opaque.api.*` is **not part of the user-facing surface**:

- Imports from `opaque.api.*` won't break sibling-wheel imports, but
  they are unstable across minor versions. Pin to a specific opaque
  release if you depend on them.
- IDE / traceback paths surface `opaque.api.*`; that's intentional
  ("internal but discoverable").
- This is the only docs section that documents `opaque.api.*` paths.

## Dependency cone

Each wheel ships impl under a specific `opaque.api.<contrib>.*` root:

| Wheel | Contrib root | Depends on |
|---|---|---|
| `opaque-base` | `opaque.api.base.*` | stdlib only |
| `opaque-engine` | `opaque.api.engine.*` | `opaque-base`, torch, numpy, optree |
| `opaque-optimizers` | `opaque.api.optimizers.*` | `opaque-engine`, torchopt |
| `opaque-accounting` | `opaque.api.accounting.core.*` | `opaque-base` (torch-free) |
| `opaque-dpsgd` | `opaque.api.dpsgd.*` and `opaque.api.accounting.dpsgd.*` | `opaque-engine`, `opaque-accounting` |
| `opaque-dpftrl` | `opaque.api.dpftrl.*` and `opaque.api.accounting.dpftrl.*` | `opaque-engine`, `opaque-accounting` |
| `opaque-auditing` | `opaque.api.auditing.*` | `opaque-engine`, `opaque-accounting` |
| `opaque-patches` | `opaque.api.patches.*` | `opaque-engine` |
| `opaque-transformers` | `opaque.api.transformers.*` | `opaque-engine`, `opaque-patches` |

A new mechanism wheel (say `opaque-lipschitz`) lands under its own
`opaque.api.lipschitz.*` and chooses its dep cone like any of the
above. See [Adding a new mechanism family](new-mechanism.md).

## What's covered here

- **[Serialization registry](serialization.md)** — registering
  `state_dict` / `from_state_dict` handlers for custom types.
- **[Distributed sync](distributed-sync.md)** — registering
  cross-rank reduction logic for custom state objects.
- **[Clipping `fun` helpers](clipping-fun.md)** — using the
  lower-level `clipped_fun` / `auto_clipped_fun` / `clip_pytree`
  surface that `clipped_grad` is built on.
- **[Adding a new mechanism family](new-mechanism.md)** — the full
  contribution path: where the impl goes, what to register, how to
  expose a façade.
