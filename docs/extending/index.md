# Extending Opaque

This section documents the **contributor-facing** surface for plugging
new mechanism families, custom state types, distributed-sync handlers,
and low-level clipping helpers into Opaque.

If you're using Opaque to train a model, you don't need anything here —
the [User Guide](../user-guide/index.md) and [API Reference](../reference/index.md)
cover everything users normally touch.

> **A note on stability.** The contributor surface described in this
> section is *not* stable yet. These pages describe today's design,
> the conventions an extension is expected to follow, and the
> tripwires it'll bump into — they don't commit to any of it as a
> versioned contract. Language like "today the code …", "the
> convention is …", "things to know about …" is used deliberately
> throughout: an extension that holds to the patterns documented here
> will interoperate with the current Opaque, but the patterns
> themselves may evolve.

## Kinds of extension

Most contributions today fall into one of five shapes. The seams a
given extension touches depend on which shape it is.

| Kind | What it adds | Nearest in-tree precedent | Seams it touches |
|---|---|---|---|
| **Noise mechanism** | A new way of privatising a `ClippedPytree` | `opaque.dpsgd.gaussian`, `opaque.dpftrl.mf_noise` | state class + serializer + sync, façade, usually an accounting factory |
| **Clipping rule** | A new sensitivity-control rule applied to per-example gradients | Fixed clipping, AUTO-S, adaptive | `clipping_fun` builders or a fresh impl, state class + serializer + sync, façade |
| **Sensitivity oracle** | A bound on per-record contribution that comes from architecture, not from a norm computation | (none in-tree today; opaque-lipschitz is the illustrative example) | builder that emits `ClippedPytree(max_norm=R)`, façade; usually *no* new accounting primitive |
| **Accounting primitive** | A new `DpProcess` subclass with PLD machinery and amplification adapters | `opaque.dpsgd.gaussian`, `opaque.dpftrl.mf`, `opaque.dpftrl.lambda_cgd` | `DpProcess` subclass (auto-registers), serializer, amplification adapters |
| **Stack of model modifiers** | A wheel that rewrites or wraps user models to enable DP training (vmap-safety, hook surgery) | `opaque-patches`, `opaque-optimizers`, `opaque-transformers` | impl tree + façade; usually no privacy-account interaction |

The first four are all covered by [Adding a new mechanism family](new-mechanism.md);
the table above is mostly for orientation, so you can tell which
sections of that page apply and which don't. The last category
(model modifiers) is closer in spirit to upstream-library integration —
see [Upstream integration](upstream-integration.md).

If your idea is to make `opaque.distributed.sync(my_state)` work for
a custom state object you've built outside any of the above, the
self-contained recipes are in [Serialization registry](serialization.md)
and [Distributed sync](distributed-sync.md).

## The `opaque.api.*` namespace

Opaque's public façades live at `opaque.<concern>` (e.g.
`opaque.serialization`, `opaque.types`, `opaque.dpsgd.clipping`). The
implementation lives parallel to that under `opaque.api.*` —
`opaque.api.base.serialization`, `opaque.api.engine.clipping`,
`opaque.api.dpsgd.noise`, and so on. The user-facing façades
re-export selected names from the impl tree.

`opaque.api.*` is the contributor surface, not the user surface:

- Imports from `opaque.api.*` won't break sibling-wheel imports, but
  they are unstable. Pin to a specific opaque release if you depend
  on them.
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

## Contract tests at a glance

A handful of CI gates check that the structural conventions
documented in this section hold. They live under
`tests/contracts/`:

| Test file | What it checks |
|---|---|
| `test_facade_discipline.py` | Public façades only re-export; no business logic or `opaque.api.*` strings in user-facing docstrings. |
| `test_no_internal_namespace_in_facades.py` | User-facing `opaque.*` modules don't leak `opaque.api.*` paths through their public surface. |
| `test_pep420_no_init.py` | The three namespace roots (`opaque`, `opaque.api`, `opaque.api.accounting`) ship no `__init__.py`. |
| `test_dependency_direction.py` | Wheels only depend on wheels lower in the dep cone. |
| `test_test_placement.py` | In-wheel tests stay in the wheel; cross-wheel tests live under `tests/integration/`. |
| `test_accounting_factory_leakage.py` | Accounting factory namespaces don't leak across stacks. |
| `test_accounting_torch_free.py` | `opaque-accounting` stays torch-free. |
| `test_docs_stack_discipline.py` | Stack-specific docs don't reach into rival-stack APIs. |

These are the gates an extension PR is most likely to trip. The
individual recipes in this section point at the relevant test
when they touch one of these surfaces.

## What's covered here

- **[Adding a new mechanism family](new-mechanism.md)** — the full
  contribution path: wheel split, layout, façade rules, the three
  registries, when you do (and don't) need a new accounting primitive,
  inline worked example.
- **[Composition](composition.md)** — what shape your output needs
  to take so the rest of the pipeline can consume it
  (`ClippedPytree`, `NoisedPytree`, `PerGroup`, the
  constant-`max_norm` consideration for MF noise).
- **[Upstream integration](upstream-integration.md)** — when to
  reuse, extend, or rewrite a third-party library; in-tree
  precedents and a decision rubric.
- **[Clipping `fun` helpers](clipping-fun.md)** — the lower-level
  `clipped_fun` / `auto_clipped_fun` / `clip_pytree` surface that
  `clipped_grad` is built on.
- **[Serialization registry](serialization.md)** — registering
  `state_dict` / `from_state_dict` handlers for custom types.
- **[Distributed sync](distributed-sync.md)** — registering
  cross-rank reduction logic for custom state objects.
