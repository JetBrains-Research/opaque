# Adding a new mechanism family

This is the full contributor recipe for plugging a new DP mechanism
family into Opaque — say a new noise mechanism, a new clipping rule,
or a new accounting primitive. The pattern mirrors the in-tree
`opaque-dpsgd` and `opaque-dpftrl` wheels.

## The wheel split

A new mechanism family lives in its own wheel, e.g. `opaque-lipschitz`.
The wheel ships:

- **Impl tree** at `opaque.api.<contrib>.*` (e.g.
  `opaque.api.lipschitz.{noise,clipping,sampling}`).
- **Public façade** at `opaque.<stack>.*` for stack mechanisms
  (e.g. `opaque.lipschitz.noise`) or `opaque.<concern>` for
  cross-cutting helpers.
- **Accounting factories** at `opaque.api.accounting.<stack>.*` and
  the matching façade (e.g. `opaque.lipschitz.accounting`) when the
  mechanism has a privacy-budget interpretation.

The wheel declares its dep cone in `pyproject.toml`. Most mechanism
wheels depend on `opaque-engine` (for `ClippedPytree` / `NoisedPytree`
/ `PerGroup`) and `opaque-accounting` (for `DpProcess` and the PLD
machinery).

## Layout template

```
packages/opaque-lipschitz/
├── pyproject.toml
├── README.md
├── src/opaque/
│   ├── api/
│   │   ├── lipschitz/
│   │   │   ├── __init__.py            # __all__ = []  (impl tree)
│   │   │   ├── noise/
│   │   │   │   ├── __init__.py        # public impl __init__
│   │   │   │   ├── _engine.py         # main impl
│   │   │   │   ├── _distributed.py    # registers sync handler
│   │   │   │   └── types.py           # state classes
│   │   │   └── clipping/
│   │   │       └── ...
│   │   └── accounting/
│   │       └── lipschitz/
│   │           ├── __init__.py
│   │           ├── mechanisms/
│   │           │   ├── __init__.py
│   │           │   ├── _<mechanism>.py
│   │           │   └── types.py
│   │           └── amplification/
│   │               └── ...
│   └── lipschitz/                     # public façade
│       ├── __init__.py
│       ├── noise/
│       │   ├── __init__.py            # re-exports from opaque.api.lipschitz.noise
│       │   └── types.py
│       └── accounting/
│           ├── __init__.py            # re-exports from opaque.api.accounting.lipschitz
│           └── types.py
└── tests/
    └── ...
```

## Façade rules

The façade modules **only** re-export from the impl tree:

```python
# opaque/lipschitz/noise/__init__.py
"""Lipschitz noise mechanisms."""

from opaque.api.lipschitz.noise import lipschitz_noise

__all__ = ["lipschitz_noise"]
```

No business logic, no conditionals, no `:mod:`opaque.api.*`` strings
in the docstring — those are for `docs/extending/`, not user-facing
docstrings. The CI contract `tests/contracts/test_facade_discipline.py`
enforces this.

## Registering with the foundation

Three registries to plug into:

1. **Serialization** — every state class your mechanism produces
   should round-trip through `opaque.serialization.state_dict`. Side-
   effect import a `_serialization.py` module that calls
   `register_serializer(MyState, dump, load)`. See
   [Serialization registry](serialization.md).

2. **Distributed sync** — every state class that needs cross-rank
   reduction registers a sync handler. Side-effect import a
   `_distributed.py`. See [Distributed sync](distributed-sync.md).

3. **PEP 420 namespaces** — your wheel ships
   `opaque/api/lipschitz/...` and `opaque/lipschitz/...` (or
   `opaque/<stack>/lipschitz/...` if it's a stack-specific
   mechanism family). Do **not** ship
   `opaque/__init__.py`, `opaque/api/__init__.py`, or
   `opaque/api/accounting/__init__.py` — those three are pure PEP
   420 implicit namespaces; multiple wheels contribute to them, and
   shipping an `__init__.py` at any of those paths breaks
   sibling-wheel imports. The CI guard
   `tests/contracts/test_pep420_no_init.py` enforces this.

## Test layout

In-wheel tests under `packages/opaque-lipschitz/tests/`. Cross-wheel
integration tests (e.g. lipschitz × patches × DDP, where neither
`opaque-lipschitz` nor `opaque-patches` depends on the other) go
under `tests/integration/`. The contract test
`tests/contracts/test_test_placement.py` enforces the dep cone.

## Worked example: a tiny noise mechanism

A complete walkthrough — registering a new "scaled-Laplace" noise
mechanism for DP-SGD-flavour training — lives in
[`docs/tutorials/extending_opaque.ipynb`](../tutorials/README.md).
That notebook builds the wheel skeleton, the impl, the registrations,
and the façade end-to-end.

## See also

- [Serialization registry](serialization.md)
- [Distributed sync](distributed-sync.md)
- [Clipping `fun` helpers](clipping-fun.md)
- [`docs/tutorials/extending_opaque.ipynb`](../tutorials/README.md) —
  hands-on contributor walkthrough.
