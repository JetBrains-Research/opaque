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

### Auto-registration via `DpProcess.__init_subclass__`

If your wheel adds a new accounting primitive, you'll subclass
`opaque.api.accounting.core._base.DpProcess`. Today that base class
defines an `__init_subclass__` hook (see `_base.py:90` and the
`_PROCESS_REGISTRY` dict at `_base.py:50`) that inserts every concrete
subclass into a process registry at class-definition time. The
practical effect is that the subclass appears in the registry as
soon as its defining module is imported — no explicit `register(...)`
call is needed for the *process* itself. Serializer registration
(item 1 above) is still explicit, and is what makes a concrete
process round-trip through `state_dict` — see
`opaque.api.accounting.core._serialization` for the pattern Opaque's
built-in processes follow.

Things to know:

- The registry is keyed on `cls.__name__`. Two processes with the
  same class name (across wheels) will collide; prefix your names
  to avoid that (`LipschitzGaussian`, not just `Gaussian`).
- This only applies to subclasses of `DpProcess`. Noise mechanisms,
  clipping rules, and sensitivity oracles aren't auto-registered;
  they're discovered by the user importing your façade.
- The registry shape is an internal detail — read from it through
  the helpers in `opaque.api.accounting.core._process_flat` rather
  than poking at `_PROCESS_REGISTRY` directly.

## Composition — what your code needs to emit today

The rest of Opaque (noise mechanisms, optimisers, second-moment
machinery, MF strategies) consumes a small set of types defined in
`opaque/api/engine/types.py`: `ClippedPytree`, `NoisedPytree`, and
`PerGroup`. If your extension produces something a downstream
component needs to act on, today it needs to emit one of these.

The most common shape: a clipping rule or a sensitivity oracle
returns `ClippedPytree(pytree=…, max_norm=R)` where `max_norm` is
either a Python `float` or a `PerGroup`. From the rest of the
pipeline's perspective, that's *all* it sees — the route the bound
was derived from (per-sample norm computation, AUTO-S scaling, a
Lipschitz constant baked into the architecture) doesn't matter at
this seam.

DP-FTRL adds one extra thing-to-know on top: the MF noise dispatcher
asserts that `max_norm` is constant across calls when an MF mechanism
is downstream. The check lives in
`opaque/api/dpftrl/noise/_dispatcher.py:251`
(`_validate_constant_max_norm`); if your clipping rule's `max_norm`
varies per step, today the MF mechanism will raise rather than
silently lose privacy. AUTO-S and fixed clipping already pass this;
adaptive clipping does not, by construction. A sensitivity-oracle
extension typically emits a fixed `max_norm` from the architecture
and so passes this naturally.

See [Composition](composition.md) for the longer treatment —
`PerGroup` semantics, MSE-optimal Mahalanobis allocation, the
tripwires to avoid.

## When you don't need a new accounting primitive

A new mechanism family doesn't always need a new
`DpProcess` subclass. The three-checkbox rule, if you can tick all
three, today you can reuse the existing primitives:

1. Your output is `NoisedPytree` with `noise_stddev` set by a
   standard Gaussian mechanism (or by a matrix-factorisation
   mechanism for DP-FTRL).
2. The relationship between `noise_multiplier` and per-record
   sensitivity is the standard one — i.e. `σ = noise_multiplier ·
   max_norm` for Gaussian, or the analogous expression for MF.
3. `max_norm` is constant across the run (per Composition above).

If all three hold, reuse `opaque.dpsgd.accounting.gaussian(nm)` for
DP-SGD or — for DP-FTRL — pick the strategy-specific factory that
matches your mechanism (`opaque.dpftrl.accounting.band_mf`,
`blt`, `bisr`, `bsr`, `identity_mf`, or `lambda_cgd`) plus the
matching amplification factory (`poisson`, `b_min_sep`, or
`balls_in_bins`), rather than defining a new process.
Most *clipping-rule* and *sensitivity-oracle* extensions land here:
the privacy bookkeeping is the same Gaussian / MF story, only the
sensitivity source changes. As an illustration, a Lipschitz-layer
wheel — where `max_norm` comes from the architecture rather than
from a norm computation — would typically reuse
`accounting.gaussian` and not ship its own `DpProcess`.

A new primitive is the right call when your *mechanism* itself is
new: a non-Gaussian noise family, a privacy-amplification analysis
that doesn't fit the existing accountants, a new participation model.

## Telemetry — the `*Aux` convention

Opaque's clipping primitives return an *aux* dataclass alongside
the privatised output: a small immutable record of what happened
this step. Today there are five in-tree:

| Aux type | Returned by | Where |
|---|---|---|
| `ClippedFunAux` | `clipped_fun` | `opaque/api/engine/clipping/_clipped_fun.py:35` |
| `ClippedGradAux` | `clipped_grad` | `opaque/api/engine/clipping/_clipped_grad.py:25` |
| `AutoClippedFunAux` | `auto_clipped_fun` | `opaque/api/engine/clipping/_auto.py:47` |
| `AutoClippedGradAux` | `auto_clipped_grad` | `opaque/api/engine/clipping/_auto.py:64` |
| `AdaptiveClippedGradAux` | `adaptive_clipped_grad` | `opaque/api/dpsgd/clipping/_adaptive.py:30` |

The convention they share, and which a new extension is expected
to follow:

- The Aux is a frozen dataclass, returned by value alongside (not
  in place of) the privatised output. Diagnostics and telemetry
  belong here; never on the `ClippedPytree` itself, whose metadata
  is part of the privacy contract.
- For per-sample clipping, the field names follow a small uniform
  convention that downstream telemetry can consume across mechanism
  families: gradient-flavoured Aux types (`ClippedGradAux` and
  subclasses) use `grad_norms` / `clipped_grad_norms` for the
  per-example pre- and post-clip L2 norms; function-flavoured Aux
  types (`ClippedFunAux` and subclasses) use `norms` /
  `clipped_norms`. Both flavours share `clipping_rate` and the
  per-group breakdown `group_norms` where applicable.
- If the Aux needs cross-rank reduction (norms averaged across
  ranks for logging), register a sync handler keyed on the Aux
  type itself — see
  `opaque/api/engine/clipping/_distributed.py`. The sync handler
  is independent of the state-class sync handler.
- A mechanism without per-sample structure (e.g. a layer-by-layer
  Lipschitz oracle) doesn't need to follow the clip-stats field
  names; it should still emit *some* Aux record so downstream tools
  have a uniform place to read per-step telemetry from.

## Test layout

In-wheel tests under `packages/opaque-lipschitz/tests/`. Cross-wheel
integration tests (e.g. lipschitz × patches × DDP, where neither
`opaque-lipschitz` nor `opaque-patches` depends on the other) go
under `tests/integration/`. The contract test
`tests/contracts/test_test_placement.py` enforces the dep cone.

## Worked example: a tiny noise mechanism

A walk-through of registering a *scaled-Laplace* noise mechanism for
DP-SGD-flavour training. The mechanism itself is contrived — Laplace
noise on summed clipped gradients isn't a typical DP-SGD pattern —
but it exercises every seam a real extension touches: a state class,
a serializer, a sync handler, a façade re-export, an accounting
factory.

We'll pretend the wheel is `opaque-laplace` with contrib root
`opaque.api.laplace.*`.

### 1. The state class

```python
# opaque/api/laplace/noise/types.py
from dataclasses import dataclass

from opaque.api.engine.types import NoiseState
from opaque.api.engine.random.types import RngKey


@dataclass(frozen=True)
class LaplaceNoiseState(NoiseState):
    """Per-step state for the scaled-Laplace mechanism."""
    _step_counter: int
    _rng_key: RngKey
    scale: float
```

### 2. The noise impl

```python
# opaque/api/laplace/noise/_engine.py
from dataclasses import replace

from opaque.api.engine.types import ClippedPytree, NoisedPytree
from opaque.api.engine.random import split

from .types import LaplaceNoiseState


def laplace_noise(*, scale: float, key) -> tuple[callable, LaplaceNoiseState]:
    state = LaplaceNoiseState(_step_counter=0, _rng_key=key, scale=scale)

    def noise_fn(clipped: ClippedPytree, st: LaplaceNoiseState):
        step_key, next_key = split(st._rng_key)
        # ... draw Laplace(0, scale * clipped.max_norm) per leaf ...
        noisy_pytree = _add_laplace(clipped.pytree, step_key, st.scale, clipped.max_norm)
        out = NoisedPytree(
            pytree=noisy_pytree,
            max_norm=clipped.max_norm,
            noise_stddev=None,  # Laplace doesn't have a Gaussian σ; field stays None
        )
        new_state = replace(st, _step_counter=st._step_counter + 1, _rng_key=next_key)
        return out, new_state

    return noise_fn, state
```

The returned `NoisedPytree` carries the original `max_norm` (the
sensitivity bound) through; downstream optimisers and aggregators
treat it the same as any other noised output.

### 3. Serializer + sync handler

```python
# opaque/api/laplace/noise/_serialization.py
from opaque.api.base.serialization import register_serializer
from .types import LaplaceNoiseState


def _dump(s): return {"type": "LaplaceNoiseState/1", "step": s._step_counter,
                      "scale": s.scale, "rng_key": s._rng_key.to_state_dict()}
def _load(_tpl, sd): ...  # rebuild from the dict


register_serializer(LaplaceNoiseState, _dump, _load)
```

```python
# opaque/api/laplace/noise/_distributed.py
from opaque.api.engine.distributed import register_sync_type
from opaque.distributed import is_distributed
from .types import LaplaceNoiseState


def _sync(state: LaplaceNoiseState) -> LaplaceNoiseState:
    if not is_distributed():
        return state
    # ... reduce step counter / re-seed key across ranks ...
    return state


register_sync_type(LaplaceNoiseState, _sync)
```

Both are loaded as side-effect imports from
`opaque/api/laplace/noise/__init__.py`:

```python
from ._engine import laplace_noise
from . import _serialization  # noqa: F401  — registers serializer
from . import _distributed    # noqa: F401  — registers sync handler

__all__ = ["laplace_noise"]
```

### 4. The façade

```python
# opaque/laplace/noise/__init__.py
"""Laplace noise mechanism."""

from opaque.api.laplace.noise import laplace_noise

__all__ = ["laplace_noise"]
```

That's the whole user-facing surface. Façade-discipline rules
(see above) keep it to a re-export.

### 5. Accounting

If the Laplace mechanism warrants its own accounting story — and it
likely does, since Laplace is pure-DP rather than approximate-DP —
you'd define a `LaplaceProcess(DpProcess)` under
`opaque.api.accounting.laplace.mechanisms`, register its serializer,
and expose it via `opaque.laplace.accounting.laplace(scale)`. The
`__init_subclass__` machinery picks the class up automatically (see
"Auto-registration" above), but the dump/load functions still need
explicit registration.

If you're instead building a *sensitivity-oracle* extension —
Lipschitz layers being the canonical example — your "mechanism"
doesn't change the noise primitive; the accounting story stays
Gaussian (or MF) and you can skip this step entirely. See "When you
don't need a new accounting primitive" above.

## See also

- [Composition](composition.md) — what your code needs to emit so
  it composes with the rest of Opaque.
- [Upstream integration](upstream-integration.md) — reuse vs extend
  vs rewrite when an upstream library is in the picture.
- [Serialization registry](serialization.md)
- [Distributed sync](distributed-sync.md)
- [Clipping `fun` helpers](clipping-fun.md)
