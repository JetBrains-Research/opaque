# Plan: Opaque package split — wheels, `opaque.api`, façades, and per-package layout

This plan supersedes prior drafts. It is the **single proposal** for how
`opaque-core` is decomposed, how `opaque.api.*` (PEP 420 namespace) carries
implementation, how user-facing façades re-export from it, and the **folder
layout, modules, and `__all__`** for every package, subpackage, and module that
results.

## 0. Goals (recap)

1. **No combinatorial explosion:** DP-SGD and DP-FTRL stay independently
   installable; their incompatible mechanisms / samplers do not mix at the
   import level.
2. **Accounting math split (per-step vs whole-process) stays;** the **user UX
   does not split** into competing public surfaces (the original confusion).
3. **`opaque-core` becomes a real foundation** — narrow contracts and
   registration hubs — and the rest moves to a small handful of focused wheels.
4. **`opaque.api` is a PEP 420 namespace** that **multiple wheels contribute
   to**, with longer paths that **expose package structure**; **`opaque.dpsgd` /
   `opaque.dpftrl` / `opaque.accounting` / …** are **thin re-export façades**
   (small `__init__.py` files) that hide depth for everyday users.
5. **Lipschitz / future stacks** plug in by adding `opaque-lipschitz` →
   `opaque.api.lipschitz.*` and an `opaque.lipschitz` façade — no foundation
   changes needed.

## 1. Final wheel set

| Wheel | Ships these import roots | Depends on |
|-------|-------------------------|------------|
| **`opaque-base`** | `opaque.api.base.*`; façades `opaque.types`, `opaque.pytree`, `opaque.random`, `opaque.serialization`, `opaque.distributed` | torch, numpy, optree |
| **`opaque-engine`** | `opaque.api.engine.*`; façades `opaque.functional`, `opaque.scheduling`, `opaque.profiling` | `opaque-base`, torch |
| **`opaque-optimizers`** | `opaque.api.optimizers.*`; façade `opaque.optimizers` | `opaque-base`, torchopt |
| **`opaque-accounting`** | `opaque.api.accounting.core.*` (+ Rust ext); façade `opaque.accounting` | `opaque-base`, `opaque_accounting` (Rust) |
| **`opaque-dpsgd`** | `opaque.api.dpsgd.*`, `opaque.api.accounting.dpsgd.*`; façade `opaque.dpsgd` (incl. `opaque.dpsgd.accounting`) | `opaque-base`, `opaque-engine`, `opaque-accounting`; extra `[optimizers]` → `opaque-optimizers` |
| **`opaque-dpftrl`** | `opaque.api.dpftrl.*`, `opaque.api.accounting.dpftrl.*`; façade `opaque.dpftrl` (incl. `opaque.dpftrl.accounting`) | `opaque-base`, `opaque-engine`, `opaque-accounting`; extra `[optimizers]` → `opaque-optimizers` |
| **`opaque-auditing`** | `opaque.api.auditing.*`; façade `opaque.auditing` | `opaque-base`, `opaque-accounting` |
| **`opaque-patches`** | `opaque.api.patches.*`; façades `opaque.patches`, `opaque.transformers`, `opaque.performance` (as today) | `opaque-base`, `opaque-engine`, optional `transformers`, `peft`, `triton` extras |
| **`opaque` (umbrella, optional)** | nothing of its own | bundle pin: `opaque-base`, `opaque-engine`, `opaque-dpsgd`, `opaque-dpftrl`, `opaque-accounting`, `opaque-auditing`, `opaque-optimizers` |

There is **no separate** `opaque-accounting-core` wheel: shared accounting
implementation lives in `opaque.api.accounting.core` inside the
`opaque-accounting` wheel; per-stack factories live in the matching stack
wheel under `opaque.api.accounting.{dpsgd,dpftrl}`.

## 2. PEP 420 invariants (CI-enforced)

The following directories are **PEP 420 implicit namespaces**. **No** wheel
ships an `__init__.py` for them:

- `opaque/`
- `opaque/api/`
- `opaque/api/accounting/`

Any other directory may have a real `__init__.py`. CI shell guard already exists
for `opaque/__init__.py`; this plan extends the same script to the two new
namespace roots above.

## 3. Façade-vs-impl convention

For every public concept:

- **Implementation** lives under **`opaque.api.<contrib>.<concern>...`** in the
  wheel that owns it. Internal modules use a leading underscore
  (`_gaussian.py`, `_helpers.py`, …).
- **Public façade** is **one** `opaque.<concern>` (or
  `opaque.<stack>.<concern>`) module whose `__init__.py` is a **small block of
  re-exports** plus a tight `__all__`.
- Façades **never** define new behaviour — they only `from opaque.api… import
  …` and append to `__all__`. Tests live next to implementation under
  `opaque.api.*`; façade tests assert that re-exports stay in sync.
- **Subpackage `__all__` rule:** every `__init__.py` declares `__all__`. Every
  multi-symbol `_*.py` module also declares `__all__` (single-class modules may
  omit it). `types.py` modules always declare `__all__` (they are the public
  type contract for that subpackage).

Two **wheels of the same façade root are forbidden:** only one wheel ships
`opaque.dpsgd/__init__.py` (i.e., `opaque-dpsgd`); only one wheel ships
`opaque.accounting/__init__.py` (i.e., `opaque-accounting`); etc. Façades are
**not** PEP 420 namespaces.

---

## 4. `opaque-base` — folder plan

### 4.1 `opaque/api/base/` (implementation)

```
opaque/api/base/
├── types/
│   ├── __init__.py          # public types contract for opaque.api.base.types
│   ├── _pytree.py           # ClippedPytree, NoisedPytree, TensorPytree
│   ├── _grouping.py         # PerGroup
│   ├── _states.py           # ClipState, NoiseState
│   └── _aux.py              # SecondMomentNoiseOutput, common aux dataclasses
├── pytree/
│   ├── __init__.py
│   └── _ops.py              # tree_map, tree_leaves, partition, merge, global_norm
├── random/
│   ├── __init__.py
│   ├── _types.py            # RngKey
│   ├── _engine.py           # key, split, fold_in
│   └── _helpers.py          # to_generator, with_torch_generator
├── serialization/
│   ├── __init__.py
│   ├── _types.py            # SerializedState, Serializer Protocol
│   ├── _registry.py         # register_serializer, lookup_serializer
│   ├── _structural.py       # state_dict / from_state_dict scaffolding
│   └── _compat.py           # version-tag helpers
├── distributed/
│   ├── __init__.py
│   ├── _types.py            # SyncHandler Protocol
│   ├── _registry.py         # register_sync_handler, lookup_sync_handler
│   ├── _sync.py             # sync(obj) dispatcher
│   ├── _collectives.py      # all_reduce, all_reduce_, barrier
│   ├── _gradients.py        # reduce_pytree, sum_gradients, sum_gradients_
│   ├── _shard.py            # local_shard, _local_shard_bounds
│   └── _detection.py        # is_distributed, get_rank, get_world_size
└── _noise_allocation.py
```

`__all__` per file:

- `opaque/api/base/types/__init__.py`:
  ```python
  __all__ = [
      "ClippedPytree", "NoisedPytree", "TensorPytree",
      "PerGroup", "ClipState", "NoiseState",
      "SecondMomentNoiseOutput",
  ]
  ```
- `opaque/api/base/pytree/__init__.py`:
  ```python
  __all__ = ["tree_map", "tree_leaves", "partition", "merge", "global_norm"]
  ```
- `opaque/api/base/random/__init__.py`:
  ```python
  __all__ = ["RngKey", "key", "split", "fold_in", "to_generator", "with_torch_generator"]
  ```
- `opaque/api/base/serialization/__init__.py`:
  ```python
  __all__ = [
      "Serializer", "register_serializer", "lookup_serializer",
      "state_dict", "from_state_dict", "SerializedState",
  ]
  ```
- `opaque/api/base/distributed/__init__.py`:
  ```python
  __all__ = [
      "SyncHandler", "register_sync_handler", "lookup_sync_handler",
      "sync", "is_distributed", "get_rank", "get_world_size",
      "all_reduce", "all_reduce_", "barrier",
      "reduce_pytree", "reduce_pytree_", "sum_gradients", "sum_gradients_",
      "local_shard",
  ]
  ```

`_*.py` modules each declare a local `__all__` of the names they export to
their package `__init__.py` (e.g. `_collectives.py` exports `__all__ =
["all_reduce", "all_reduce_", "barrier"]`). `_noise_allocation.py` exports
`__all__ = ["allocate_noise"]` (or whatever its current public name is).

### 4.2 Façade modules (`opaque-base` wheel)

```
opaque/
├── types.py                 # re-exports opaque.api.base.types.*
├── pytree.py                # re-exports opaque.api.base.pytree.*
├── random/__init__.py       # re-exports opaque.api.base.random.*
├── serialization/__init__.py# re-exports user-facing serialization API
└── distributed/__init__.py  # re-exports user-facing distributed API
```

Façade `__all__` examples:

- `opaque/types.py`:
  ```python
  from opaque.api.base.types import (
      ClippedPytree, NoisedPytree, TensorPytree,
      PerGroup, ClipState, NoiseState, SecondMomentNoiseOutput,
  )

  __all__ = [
      "ClippedPytree", "NoisedPytree", "TensorPytree",
      "PerGroup", "ClipState", "NoiseState", "SecondMomentNoiseOutput",
  ]
  ```
- `opaque/serialization/__init__.py` re-exports **only** `state_dict`,
  `from_state_dict`, `SerializedState` (user-facing). Registry helpers
  (`register_serializer`, `Serializer`) stay in `opaque.api.base.serialization`
  for power users.
- `opaque/distributed/__init__.py` re-exports `sync`, `is_distributed`,
  `get_rank`, `get_world_size`, `all_reduce`, `reduce_pytree`,
  `sum_gradients`, `local_shard`. `register_sync_handler` and `SyncHandler` stay
  in `opaque.api.base.distributed`.

---

## 5. `opaque-engine` — folder plan

### 5.1 `opaque/api/engine/` (implementation)

```
opaque/api/engine/
├── functional/
│   ├── __init__.py
│   ├── _make_functional.py  # make_functional, _set_module_params
│   ├── _batch_dim.py        # with_batch_dim
│   └── _collate.py          # empty_collate (PoissonSubsampler / CyclicPoissonSampler glue)
├── clipping/
│   ├── __init__.py
│   ├── types.py             # ClipState public contract; PerGroupAux
│   ├── fun.py               # power-user functional surface (auto_clipped_fun, ...)
│   ├── _clipped_grad.py     # clipped_grad (fixed threshold)
│   ├── _auto.py             # auto_clipped_grad (AUTO-S)
│   ├── _per_group.py        # per_group helper
│   ├── _clipped_fun.py      # internal lower-level fun helpers
│   ├── _helpers.py          # private helpers used across clip implementations
│   ├── _pytree.py           # internal pytree shaping for clipping
│   └── _distributed.py      # registers ClipState sync handler at import
├── scheduling/
│   ├── __init__.py
│   ├── types.py             # Schedule, ScheduleState
│   ├── _curves.py           # constant_schedule, cosine_schedule, inverse_sqrt_schedule, ...
│   └── _compose.py          # with_warmup, with_restarts
└── profiling/
    ├── __init__.py
    ├── types.py             # StepRecord
    ├── _memory.py           # memory measurement helpers
    └── _distributed.py      # rank-aware profiler bits
```

`__all__` per `__init__.py`:

- `opaque/api/engine/functional/__init__.py`:
  ```python
  __all__ = ["make_functional", "with_batch_dim", "empty_collate"]
  ```
- `opaque/api/engine/clipping/__init__.py`:
  ```python
  __all__ = ["clipped_grad", "auto_clipped_grad", "per_group"]
  ```
- `opaque/api/engine/clipping/types.py`:
  ```python
  __all__ = ["ClipState", "PerGroupAux"]
  ```
- `opaque/api/engine/clipping/fun.py`:
  ```python
  __all__ = ["auto_clipped_fun", "clipped_fun"]
  ```
- `opaque/api/engine/scheduling/__init__.py`:
  ```python
  __all__ = [
      "constant_schedule", "cosine_schedule", "inverse_sqrt_schedule",
      "one_minus_sqrt_schedule",
      "with_warmup", "with_restarts",
  ]
  ```
- `opaque/api/engine/scheduling/types.py`: `__all__ = ["Schedule", "ScheduleState"]`
- `opaque/api/engine/profiling/__init__.py`:
  ```python
  __all__ = ["TrainingProfiler", "StepTimer", "StepRecord"]
  ```

### 5.2 Façades (`opaque-engine` wheel)

```
opaque/
├── functional/__init__.py    # re-exports opaque.api.engine.functional.*
├── scheduling/__init__.py    # re-exports opaque.api.engine.scheduling.*
└── profiling/__init__.py     # re-exports opaque.api.engine.profiling.*
```

Each façade copies the same `__all__` as its impl `__init__.py`. **Clipping has
no top-level façade** (`opaque.clipping` does not exist) — clipping is
always imported through `opaque.dpsgd.clipping` / `opaque.dpftrl.clipping`.
This is the post-refactor ADR rule.

---

## 6. `opaque-optimizers` — folder plan

### 6.1 `opaque/api/optimizers/`

```
opaque/api/optimizers/
├── __init__.py              # all factory names
├── types.py                 # state dataclasses for non-trivial optimizers
├── _chain.py                # make_optimizer_chain (DP-aware chain wrapper)
├── _bias_correction.py      # is_per_group, resolve_noise_variance, ...
├── _sgd.py                  # sgd
├── _adam.py                 # adam, adamw
├── _adagrad.py              # adagrad
├── _adafactor.py            # adafactor
├── _adadelta.py             # adadelta (vanilla torchopt re-export)
├── _radam.py                # radam (vanilla torchopt re-export)
├── _rmsprop.py              # rmsprop
├── _lion.py                 # lion
├── _ademamix.py             # ademamix
└── _schedule_free.py        # schedule_free wrapper
```

`__all__`:

- `opaque/api/optimizers/__init__.py`:
  ```python
  __all__ = [
      "sgd", "adam", "adamw", "adagrad", "adafactor", "adadelta",
      "radam", "rmsprop", "lion", "ademamix", "schedule_free",
  ]
  ```
- `opaque/api/optimizers/types.py`:
  ```python
  __all__ = [
      "AdamState", "AdamWState", "AdagradState", "AdafactorState",
      "AdadeltaState", "RAdamState", "RMSpropState", "LionState",
      "AdEMAMixState", "ScheduleFreeState",
  ]
  ```
- `_chain.py`: `__all__ = ["make_optimizer_chain"]`
- `_bias_correction.py`: `__all__ = ["is_per_group", "resolve_noise_variance"]`
- Each `_<name>.py` module: `__all__ = ["<name>"]` (or `[..., "<name>w"]` for adam/adamw).

### 6.2 Façade

```
opaque/
└── optimizers/__init__.py   # re-exports opaque.api.optimizers.*
```

Façade `__all__` matches the impl package; `types.py` re-export stays at
`opaque.optimizers.types` for stability.

---

## 7. `opaque-accounting` — folder plan

### 7.1 `opaque/api/accounting/core/` (impl + Rust binding)

```
opaque/api/accounting/core/
├── __init__.py              # core public surface (composition algebra)
├── types.py                 # public dataclasses: DpProcess, Pld, Budget, ...
├── discretization.py        # get_discretization, default config
├── calibration.py           # calibrate, epsilon_budget, delta_budget, ...
├── _native.py               # `from . import opaque_accounting as _native` re-aliasing
├── _base.py                 # DpProcess, Pld base classes
├── _accountant.py           # Accountant (state machine)
├── _budgets.py              # epsilon_budget, delta_budget, advantage_budget, ...
├── _process_flat.py         # flat composition operators (`*`, `|`, ...)
├── _serialization.py        # registers tag/load/dump for accounting types
├── composition/
│   ├── __init__.py
│   ├── types.py             # Composition / Composed / Repeated / Cached types
│   ├── _composed.py
│   ├── _repeated.py
│   └── _cached.py
├── mechanisms/
│   ├── __init__.py
│   ├── types.py
│   ├── _identity.py         # identity (composition algebra identity ≈ ε=0)
│   ├── _eps_delta.py        # eps_delta mechanism
│   └── _nonprivate.py       # nonprivate (∞)
├── transformations/
│   ├── __init__.py
│   └── types.py
├── amplification/
│   ├── __init__.py          # placeholder (per-stack amplification lives elsewhere)
│   └── types.py             # generic amplification protocol types
└── opaque_accounting.abi3.so  # Rust PyO3 extension (built by maturin)
```

`__all__`:

- `opaque/api/accounting/core/__init__.py`:
  ```python
  __all__ = [
      "DpProcess", "Pld", "Accountant",
      "calibrate",
      "epsilon_budget", "delta_budget", "advantage_budget",
      "beta_budget", "risk_budget",
      "identity", "eps_delta", "nonprivate",
      "get_discretization",
  ]
  ```
- `opaque/api/accounting/core/types.py`:
  ```python
  __all__ = [
      "DpProcess", "Pld", "Accountant", "Budget",
      "Discretization",
  ]
  ```
- `opaque/api/accounting/core/composition/__init__.py`: `__all__ = ["Composed", "Repeated", "Cached"]`
- `opaque/api/accounting/core/mechanisms/__init__.py`: `__all__ = ["identity", "eps_delta", "nonprivate"]`
- `opaque/api/accounting/core/calibration.py`: `__all__ = ["calibrate"]`
- `opaque/api/accounting/core/discretization.py`: `__all__ = ["get_discretization", "Discretization"]`
- `_native.py`: `__all__ = ["_native"]`

### 7.2 Façade — `opaque/accounting/`

The façade **mirrors the impl tree** the same way `opaque/dpsgd/accounting/` /
`opaque/dpftrl/accounting/` mirror their impl trees: subpackages are exposed in
the same shape; the root `__init__.py` is a small convenience layer that
re-exports the algebra entry points and keeps submodules accessible.

```
opaque/
└── accounting/
    ├── __init__.py
    ├── types.py
    ├── calibration.py
    ├── discretization.py
    ├── composition/
    │   ├── __init__.py
    │   └── types.py
    ├── mechanisms/
    │   ├── __init__.py
    │   └── types.py
    ├── transformations/
    │   ├── __init__.py
    │   └── types.py
    └── amplification/
        ├── __init__.py
        └── types.py
```

`opaque/accounting/__init__.py`:

```python
from opaque.accounting import (
    composition, mechanisms, transformations, amplification,
)
from opaque.api.accounting.core import (
    DpProcess, Pld, Accountant,
    calibrate,
    epsilon_budget, delta_budget, advantage_budget,
    beta_budget, risk_budget,
    get_discretization,
)

__all__ = [
    "composition", "mechanisms", "transformations", "amplification",
    "DpProcess", "Pld", "Accountant",
    "calibrate",
    "epsilon_budget", "delta_budget", "advantage_budget",
    "beta_budget", "risk_budget",
    "get_discretization",
]
```

`opaque/accounting/types.py`:

```python
from opaque.api.accounting.core.types import (
    DpProcess, Pld, Accountant, Budget, Discretization,
)
__all__ = ["DpProcess", "Pld", "Accountant", "Budget", "Discretization"]
```

`opaque/accounting/calibration.py`:

```python
from opaque.api.accounting.core.calibration import calibrate
__all__ = ["calibrate"]
```

`opaque/accounting/discretization.py`:

```python
from opaque.api.accounting.core.discretization import (
    get_discretization, Discretization,
)
__all__ = ["get_discretization", "Discretization"]
```

`opaque/accounting/mechanisms/__init__.py`:

```python
from opaque.api.accounting.core.mechanisms import (
    identity, eps_delta, nonprivate,
)
__all__ = ["identity", "eps_delta", "nonprivate"]
```

`opaque/accounting/mechanisms/types.py`:

```python
from opaque.api.accounting.core.mechanisms.types import *  # noqa: F401,F403
from opaque.api.accounting.core.mechanisms.types import __all__ as __all__
```

`opaque/accounting/composition/__init__.py`:

```python
from opaque.api.accounting.core.composition import Composed, Repeated, Cached
__all__ = ["Composed", "Repeated", "Cached"]
```

`opaque/accounting/composition/types.py` re-exports from
`opaque.api.accounting.core.composition.types` with the same `__all__` policy.

`opaque/accounting/transformations/__init__.py` and
`opaque/accounting/amplification/__init__.py` re-export their respective
`opaque.api.accounting.core.transformations` /
`opaque.api.accounting.core.amplification` surfaces (today these are
mostly type / protocol holders; if they grow, the façade grows symmetrically).
Each declares an explicit `__all__`.

**`opaque.accounting` does NOT re-export `gaussian` / `band_mf` / `poisson`
/ etc.** Those live only on `opaque.dpsgd.accounting` and
`opaque.dpftrl.accounting`. The mechanisms exposed here
(`identity`, `eps_delta`, `nonprivate`) are **algebra primitives**, not
DP-SGD- or DP-FTRL-specific factories — keeping them under the same
`opaque.accounting.mechanisms` submodule that exists in the impl tree
preserves §0’s “no parallel factory namespace” rule while matching the
shape of the other façades.

---

## 8. `opaque-dpsgd` — folder plan

### 8.1 `opaque/api/dpsgd/` (training impl)

```
opaque/api/dpsgd/
├── __init__.py              # __all__ = [] (impl tree; users go through facade)
├── types.py                 # DP-SGD state dataclasses
├── clipping/
│   ├── __init__.py
│   ├── types.py             # AdaptiveClipState
│   └── _adaptive.py         # adaptive_clipped_grad
│   # Fixed / AUTO-S / per_group are NOT duplicated here — the dpsgd facade
│   # re-exports them from opaque.api.engine.clipping.
├── noise/
│   ├── __init__.py
│   ├── types.py             # GaussianNoiseState (shared by both noise functions)
│   ├── _gaussian.py         # gaussian_noise (defines GaussianNoiseState)
│   ├── _truncated_gaussian.py # truncated_gaussian_noise (reuses GaussianNoiseState)
│   └── _distributed.py      # registers noise-state sync handlers
└── sampling/
    ├── __init__.py
    ├── _poisson.py          # PoissonSubsampler
    └── _helpers.py          # _maybe_truncate_indices, _plain_poisson_step_indices
```

`__all__`:

- `opaque/api/dpsgd/__init__.py`: `__all__ = []` (this tree is browsed by power
  users; everyday imports go through the façade).
- `opaque/api/dpsgd/clipping/__init__.py`: `__all__ = ["adaptive_clipped_grad"]`
- `opaque/api/dpsgd/clipping/types.py`: `__all__ = ["AdaptiveClipState"]`
- `opaque/api/dpsgd/noise/__init__.py`:
  ```python
  __all__ = ["gaussian_noise", "truncated_gaussian_noise"]
  ```
- `opaque/api/dpsgd/noise/types.py`:
  ```python
  __all__ = ["GaussianNoiseState"]
  ```
  (Both `gaussian_noise()` and `truncated_gaussian_noise()` return the **same**
  `GaussianNoiseState`; per-group noise is realized by passing a `PerGroup`
  `max_norm` on the `ClippedPytree` input — there is no separate noise state
  type for it.)
- `opaque/api/dpsgd/sampling/__init__.py`: `__all__ = ["PoissonSubsampler"]`

### 8.2 `opaque/api/accounting/dpsgd/` (DP-SGD accounting factories)

```
opaque/api/accounting/dpsgd/
├── __init__.py              # public DP-SGD factories
├── types.py                 # Gaussian, Adaclip, Poisson, ParallelPoisson dataclasses
├── mechanisms/
│   ├── __init__.py          # __all__ = ["gaussian", "adaclip"]
│   ├── types.py             # __all__ = ["Gaussian", "Adaclip"]
│   ├── _gaussian.py
│   └── _adaclip.py
└── amplification/
    ├── __init__.py          # __all__ = ["poisson", "parallel_poisson"]
    ├── types.py             # __all__ = ["Poisson", "ParallelPoisson"]
    ├── _poisson.py
    └── _parallel_poisson.py
```

`opaque/api/accounting/dpsgd/__init__.py`:

```python
from opaque.api.accounting.dpsgd.mechanisms import gaussian, adaclip
from opaque.api.accounting.dpsgd.amplification import poisson, parallel_poisson

__all__ = ["gaussian", "adaclip", "poisson", "parallel_poisson"]
```

### 8.3 Façade — `opaque/dpsgd/`

```
opaque/
└── dpsgd/
    ├── __init__.py          # convenience root re-exports (small)
    ├── types.py             # DP-SGD-specific public types
    ├── clipping/
    │   ├── __init__.py      # ALL clipping the DP-SGD user needs (engine + dpsgd-only)
    │   ├── types.py
    │   └── fun.py
    ├── noise/
    │   ├── __init__.py
    │   └── types.py
    ├── sampling/
    │   ├── __init__.py
    │   └── types.py
    └── accounting/
        ├── __init__.py      # __all__ = ["gaussian", "adaclip", "poisson", "parallel_poisson"]
        ├── types.py
        ├── mechanisms/__init__.py
        └── amplification/__init__.py
```

`opaque/dpsgd/clipping/__init__.py` (the “self-sufficient” story):

```python
from opaque.api.engine.clipping import clipped_grad, auto_clipped_grad, per_group
from opaque.api.dpsgd.clipping import adaptive_clipped_grad

__all__ = [
    "clipped_grad", "auto_clipped_grad", "per_group",
    "adaptive_clipped_grad",
]
```

`opaque/dpsgd/clipping/types.py`:

```python
from opaque.api.engine.clipping.types import ClipState
from opaque.api.dpsgd.clipping.types import AdaptiveClipState

__all__ = ["ClipState", "AdaptiveClipState"]
```

`opaque/dpsgd/clipping/fun.py`: `from opaque.api.engine.clipping.fun import *`
with `__all__ = [...]` mirrored explicitly.

`opaque/dpsgd/noise/__init__.py`:

```python
from opaque.api.dpsgd.noise import gaussian_noise, truncated_gaussian_noise

__all__ = ["gaussian_noise", "truncated_gaussian_noise"]
```

`opaque/dpsgd/sampling/__init__.py`:

```python
from opaque.api.dpsgd.sampling import PoissonSubsampler
__all__ = ["PoissonSubsampler"]
```

`opaque/dpsgd/accounting/__init__.py`:

```python
from opaque.api.accounting.dpsgd import gaussian, adaclip, poisson, parallel_poisson
__all__ = ["gaussian", "adaclip", "poisson", "parallel_poisson"]
```

`opaque/dpsgd/__init__.py` (root convenience — keep narrow):

```python
from opaque.dpsgd import accounting, clipping, noise, sampling

__all__ = ["accounting", "clipping", "noise", "sampling"]
```

(Root does not re-export every leaf name — submodules are the canonical home.)

---

## 9. `opaque-dpftrl` — folder plan

### 9.1 `opaque/api/dpftrl/` (training impl)

```
opaque/api/dpftrl/
├── __init__.py              # __all__ = []
├── types.py                 # DP-FTRL state dataclasses
├── clipping/
│   └── __init__.py          # empty (`__all__ = []`); FTRL has no extra clip methods
├── noise/
│   ├── __init__.py
│   ├── types.py             # MfNoiseState, IdentityState, BandMfState, BltState, BisrState, BsrState, LambdaCgdState
│   ├── _dispatcher.py       # mf_noise dispatcher
│   ├── _engine.py           # streaming MF engine
│   ├── _identity.py         # identity_strategy
│   ├── _band_mf.py          # band_mf_strategy
│   ├── _blt.py              # blt_strategy
│   ├── _blt_math.py
│   ├── _bisr.py             # bisr_strategy
│   ├── _bsr.py              # bsr_strategy
│   ├── _lambda_cgd.py       # lambda_cgd_strategy
│   ├── _toeplitz.py         # internal Toeplitz utils
│   ├── _streaming_matrix.py # internal streaming matrix base
│   ├── _sensitivity.py      # internal sensitivity helpers
│   ├── _second_moment.py    # private second moment streams
│   ├── _checks.py           # internal validation
│   └── _distributed.py      # registers MF-noise sync handlers
└── sampling/
    ├── __init__.py
    ├── types.py             # PartitionType enum
    ├── _partitions.py       # _equal_split_partition, _independent_partition
    ├── _poisson.py          # CyclicPoissonSampler
    ├── _b_min_sep.py        # BMinSepSampler
    ├── _balls_in_bins.py    # BallsInBinsSampler
    └── _sequential.py       # SequentialBatchSampler
```

`__all__`:

- `opaque/api/dpftrl/clipping/__init__.py`: `__all__ = []`
- `opaque/api/dpftrl/noise/__init__.py`:
  ```python
  __all__ = [
      "mf_noise",
      "identity_strategy", "band_mf_strategy",
      "blt_strategy", "bisr_strategy", "bsr_strategy", "lambda_cgd_strategy",
  ]
  ```
- `opaque/api/dpftrl/noise/types.py`:
  ```python
  __all__ = [
      "MfNoiseState",
      "IdentityState", "BandMfState",
      "BltState", "BisrState", "BsrState", "LambdaCgdState",
  ]
  ```
- `opaque/api/dpftrl/sampling/__init__.py`:
  ```python
  __all__ = [
      "CyclicPoissonSampler", "BMinSepSampler",
      "BallsInBinsSampler", "SequentialBatchSampler",
  ]
  ```
- `opaque/api/dpftrl/sampling/types.py`: `__all__ = ["PartitionType"]`

### 9.2 `opaque/api/accounting/dpftrl/` (DP-FTRL accounting factories)

```
opaque/api/accounting/dpftrl/
├── __init__.py              # public DP-FTRL factories
├── types.py                 # MF mechanism dataclasses + amplification dataclasses
├── mechanisms/
│   ├── __init__.py          # __all__ = ["band_mf", "blt", "bisr", "bsr", "lambda_cgd", "mf_identity"]
│   ├── types.py             # __all__ = ["BandMf", "Blt", "Bisr", "Bsr", "LambdaCgd", "IdentityMf", "MfGaussian"]
│   ├── _band_mf.py
│   ├── _blt.py
│   ├── _bisr.py
│   ├── _bsr.py
│   ├── _lambda_cgd.py
│   ├── _identity.py
│   └── _mf_gaussian.py
└── amplification/
    ├── __init__.py          # __all__ = ["poisson", "b_min_sep", "balls_in_bins"]
    ├── types.py             # __all__ = ["MfPoisson", "BMinSep", "BallsInBins"]
    ├── _poisson.py          # MfPoisson + poisson factory
    ├── _b_min_sep.py        # BMinSep + b_min_sep factory
    ├── _b_min_sep_transcript_cache.py
    └── _balls_in_bins.py    # BallsInBins + balls_in_bins factory
```

`opaque/api/accounting/dpftrl/__init__.py`:

```python
from opaque.api.accounting.dpftrl.mechanisms import (
    band_mf, blt, bisr, bsr, lambda_cgd, mf_identity,
)
from opaque.api.accounting.dpftrl.amplification import (
    poisson, b_min_sep, balls_in_bins,
)

__all__ = [
    "band_mf", "blt", "bisr", "bsr", "lambda_cgd", "mf_identity",
    "poisson", "b_min_sep", "balls_in_bins",
]
```

### 9.3 Façade — `opaque/dpftrl/`

```
opaque/
└── dpftrl/
    ├── __init__.py          # convenience root
    ├── types.py
    ├── clipping/
    │   ├── __init__.py      # __all__ = ["clipped_grad", "auto_clipped_grad", "per_group"]
    │   ├── types.py
    │   └── fun.py
    ├── noise/
    │   ├── __init__.py
    │   └── types.py
    ├── sampling/
    │   ├── __init__.py
    │   └── types.py
    └── accounting/
        ├── __init__.py      # __all__ = ["band_mf", "blt", "bisr", "bsr", "lambda_cgd", "mf_identity", "poisson", "b_min_sep", "balls_in_bins"]
        ├── types.py
        ├── mechanisms/__init__.py
        └── amplification/__init__.py
```

`opaque/dpftrl/clipping/__init__.py`:

```python
from opaque.api.engine.clipping import clipped_grad, auto_clipped_grad, per_group

__all__ = ["clipped_grad", "auto_clipped_grad", "per_group"]
```

(No `adaptive_clipped_grad` here — DP-FTRL does not expose it; the §0 “stacks
self-sufficient” rule.)

`opaque/dpftrl/noise/__init__.py`:

```python
from opaque.api.dpftrl.noise import (
    mf_noise,
    identity_strategy, band_mf_strategy,
    blt_strategy, bisr_strategy, bsr_strategy, lambda_cgd_strategy,
)
__all__ = [
    "mf_noise",
    "identity_strategy", "band_mf_strategy",
    "blt_strategy", "bisr_strategy", "bsr_strategy", "lambda_cgd_strategy",
]
```

`opaque/dpftrl/sampling/__init__.py`:

```python
from opaque.api.dpftrl.sampling import (
    CyclicPoissonSampler, BMinSepSampler, BallsInBinsSampler, SequentialBatchSampler,
)
__all__ = [
    "CyclicPoissonSampler", "BMinSepSampler",
    "BallsInBinsSampler", "SequentialBatchSampler",
]
```

`opaque/dpftrl/accounting/__init__.py`:

```python
from opaque.api.accounting.dpftrl import (
    band_mf, blt, bisr, bsr, lambda_cgd, mf_identity,
    poisson, b_min_sep, balls_in_bins,
)
__all__ = [
    "band_mf", "blt", "bisr", "bsr", "lambda_cgd", "mf_identity",
    "poisson", "b_min_sep", "balls_in_bins",
]
```

`opaque/dpftrl/__init__.py`:

```python
from opaque.dpftrl import accounting, clipping, noise, sampling
__all__ = ["accounting", "clipping", "noise", "sampling"]
```

---

## 10. `opaque-auditing` — folder plan

### 10.1 `opaque/api/auditing/`

```
opaque/api/auditing/
├── __init__.py
├── types.py                 # AuditResult, AuditConfig, ...
├── _coin_flip.py            # coin_flip estimator
├── attacks/
│   ├── __init__.py
│   └── _loss.py             # loss-based attack
└── one_run/
    ├── __init__.py
    ├── _estimate.py
    ├── _stats.py
    └── _roc.py
```

`__all__`:

- `opaque/api/auditing/__init__.py`: `__all__ = ["coin_flip", "attacks", "one_run"]`
- `opaque/api/auditing/types.py`: `__all__ = ["AuditResult", "AuditConfig"]`
- `opaque/api/auditing/attacks/__init__.py`: `__all__ = ["loss_attack"]`
- `opaque/api/auditing/one_run/__init__.py`: `__all__ = ["one_run_estimate", "one_run_roc", "one_run_stats"]`

### 10.2 Façade

```
opaque/
└── auditing/
    ├── __init__.py          # re-exports opaque.api.auditing.*
    ├── types.py
    ├── attacks/__init__.py
    └── one_run/__init__.py
```

Same `__all__` as the impl tree.

---

## 11. `opaque-patches` — folder plan

The patches package is **not** a pure side-effect package: it ships several
**explicit entry functions** users call (`apply_model_patches`,
`apply_runtime_patches`, plus per-family `apply_*_model_patches` / runtime
`apply_*_patches`), a **family-registration / factory API** for downstream
projects with their own architectures, **kernel functions** (`opaque_swiglu`,
`opaque_rope`, …), and **PEFT / Torch checkpoint** entry points. Those have to
be reflected in the façade.

### 11.1 `opaque/api/patches/` (implementation)

Layout mirrors today's `opaque.patches.*` but lives under `opaque.api.patches`:

```
opaque/api/patches/
├── __init__.py
├── _runtime.py                # apply_runtime_patches orchestrator (was top-level patches/__init__.py)
├── _model.py                  # apply_model_patches orchestrator
├── torch/
│   ├── __init__.py
│   └── runtime.py             # apply_checkpoint_patch, is_checkpoint_patched
├── kernels/
│   ├── __init__.py
│   ├── _utils.py
│   ├── cross_entropy.py
│   ├── linear_cross_entropy.py
│   ├── swiglu.py
│   ├── geglu.py
│   ├── rope_embedding.py
│   ├── lora.py
│   ├── rms_norm.py
│   └── fused_add_rms_norm.py
├── transformers/
│   ├── __init__.py            # public factories + registry
│   ├── _factory.py            # make_apply_model_patches, register_*_kind
│   ├── _family.py             # family_name, make_apply_family_patches
│   ├── _registry.py           # register_family, supported_families
│   ├── _router.py             # apply_transformers_model_patches
│   ├── runtime/
│   │   ├── __init__.py        # apply_collator_patches, apply_masking_patches
│   │   ├── collator.py
│   │   └── masking.py
│   ├── components/
│   │   ├── __init__.py        # __all__ = []
│   │   ├── attention.py
│   │   ├── batchify.py
│   │   ├── cross_entropy.py
│   │   ├── fused_add_rms_norm.py
│   │   ├── geglu.py
│   │   ├── kv_cache.py
│   │   ├── masking.py
│   │   ├── rms_norm.py
│   │   ├── rope.py
│   │   └── swiglu.py
│   └── models/
│       ├── __init__.py        # __all__ = []  (each module registers on import)
│       ├── cohere.py
│       ├── cohere2.py
│       ├── exaone4.py
│       ├── gemma.py
│       ├── gemma2.py
│       ├── gemma3.py
│       ├── glm4.py
│       ├── gpt2.py
│       ├── granite.py
│       ├── llama.py
│       ├── ministral.py
│       ├── mistral.py
│       ├── olmo2.py
│       ├── olmo3.py
│       ├── phi3.py
│       ├── qwen2.py
│       ├── qwen3.py
│       └── smollm3.py
└── peft/
    ├── __init__.py            # apply_peft_model_patches
    ├── _router.py
    └── components/
        ├── __init__.py        # __all__ = []
        ├── _utils.py
        ├── linear.py
        ├── mlp.py
        └── qkv.py
```

`__all__` per public `__init__.py`:

- `opaque/api/patches/__init__.py`:
  ```python
  __all__ = [
      "apply_model_patches",
      "apply_runtime_patches",
      "apply_transformers_model_patches",
      "apply_peft_model_patches",
  ]
  ```
- `opaque/api/patches/torch/__init__.py`:
  ```python
  __all__ = ["apply_checkpoint_patch", "is_checkpoint_patched"]
  ```
- `opaque/api/patches/kernels/__init__.py`:
  ```python
  __all__ = [
      # Loss
      "opaque_cross_entropy_loss",
      "opaque_linear_cross_entropy_loss",
      # Activations
      "opaque_swiglu", "opaque_geglu_exact", "opaque_geglu_approx",
      "opaque_rms_norm", "opaque_fused_add_rms_norm",
      # Position embeddings
      "opaque_rope", "opaque_rope_qk", "opaque_slow_rope",
      # LoRA
      "opaque_lora_w", "opaque_lora_qkv", "opaque_lora_mlp",
      "ACTIVATION_SWIGLU", "ACTIVATION_GEGLU_EXACT", "ACTIVATION_GEGLU_APPROX",
  ]
  ```
- `opaque/api/patches/transformers/__init__.py`:
  ```python
  __all__ = [
      "apply_transformers_model_patches",
      "family_name",
      "make_apply_family_patches",
      "make_apply_model_patches",
      "register_activation_kind",
      "register_family",
      "register_fused_add_rms_kind",
      "register_rms_norm_kind",
      "supported_families",
  ]
  ```
- `opaque/api/patches/transformers/runtime/__init__.py`:
  ```python
  __all__ = ["apply_collator_patches", "apply_masking_patches"]
  ```
- `opaque/api/patches/transformers/components/__init__.py`: `__all__ = []`
  (per-instance components imported by patch internals).
- `opaque/api/patches/transformers/models/__init__.py`: `__all__ = []`
  (each model file does `register_family(...)` at import).
- `opaque/api/patches/peft/__init__.py`:
  ```python
  __all__ = ["apply_peft_model_patches"]
  ```
- `opaque/api/patches/peft/components/__init__.py`: `__all__ = []`.

### 11.2 Façade — `opaque/patches/`

The façade mirrors the impl tree's public sub-API and **re-exports every entry
function**, not just the orchestrators:

```
opaque/
└── patches/
    ├── __init__.py
    ├── torch/__init__.py
    ├── kernels/__init__.py
    ├── transformers/
    │   ├── __init__.py
    │   └── runtime/__init__.py
    └── peft/__init__.py
```

`opaque/patches/__init__.py`:

```python
from opaque.api.patches import (
    apply_model_patches,
    apply_runtime_patches,
    apply_transformers_model_patches,
    apply_peft_model_patches,
)

__all__ = [
    "apply_model_patches",
    "apply_runtime_patches",
    "apply_transformers_model_patches",
    "apply_peft_model_patches",
]
```

`opaque/patches/torch/__init__.py`:

```python
from opaque.api.patches.torch import apply_checkpoint_patch, is_checkpoint_patched

__all__ = ["apply_checkpoint_patch", "is_checkpoint_patched"]
```

`opaque/patches/kernels/__init__.py`:

```python
from opaque.api.patches.kernels import (
    opaque_cross_entropy_loss, opaque_linear_cross_entropy_loss,
    opaque_swiglu, opaque_geglu_exact, opaque_geglu_approx,
    opaque_rms_norm, opaque_fused_add_rms_norm,
    opaque_rope, opaque_rope_qk, opaque_slow_rope,
    opaque_lora_w, opaque_lora_qkv, opaque_lora_mlp,
    ACTIVATION_SWIGLU, ACTIVATION_GEGLU_EXACT, ACTIVATION_GEGLU_APPROX,
)

__all__ = [
    "opaque_cross_entropy_loss", "opaque_linear_cross_entropy_loss",
    "opaque_swiglu", "opaque_geglu_exact", "opaque_geglu_approx",
    "opaque_rms_norm", "opaque_fused_add_rms_norm",
    "opaque_rope", "opaque_rope_qk", "opaque_slow_rope",
    "opaque_lora_w", "opaque_lora_qkv", "opaque_lora_mlp",
    "ACTIVATION_SWIGLU", "ACTIVATION_GEGLU_EXACT", "ACTIVATION_GEGLU_APPROX",
]
```

`opaque/patches/transformers/__init__.py`:

```python
from opaque.api.patches.transformers import (
    apply_transformers_model_patches,
    family_name,
    make_apply_family_patches,
    make_apply_model_patches,
    register_activation_kind,
    register_family,
    register_fused_add_rms_kind,
    register_rms_norm_kind,
    supported_families,
)

__all__ = [
    "apply_transformers_model_patches",
    "family_name",
    "make_apply_family_patches",
    "make_apply_model_patches",
    "register_activation_kind",
    "register_family",
    "register_fused_add_rms_kind",
    "register_rms_norm_kind",
    "supported_families",
]
```

`opaque/patches/transformers/runtime/__init__.py`:

```python
from opaque.api.patches.transformers.runtime import (
    apply_collator_patches, apply_masking_patches,
)

__all__ = ["apply_collator_patches", "apply_masking_patches"]
```

`opaque/patches/peft/__init__.py`:

```python
from opaque.api.patches.peft import apply_peft_model_patches

__all__ = ["apply_peft_model_patches"]
```

### 11.3 `opaque-transformers` (placeholder wheel)

Today `opaque-transformers` ships only `opaque/transformers/__init__.py` with
`__version__`. Two options to lock in:

1. **Drop `opaque-transformers`** as a separate distribution and route HF
   patching exclusively through `opaque.patches.transformers` (preferred —
   avoids two “transformers” entry points).
2. **Keep `opaque-transformers`** as a thin alias that re-exports the same
   surface as `opaque.patches.transformers` (`apply_transformers_model_patches`
   and the registration helpers above), plus `__version__`. Same `__all__` as
   §11.2 façade for transformers, with `__version__` appended.

This plan recommends **option 1** unless we actively want a top-level
`opaque.transformers` namespace for future features beyond patches.

### 11.4 No `opaque.performance` façade for now

There is no `opaque/performance/` in the current tree. Kernels and the torch
checkpoint patch are reachable from `opaque.patches.kernels` and
`opaque.patches.torch`. If a `opaque.performance` umbrella is introduced
later, it can be a thin re-export over both — added in a separate ADR.

---

## 12. Cross-cutting registries

### 12.1 `sync` registry

- **Defined in:** `opaque.api.base.distributed._registry` and `_sync`.
- **Public dispatch:** `opaque.distributed.sync(obj)`.
- **Power-user registration:** `opaque.api.base.distributed.register_sync_handler(type_, handler)`.
- **Registered handlers (side-effect imports on stack import):**
  - `opaque.api.engine.clipping._distributed` → `ClipState`
  - `opaque.api.dpsgd.noise._distributed` → `GaussianNoiseState`, …
  - `opaque.api.dpftrl.noise._distributed` → `MfNoiseState`, …
- The dispatcher never imports stack code; stacks register on import.

### 12.2 Serialization registry

- **Defined in:** `opaque.api.base.serialization._registry`.
- **Public load/save:** `opaque.serialization.state_dict` / `from_state_dict`.
- **Power-user registration:** `opaque.api.base.serialization.register_serializer(tag, dumper, loader)`.
- **Registered serializers (side-effect imports on stack import):**
  - `opaque.api.accounting.core._serialization`: PLD types, DpProcess base.
  - `opaque.api.accounting.dpsgd._serialization` (new file): SGD process types.
  - `opaque.api.accounting.dpftrl._serialization` (new file): FTRL process types.
  - `opaque.api.dpsgd.<concern>._serialization`, … as needed for state objects.
- Tags are versioned strings (`"opaque.dpsgd.gaussian/1"`).

---

## 13. Migration phases

| Phase | Scope |
|-------|-------|
| **0** | ADR finalising names, façade contract, PEP 420 list (`opaque/`, `opaque/api/`, `opaque/api/accounting/`); CI guard updated. |
| **1** | Create `opaque/api/base/` inside today’s `opaque-core`; move types/pytree/random/serialization/distributed implementations there; convert `opaque/types.py` etc. into re-export façades. **Same wheel** for now. |
| **2** | Split `opaque-engine` out of `opaque-core`: move `_clipping` → `opaque/api/engine/clipping`, `functional` / `scheduling` / `profiling` → `opaque/api/engine/...`; create façades. |
| **3** | Split `opaque-optimizers` out: move `optimizers/` → `opaque/api/optimizers/`, façade at `opaque/optimizers/`. |
| **4** | Inside `opaque-accounting`, reorganise `opaque/accounting/_*` into `opaque/api/accounting/core/...`; thin the façade per §7.2. |
| **5** | Inside `opaque-dpsgd` / `opaque-dpftrl`: move impl into `opaque/api/dpsgd/...` and `opaque/api/dpftrl/...`; move accounting factories into `opaque/api/accounting/{dpsgd,dpftrl}/...`; convert `opaque/dpsgd/...` and `opaque/dpftrl/...` to thin façades. |
| **6** | `opaque-auditing` and `opaque-patches` follow the same façade/impl rule. |
| **7** | Drop `opaque-core` as a published wheel name: it is now empty (or deprecated to a meta-pin equivalent of the umbrella `opaque`). |

Each phase keeps the public façade names byte-stable, so application code does
not break between phases.

## 14. CI guards

1. **PEP 420 namespace check:** fail if any wheel ships `opaque/__init__.py`,
   `opaque/api/__init__.py`, or `opaque/api/accounting/__init__.py`.
2. **Façade discipline:** scripted check that `opaque.dpsgd.*` / `opaque.dpftrl.*`
   modules contain only `from opaque.api... import ...` statements plus
   `__all__` (no other code; comments + docstrings allowed).
3. **No accounting factory leakage:** `opaque/accounting/__init__.py`,
   `opaque/accounting/mechanisms/__init__.py`, and any other façade module under
   `opaque/accounting/` must not re-export names from
   `opaque/api/accounting/dpsgd/...` or `opaque/api/accounting/dpftrl/...`.
   CI compares the cross-cutting façade `__all__` against the per-stack
   factory names exported by `opaque/api/accounting/{dpsgd,dpftrl}/__init__.py`
   (and their submodules) and fails on any overlap.
4. **Dependency direction:** `opaque-accounting` wheel may not import from
   `opaque.api.engine`, `opaque.api.optimizers`, `opaque.api.dpsgd`, or
   `opaque.api.dpftrl`. Static check via `rg`/`ruff`-style import-allowlist.
5. **`__all__` presence:** every `__init__.py` and every multi-symbol `_*.py`
   module declares `__all__`.
6. **Existing `opaque.clipping` import ban** in `docs/` and `examples/` stays.

## 15. Open decisions (small)

- **Naming `opaque-engine` vs `opaque-runtime`:** plan uses **`opaque-engine`**.
- **Should `opaque.types` keep being a single-file module?** Yes (re-export
  shim) — IDE jump-to-def is cleaner than a folder for a re-export.
- **Should `opaque.api.optimizers` live in `opaque-engine` instead of its own
  wheel?** Default: **own wheel** (drops torchopt from base/accounting cone).
  Foldable into engine if torchopt is judged acceptable there.
- **Future `opaque-lipschitz`:** mirrors `opaque-dpsgd` (`opaque.api.lipschitz.*`
  + `opaque.api.accounting.lipschitz.*` + façade `opaque.lipschitz`).
- **`gaussian_noise` vs `truncated_gaussian_noise`:** today these are two
  separate functions sharing the same `GaussianNoiseState` (the truncated
  variant takes an extra `radius`; math is from the *Bounded Gaussian Mechanism*
  — Chen & Hale 2024 — not a clipping post-process). The Poisson refactor
  collapsed `Poisson` / `TruncatedPoisson` into one factory dispatching on
  `truncated_batch_size` because both use the same draw and only differ in PLD;
  Gaussian is **not** symmetric: the truncated version uses a different
  sampling routine (truncated normal via inverse-CDF) and has its own privacy
  bound. Open question: **fold them into one `gaussian_noise(..., radius=None)`
  signature** (single user-facing function with dispatch on `radius is not None`)
  vs **keep them separate** (current plan). Recommendation: **keep separate
  unless we also unify the accountant story**; revisit when truncated-Gaussian
  accounting moves under `opaque.api.accounting.dpsgd`.
