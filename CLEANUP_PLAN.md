# Noise Module Cleanup Plan

Agreements and decisions for reorganizing `opaque/noise/` and
`opaque/matrix_factorization/`.

---

## 1. Naming convention: `_noise` suffix on ALL recipes

Every mechanism function gets the `_noise` suffix. These are noise mechanisms;
the suffix makes that explicit regardless of import context.

```python
from opaque import gaussian_noise, bounded_gaussian_noise
from opaque import band_mf_noise, blt_noise, dense_noise
```

File names match: `gaussian_noise.py`, `bounded_gaussian_noise.py`,
`band_mf_noise.py`, `blt_noise.py`, `dense_noise.py`.

The existing `gaussian.py` and `bounded_gaussian.py` get renamed.

## 2. Top-level API (`from opaque import ...`)

```python
from opaque import gaussian_noise            # stateless, simple
from opaque import bounded_gaussian_noise    # stateless, simple
from opaque import band_mf_noise             # stateful (init_fn, noise_fn)
from opaque import blt_noise                 # stateful (init_fn, noise_fn)
from opaque import dense_noise               # stateful (init_fn, noise_fn)
```

Power-user variants stay deeper:
```python
from opaque.noise import gaussian_noise_stateful
from opaque.noise import bounded_gaussian_noise_stateful
from opaque.noise.matrix_factorization import matrix_factorization_noise, MFNoiseState
```

## 3. Move `opaque/matrix_factorization/` into `opaque/noise/matrix_factorization/`

The math library exists solely to serve noise mechanisms. No other consumer.

## 4. File layout

```
opaque/noise/
    __init__.py                    # re-exports the 5 mechanisms + stateful variants
    gaussian_noise.py              # RENAMED from gaussian.py
    bounded_gaussian_noise.py      # RENAMED from bounded_gaussian.py
    band_mf_noise.py               # NEW recipe (thin wrapper)
    blt_noise.py                   # NEW recipe (thin wrapper)
    dense_noise.py                 # NEW recipe (thin wrapper)
    matrix_factorization/          # MOVED from opaque/matrix_factorization/
        __init__.py                # re-exports StreamingMatrix, identity, etc.
        streaming_matrix.py        # StreamingMatrix class + identity, prefix_sum, ...
        toeplitz.py                # Toeplitz algebra: optimize, inverse, error fns
        buffered_toeplitz.py       # BLT class: optimize, inverse, error fns
        dense.py                   # Dense matrix: optimize, error fns
        sensitivity.py             # Participation patterns, min-sep, etc.
        checks.py                  # Matrix validation helpers
        optimization.py            # L-BFGS wrapper (scipy)
        noise.py                   # MOVED from opaque/noise/matrix_factorization.py
                                   #   contains: matrix_factorization_noise, MFNoiseState
        banded.py                  # ColumnNormalizedBanded (keep for now, see §12)
```

## 5. How `matrix_factorization_noise` works and where its inputs come from

`matrix_factorization_noise(noising, stddev, seed)` is the **generic
correlated noise wrapper**. It takes a noising representation (either a dense
`torch.Tensor` or a `StreamingMatrix`) and returns `(init_fn, noise_fn)`.

The noising representation is produced by mechanism-specific math:

| Recipe           | How the noising representation is built                              |
|------------------|----------------------------------------------------------------------|
| `band_mf_noise`  | `toeplitz.optimize_banded_toeplitz(n, bands)` → coefs                |
|                  | → `toeplitz.inverse_as_streaming_matrix(coefs)` → `StreamingMatrix`  |
| `blt_noise`      | `buffered_toeplitz.optimize(n=n, ...)` → `BufferedToeplitz`          |
|                  | → `blt.inverse_as_streaming_matrix()` → `StreamingMatrix`            |
| `dense_noise`    | `dense.optimize(n, ...)` → strategy `Tensor`                        |
|                  | → `torch.linalg.solve(C, I)` → noising `Tensor`                     |
| identity/DP-SGD  | `streaming_matrix.identity()` → `StreamingMatrix`                    |

Each recipe file (`band_mf_noise.py`, etc.) wires up the optimize → convert →
wrap pipeline into a single call.

### Function name: keep `matrix_factorization_noise`

It lives at `opaque.noise.matrix_factorization.matrix_factorization_noise`.
The name is explicit. Renaming would be churn for no clarity gain.

## 6. Naming convention: strategy / noising across modules

### The problem today

The same two concepts (C = strategy, C^{-1} = noising) use inconsistent
suffixes and sometimes inconsistent prefixes:

| Module              | Strategy (C)          | Noising (C^{-1})              | Convention       |
|---------------------|-----------------------|-------------------------------|------------------|
| toeplitz            | `strategy_coef`       | `noising_coef`                | `_coef`          |
| dense               | `strategy_matrix`     | `noising_matrix`              | `_matrix`        |
| buffered_toeplitz   | `blt`                 | `inv_blt`                     | bare object      |
| banded              | `C` or `strategy`     | via method call               | mixed!           |
| noise.py            | —                     | `noising_matrix` (param name) | `_matrix` but accepts `StreamingMatrix` too |

### The fix

**Rule: suffix describes the representation, prefix describes the role.**

- `_coef` → 1D Toeplitz coefficients (`torch.Tensor`)
- `_matrix` → 2D dense tensor (`torch.Tensor`)
- no suffix → typed object (`BufferedToeplitz`, `StreamingMatrix`, `ColumnNormalizedBanded`)

Apply consistently:

| Module              | Strategy (C)          | Noising (C^{-1})       | Change needed?  |
|---------------------|-----------------------|------------------------|-----------------|
| toeplitz            | `strategy_coef`       | `noising_coef`         | No              |
| dense               | `strategy_matrix`     | `noising_matrix`       | No              |
| buffered_toeplitz   | `blt`                 | `inv_blt`              | No              |
| banded              | `strategy`            | (via method)           | Yes — fix `C` → `strategy` in `per_query_error` |
| noise.py            | —                     | `noising`              | Yes — drop `_matrix` suffix, it accepts StreamingMatrix too |

**Specific renames in noise.py:**
```python
# Before
def matrix_factorization_noise(
    noising_matrix: torch.Tensor | StreamingMatrix, ...

# After
def matrix_factorization_noise(
    noising: torch.Tensor | StreamingMatrix, ...
```

**Specific renames in banded.py:**
```python
# Before
def per_query_error(C: ColumnNormalizedBanded, A=None):

# After
def per_query_error(strategy: ColumnNormalizedBanded, workload=None):
```

## 7. Eliminating repetitions

### 7a. Repeated `max_error` / `mean_error` / `last_error` wrappers

Current state — 4 modules each define thin wrappers around `per_query_error`:

| Module              | `max_error`                     | `mean_error`                    | Other          |
|---------------------|---------------------------------|---------------------------------|----------------|
| toeplitz            | `per_query_error(...)[-1]`      | `per_query_error(...).mean()`   |                |
| dense               | `per_query_error(...).max()`    | `per_query_error(...).mean()`   |                |
| buffered_toeplitz   | `iteration_error(inv_blt, n-1)` | (falls back to toeplitz)        |                |
| banded              | `per_query_error(...).max()`    | `per_query_error(...).mean()`   | `last_error`, `max_error_fn` |

Problems:
1. `mean_error = per_query_error(...).mean()` is copy-pasted 3× identically
2. `banded.py` calls it `max_error_fn` instead of `max_error`
3. `banded.py` has both `last_error` and `max_error_fn` — redundant
4. toeplitz uses `[-1]` (assumes monotone), dense uses `.max()` (general) — same
   name, different semantics

**Fix:**

Keep `per_query_error` as each module's core function (these are genuinely
different algorithms). For the wrappers:

- **Drop `mean_error` wrappers** from toeplitz, dense, banded. It's `.mean()`.
  One line. Users can call it. Or if we keep them, fine — but don't pretend
  they're separate implementations.
- **Standardize `max_error` name** everywhere (banded: rename `max_error_fn` → `max_error`)
- **Drop `last_error`** from banded (it's `per_query_error(...)[-1]`, same as
  toeplitz's max_error; not needed separately)
- **Document the monotone assumption** in toeplitz's `max_error` (it uses `[-1]`
  because Toeplitz per-query error is known to be non-decreasing)

### 7b. "Exactly one of strategy/noising" validation

Copy-pasted identically in `toeplitz.per_query_error` and `dense.per_query_error`:

```python
if (strategy_coef is None) == (noising_coef is None):
    raise ValueError("Specify exactly one of ...")
```

**Fix:** Add a helper to `checks.py`:

```python
def check_exactly_one(**kwargs) -> str:
    """Check exactly one kwarg is not None. Return the name of the one that's set."""
    provided = {k for k, v in kwargs.items() if v is not None}
    if len(provided) != 1:
        names = ", ".join(kwargs.keys())
        raise ValueError(f"Specify exactly one of: {names}. Got: {provided or 'none'}")
    return provided.pop()
```

Then:
```python
# toeplitz.py
which = checks.check_exactly_one(strategy_coef=strategy_coef, noising_coef=noising_coef)

# dense.py
which = checks.check_exactly_one(strategy_matrix=strategy_matrix, noising_matrix=noising_matrix)
```

### 7c. `sensitivity_squared` — different implementations, same name

These are NOT redundant — each uses representation-specific math:
- toeplitz: `dot(coef, coef)` (O(b))
- buffered_toeplitz: closed-form geometric sum (O(buffers²))
- banded: `= max_participations` (trivial for column-normalized)

**No change needed.** Same name, different algorithms — that's correct
module-namespaced polymorphism.

### 7d. `optimize` — different implementations, same name

Same situation:
- toeplitz: optimizes b coefficients via L-BFGS
- buffered_toeplitz: optimizes (theta, theta_hat) pairs, auto-selects num_buffers
- dense: optimizes Gram matrix X = C^T C via L-BFGS

**No change needed.** These are fundamentally different optimization problems.
The recipe wrappers (`band_mf_noise`, `blt_noise`, `dense_noise`) hide the
differences from users.

### 7e. `banded.py` internal naming chaos

Current state in `banded.py`:
- `per_query_error(C=...)` — uses `C` as param name
- `minsep_sensitivity_squared(strategy=...)` — uses `strategy` as param name
- `max_error_fn(...)` — different name than everywhere else
- `last_error(...)` — redundant with max_error for monotone case
- `mean_error(...)` — same as everywhere

**Fix:**
- `C` → `strategy` in `per_query_error`
- `A` → `workload` in `per_query_error` (match the concept)
- `max_error_fn` → `max_error`
- Drop `last_error` (use `per_query_error(...)[-1]` directly)

## 8. Recipe API shape

### Stateful mechanisms (band_mf, blt, dense)

```python
def band_mf_noise(
    n_steps: int,
    *,
    stddev: float,
    seed: int | None = None,
    bands: int | None = None,          # default: heuristic
    min_sep: int = 1,
    max_participations: int | None = 1,
) -> tuple[InitFn, NoiseFn]:
    """BandMF correlated noise mechanism.

    Returns (init_fn, noise_fn):
    - state = init_fn(grad_template)
    - noisy_grads, new_state = noise_fn(clipped_grads, state)
    """

def blt_noise(
    n_steps: int,
    *,
    stddev: float,
    seed: int | None = None,
    max_buffers: int = 10,
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = "max",
) -> tuple[InitFn, NoiseFn]:
    """BLT correlated noise mechanism.

    Returns (init_fn, noise_fn):
    - state = init_fn(grad_template)
    - noisy_grads, new_state = noise_fn(clipped_grads, state)
    """

def dense_noise(
    n_steps: int,
    *,
    stddev: float,
    seed: int | None = None,
    epochs: int = 1,
    bands: int | None = None,
    equal_norm: bool = False,
) -> tuple[InitFn, NoiseFn]:
    """Dense matrix correlated noise mechanism.

    Returns (init_fn, noise_fn):
    - state = init_fn(grad_template)
    - noisy_grads, new_state = noise_fn(clipped_grads, state)
    """
```

### Stateless mechanisms (gaussian, bounded_gaussian)

```python
def gaussian_noise(stddev: float) -> NoiseFn:
    """Returns noise_fn(grads) -> noisy_grads"""

def bounded_gaussian_noise(stddev: float, bounds: tuple[float, float]) -> NoiseFn:
    """Returns noise_fn(grads) -> noisy_grads"""
```

The stateful vs stateless distinction is intentional — MF mechanisms need memory
for correlated noise, gaussian does not.

## 9. `noise/__init__.py` exports

```python
# The 5 mechanisms (re-exported to opaque/__init__.py)
from opaque.noise.gaussian_noise import gaussian_noise, gaussian_noise_stateful
from opaque.noise.bounded_gaussian_noise import bounded_gaussian_noise, bounded_gaussian_noise_stateful
from opaque.noise.band_mf_noise import band_mf_noise
from opaque.noise.blt_noise import blt_noise
from opaque.noise.dense_noise import dense_noise
```

## 10. What stays in `noise/matrix_factorization/`

After moving the recipe wrappers up to `noise/`, the `matrix_factorization/`
package contains only **shared math infrastructure**:

| File                    | Purpose                                                | Used by                        |
|-------------------------|--------------------------------------------------------|--------------------------------|
| `streaming_matrix.py`   | `StreamingMatrix` class, identity, prefix_sum, etc.    | toeplitz, buffered_toeplitz    |
| `toeplitz.py`           | Toeplitz algebra, optimize, inverse, error, sensitivity| band_mf_noise, buffered_toeplitz|
| `buffered_toeplitz.py`  | BLT class, optimize, inverse, error, sensitivity       | blt_noise                      |
| `dense.py`              | Dense optimization, error functions                    | dense_noise                    |
| `sensitivity.py`        | Min-sep, participation patterns, banded masks          | toeplitz, buffered_toeplitz, dense|
| `checks.py`             | Matrix validation (lower-tri, square, symmetric)       | sensitivity, dense             |
| `optimization.py`       | L-BFGS wrapper via scipy                               | toeplitz, dense, buffered_toeplitz (via optimize_loss)|
| `noise.py`              | `matrix_factorization_noise`, `MFNoiseState`           | all 3 recipes                  |
| `banded.py`             | `ColumnNormalizedBanded` (see §12)                     | only its own test              |

### Can we merge any of these?

**No.** Every file has 2+ consumers or is large enough to stand alone:

- `checks.py` — `sensitivity.py` + `dense.py`
- `optimization.py` — `toeplitz.py` + `dense.py` + `buffered_toeplitz.py`
- `sensitivity.py` — `toeplitz.py` + `buffered_toeplitz.py` + `dense.py`
- `toeplitz.py` — `band_mf_noise.py` + `buffered_toeplitz.py`

The modules are correctly factored. Leave them as-is.

### `__init__.py` re-exports

```python
# opaque/noise/matrix_factorization/__init__.py
from opaque.noise.matrix_factorization.noise import matrix_factorization_noise, MFNoiseState
from opaque.noise.matrix_factorization.streaming_matrix import (
    StreamingMatrix,
    identity,
    prefix_sum,
    diagonal,
    momentum_sgd_matrix,
    multiply_array,
    multiply_streaming_matrices,
    scale_rows_and_columns,
)
```

## 11. Renames summary

| Before                                    | After                                           |
|-------------------------------------------|-------------------------------------------------|
| `opaque/noise/gaussian.py`                | `opaque/noise/gaussian_noise.py`                |
| `opaque/noise/bounded_gaussian.py`        | `opaque/noise/bounded_gaussian_noise.py`        |
| `gaussian()`                              | `gaussian_noise()`                              |
| `gaussian_stateful()`                     | `gaussian_noise()` (returns stateful tuple)     |
| `bounded_gaussian()`                      | `bounded_gaussian_noise()`                      |
| `bounded_gaussian_stateful()`             | `bounded_gaussian_noise()` (returns stateful tuple) |
| `opaque/matrix_factorization/`            | `opaque/noise/matrix_factorization/`            |
| `opaque/noise/matrix_factorization.py`    | `opaque/noise/matrix_factorization/noise.py`    |
| `noising_matrix` param in noise.py        | `noising`                                       |
| banded: `C` param                         | `strategy`                                      |
| banded: `A` param                         | `workload`                                      |
| banded: `max_error_fn()`                  | `max_error()`                                   |
| banded: `last_error()`                    | (drop)                                          |
| (new)                                     | `opaque/noise/band_mf_noise.py`                 |
| (new)                                     | `opaque/noise/blt_noise.py`                     |
| (new)                                     | `opaque/noise/dense_noise.py`                   |
| (new)                                     | `checks.check_exactly_one()`                    |

## 12. `banded.py` (`ColumnNormalizedBanded`)

Status: **only used by its own test.** Nobody else imports it.

- Always initialized from Toeplitz coefs (`from_banded_toeplitz`)
- No optimizer for the non-Toeplitz case
- `toeplitz.inverse_as_streaming_matrix(coef, column_normalize_for_n=n)` does
  column normalization already

**DECISION NEEDED:** Drop it? Keep as internal?

TENTATIVE: Keep for now, revisit when multi-epoch BandMF needs per-row variation.

## 13. What's NOT changing

- `per_query_error` implementations (genuinely different algorithms per module)
- `sensitivity_squared` implementations (different math per representation)
- `optimize` implementations (different optimization problems)
- Internal relative imports within `matrix_factorization/` (`from . import`)
- `opaque/sampling/` — untouched
- `opaque/clipping/` — untouched

## 14. Execution order

1. Create branch
2. Naming fixes in existing code:
   a. `noise.py`: `noising_matrix` param → `noising`
   b. `banded.py`: `C` → `strategy`, `A` → `workload`, `max_error_fn` → `max_error`, drop `last_error`
   c. Add `checks.check_exactly_one()`, use in toeplitz + dense
3. Rename `gaussian.py` → `gaussian_noise.py`, update function names + `_noise` suffix
4. Rename `bounded_gaussian.py` → `bounded_gaussian_noise.py`, same
5. Move `opaque/matrix_factorization/` → `opaque/noise/matrix_factorization/`
6. Move `opaque/noise/matrix_factorization.py` → `opaque/noise/matrix_factorization/noise.py`
7. Create recipe files: `band_mf_noise.py`, `blt_noise.py`, `dense_noise.py`
8. Update `noise/__init__.py` and `opaque/__init__.py`
9. Update all import paths in tests and docs
10. Run tests, fix breakage
11. Update docs

## Open questions

- [ ] Drop `ColumnNormalizedBanded` or keep?
- [x] The `add_gaussian_noise` references in examples — fixed to use `gaussian_noise()`
