# Plan: Unify Noise API & Clean Up Internal MF Inconsistencies

## Agreed Design Decisions

1. **State is always immutable** (frozen dataclasses only)
2. **Merge stateless and stateful variants** — every noise function always returns
   `(noise_fn, state)`; accepts `generator=None|int|torch.Generator` to configure RNG
3. **MF constructors take `grad_template` as first required arg and return state directly** —
   no separate `init_fn` callable; the constructor IS the init
4. **Consistent `*_mf_noise` naming** for all MF recipes, plus `identity_mf_noise` for
   DP-SGD baseline via MF API
5. **`custom_mf_noise`** is the user-facing name for bring-your-own-matrix;
   `matrix_factorization_noise` becomes internal
6. **Remove state types from top-level exports** — import from module if needed for annotations
7. **Internal MF submodules: standalone functions** (functional API, not methods on classes)
8. **Each MF submodule defines `__all__`** with a clear public/internal split

---

## Target User-Facing API

### Layer 1-2: `from opaque import ...` / `from opaque.noise import ...`

```python
from opaque.noise import gaussian_noise          # independent noise (DP-SGD)
from opaque.noise import bounded_gaussian_noise  # bounded domain
from opaque.noise import band_mf_noise           # banded Toeplitz correlated noise
from opaque.noise import blt_mf_noise            # buffered linear Toeplitz
from opaque.noise import dense_mf_noise          # dense optimal (small n)
from opaque.noise import identity_mf_noise       # identity (DP-SGD via MF API, swappable)
from opaque.noise import custom_mf_noise         # bring-your-own-matrix
```

All return `(noise_fn, state)`. All `noise_fn` signatures: `(grads, state) -> (noisy_grads, state)`.

### Layer 3: `from opaque.noise.matrix_factorization import ...`

Power-user building blocks for constructing custom noising matrices:

```python
from opaque.noise.matrix_factorization import StreamingMatrix
from opaque.noise.matrix_factorization import MFNoiseState          # type annotation only
from opaque.noise.matrix_factorization import identity              # identity StreamingMatrix
from opaque.noise.matrix_factorization import prefix_sum
from opaque.noise.matrix_factorization import diagonal
from opaque.noise.matrix_factorization import momentum_sgd_matrix
from opaque.noise.matrix_factorization import multiply_array
from opaque.noise.matrix_factorization import multiply_streaming_matrices
from opaque.noise.matrix_factorization import scale_rows_and_columns
```

### Layer 4: Deep submodule imports (researcher API)

Each module exposes a consistent shape:

```python
# Every MF submodule provides (where applicable):
optimize(...)                        # produce optimal strategy
materialize(repr, n)                 # inspect as dense matrix
inverse_as_streaming_matrix(repr)    # convert to noising StreamingMatrix
sensitivity_squared(repr, ...)       # privacy analysis
per_query_error / max_error / mean_error  # utility analysis
```

---

## Step 1: Unify `gaussian_noise` (merge stateless + stateful)

**Files**: `src/opaque/noise/gaussian_noise.py`, `tests/noise/test_noise.py`

### Current API
```python
noise_fn = gaussian_noise(stddev=1.0)
noisy = noise_fn(grads)

noise_fn, generator = gaussian_noise_stateful(stddev=1.0, seed=42)
noisy = noise_fn(grads, generator)  # mutates generator in-place
```

### New API
```python
noise_fn, state = gaussian_noise(stddev=1.0, generator=None)
# generator=None  → new unseeded Generator
# generator=42    → Generator seeded with 42
# generator=gen   → use passed torch.Generator

noisy, state = noise_fn(grads, state)
```

### Implementation
- Create frozen dataclass `GaussianNoiseState` wrapping `torch.Generator`
- Unify into single `gaussian_noise(stddev, *, generator=None)` → `(noise_fn, state)`
- `noise_fn(grads, state) -> (noisy_grads, state)`
- Delete `gaussian_noise_stateful`
- Update `__all__` in `noise/__init__.py` and `opaque/__init__.py`
- Update tests

---

## Step 2: Unify `bounded_gaussian_noise` (merge stateless + stateful)

**Files**: `src/opaque/noise/bounded_gaussian_noise.py`, `tests/noise/test_bounded_gaussian.py`

### New API
```python
noise_fn, state = bounded_gaussian_noise(stddev=1.0, bounds=(-3.0, 3.0), generator=None)
noisy, state = noise_fn(grads, state)
```

Same `generator=None|int|torch.Generator` pattern. Same immutable state.
Delete `bounded_gaussian_noise_stateful`. Update tests.

---

## Step 3: Rename MF recipes + add `identity_mf_noise` + `custom_mf_noise`

**Renames**:
- `blt_noise` → `blt_mf_noise` (file: `blt_noise.py` → `blt_mf_noise.py`)
- `dense_noise` → `dense_mf_noise` (file: `dense_noise.py` → `dense_mf_noise.py`)
- `band_mf_noise` → stays (already has `_mf_`)

**New files**:
- `src/opaque/noise/identity_mf_noise.py` — thin wrapper:
  ```python
  def identity_mf_noise(grad_template, *, stddev, generator=None):
      return custom_mf_noise(grad_template, identity(), stddev=stddev, generator=generator)
  ```
- `src/opaque/noise/custom_mf_noise.py` — user-facing name for `matrix_factorization_noise`:
  ```python
  def custom_mf_noise(grad_template, noising, *, stddev, generator=None, dtype=None):
      # ... same logic as current matrix_factorization_noise but with
      # grad_template as first arg and generator convention
  ```

**Internal**: `matrix_factorization_noise` in `noise.py` stays as the engine but is no
longer exported from `opaque.noise` or `opaque.noise.matrix_factorization.__init__`.

---

## Step 4: Unify MF constructors — `grad_template` first, return state directly

**Files**: `band_mf_noise.py`, `blt_mf_noise.py`, `dense_mf_noise.py`,
`identity_mf_noise.py`, `custom_mf_noise.py`, `matrix_factorization/noise.py`

### Current API
```python
init_fn, noise_fn = band_mf_noise(n_steps=1000, stddev=1.0, seed=42, bands=4)
state = init_fn(grad_template)
noisy, state = noise_fn(grads, state)
```

### New API
```python
noise_fn, state = band_mf_noise(grad_template, n_steps=1000, stddev=1.0, generator=42, bands=4)
noisy, state = noise_fn(grads, state)
```

Changes for all `*_mf_noise` functions:
- `grad_template` is **first positional argument**
- `seed` → `generator` (same `None|int|Generator` convention)
- Returns `(noise_fn, state)` directly — no `init_fn`

Update `matrix_factorization_noise()` internal accordingly. Update all tests.

---

## Step 5: Clean up exports

**`src/opaque/__init__.py`**:
```python
from opaque.noise import (
    band_mf_noise,
    blt_mf_noise,              # renamed
    bounded_gaussian_noise,
    custom_mf_noise,           # new
    dense_mf_noise,            # renamed
    gaussian_noise,
    identity_mf_noise,         # new
)
# NO: gaussian_noise_stateful, bounded_gaussian_noise_stateful (deleted)
# NO: MFNoiseState, GaussianNoiseState (import from module)
```

**`src/opaque/noise/__init__.py`**: same set.

**`src/opaque/noise/matrix_factorization/__init__.py`**:
```python
from opaque.noise.matrix_factorization.streaming_matrix import (
    StreamingMatrix,
    diagonal,
    identity,
    momentum_sgd_matrix,
    multiply_array,
    multiply_streaming_matrices,
    prefix_sum,
    scale_rows_and_columns,
)
from opaque.noise.matrix_factorization.noise import MFNoiseState

# NO: matrix_factorization_noise (internal, users use custom_mf_noise)
```

---

## Step 6: Internal MF submodules — methods to standalone functions

### banded.py

Convert `ColumnNormalizedBanded` methods to standalone:
- `materialize(cnb)` — was `cnb.materialize()`
- `inverse_as_streaming_matrix(cnb)` — was `cnb.inverse_as_streaming_matrix()`

Keep as data container with classmethods `default()`, `from_banded_toeplitz()`.

**`__all__`** (public):
```python
__all__ = [
    "ColumnNormalizedBanded",
    "materialize",
    "inverse_as_streaming_matrix",
    "minsep_sensitivity_squared",
    "per_query_error",
    "max_error",
    "mean_error",
]
```

### buffered_toeplitz.py

Convert `BufferedToeplitz` methods to standalone:
- `materialize(blt, n)` — was `blt.materialize(n)`
- `inverse(blt)` — was `blt.inverse()`
- `inverse_as_streaming_matrix(blt)` — was `blt.inverse_as_streaming_matrix()`
- `as_streaming_matrix(blt)` — was `blt.as_streaming_matrix()`
- `toeplitz_coefs(blt, n)` — was `blt.toeplitz_coefs(n)`
- `canonicalize(blt)` — was `blt.canonicalize()`

Convert `LossFn` methods: `loss(loss_fn, blt)`, `penalized_loss(loss_fn, blt, inv_blt)`.
Convert `Parameterization` method: `get_parameterized_loss(param, loss_fn)`.

Rename `StreamingMatrixBuilder` → `_StreamingMatrixBuilder` (internal only).

**`__all__`** (public):
```python
__all__ = [
    "BufferedToeplitz",
    "optimize",
    "materialize",
    "inverse",
    "inverse_as_streaming_matrix",
    "sensitivity_squared",
    "max_error",
    "iteration_error",
]
```

Internal (no `__all__`, accessible but not advertised):
`as_streaming_matrix`, `toeplitz_coefs`, `canonicalize`, `LossFn`, `loss`,
`penalized_loss`, `Parameterization`, `get_parameterized_loss`, `optimize_loss`,
`get_init_blt`, `blt_pair_from_theta_pair`, `geometric_sum`, `min_buf_decay_gap`,
`robust_max_error_Gamma_j`, `robust_max_error_Gamma_jk`, `_StreamingMatrixBuilder`.

### toeplitz.py

Rename `materialize_lower_triangular` → keep as is (discussed: users qualify by full path).
Rename `optimize_banded_toeplitz` → `optimize`.

**`__all__`** (public):
```python
__all__ = [
    "optimize",
    "inverse_as_streaming_matrix",
    "materialize_lower_triangular",
    "sensitivity_squared",
    "minsep_sensitivity_squared",
    "per_query_error",
    "max_error",
    "mean_error",
    "optimal_max_error_strategy_coefs",
]
```

Internal: `inverse_coef`, `solve_banded`, `multiply`, `pad_coefs_to_n`,
`optimal_max_error_noising_coefs`, `loss`, `mean_loss`, `max_loss`.

### dense.py

No method→function changes needed (already all functions).

**`__all__`** (public):
```python
__all__ = [
    "optimize",
    "per_query_error",
    "max_error",
    "mean_error",
]
```

Internal: `strategy_from_X`, `get_orthogonal_mask`.

### sensitivity.py

No changes needed (already all standalone functions).

**`__all__`** (public):
```python
__all__ = [
    "single_participation_sensitivity",
    "max_participation_for_linear_fn",
    "minsep_true_max_participations",
    "get_min_sep_sensitivity_upper_bound",
    "get_sensitivity_banded",
    "fixed_epoch_sensitivity",
]
```

Internal: `*_for_X` variants.

---

## Step 7: Update wrapper files for step 6 changes

- `blt_mf_noise.py`: `blt.inverse_as_streaming_matrix()` →
  `inverse_as_streaming_matrix(blt)` (import from buffered_toeplitz)
- `band_mf_noise.py`: `inverse_as_streaming_matrix(coefs)` — already a function, no change.
  `optimize_banded_toeplitz` → `optimize` (import update)
- `dense_mf_noise.py`: no method calls, no change needed

---

## Step 8: Update all tests

Tests to update (incrementally, after each step):
- `tests/noise/test_noise.py` — `gaussian_noise` new API
- `tests/noise/test_bounded_gaussian.py` — `bounded_gaussian_noise` new API
- `tests/matrix_factorization/test_noise_addition.py` — new constructor API, `custom_mf_noise`
- `tests/test_dp_ftrl.py` — `(noise_fn, state)` pattern, renamed functions
- `tests/matrix_factorization/test_banded.py` — methods → functions
- `tests/matrix_factorization/test_buffered_toeplitz.py` — methods → functions
- `tests/matrix_factorization/test_toeplitz.py` — `optimize_banded_toeplitz` → `optimize`
- `tests/clipping/test_adaptive.py` — imports `gaussian_noise`
- `tests/validation/test_lora_dp_training.py` — imports `gaussian_noise`
- `examples/dp_sgd_simple.py` — imports `gaussian_noise`, `bounded_gaussian_noise`

---

## Step 9: Update `CLAUDE.MD` and docs

- Update code tree and API description in `CLAUDE.MD`
- Update any docs referencing old function names

---

## Execution Order

1. **Steps 1+2** in parallel — gaussian + bounded_gaussian unification
2. **Step 3** — rename MF recipes + add identity_mf_noise + custom_mf_noise
3. **Step 4** — MF constructors take grad_template, return state
4. **Step 5** — exports cleanup
5. **Step 6** — internal MF methods → functions
6. **Step 7** — wrapper updates for step 6
7. **Step 8** — tests (incremental after each step)
8. **Step 9** — docs

Run full test suite after each step. No deprecation wrappers — just execute the refactoring.

---

## What We Are NOT Doing

- Not refactoring `StreamingMatrix` itself (fine as frozen dataclass with methods)
- Not touching `optimization.py` (internal utility)
- Not adding state types to top-level exports
- Not adding stateless MF variants (MF is inherently stateful)
