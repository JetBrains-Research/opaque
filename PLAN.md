# Documentation Update Plan

## Summary of API Changes

Our changes introduced a **two-path materialization model**: `DpProcess` no longer has privacy metrics directly. Instead, users call `.pmf()` or `.cgf()` on a process to get a `PmfPld` or `CgfPld` object, then query metrics on *that*.

Key breaking changes:
1. **No more `process.epsilon_at()` etc.** — metrics live on `CgfPld`/`PmfPld`, not `DpProcess`
2. **No more `set_discretization()` / `get_discretization()`** — removed from public API
3. **No more `DiscretizationConfig` class** in public Python API — discretization params are kwargs to `.pmf()`
4. **`Pld` split into `CgfPld` and `PmfPld`** — two distinct materialization paths
5. **SPA renamed to CGF** throughout
6. **Accountant** now uses `.cgf()` / `.pmf()` instead of direct metric methods

### Canonical new pattern

```python
# OLD: training.epsilon_at(delta=1e-5)
# NEW (CGF — fast, no grid, supports epsilon_at/delta_at/advantage):
training.cgf().epsilon_at(delta=1e-5)
# NEW (PMF — grid-based, supports all 5 metrics including beta_at/risk_at):
training.pmf().epsilon_at(delta=1e-5)
```

---

## Files Requiring Updates (19 files)

### Step 1: Core accounting docs (HIGH — these are the primary user-facing docs)

#### 1a. `docs/user-guide/accounting.md` (~25 instances)
- Lines 29-32: `training.epsilon_at(delta=1e-5)` → `training.cgf().epsilon_at(delta=1e-5)` (and advantage, beta_at)
- Line 57: `total.epsilon_at(delta=1e-5)` → `.cgf()` or `.pmf()`
- Line 72: `g.epsilon_at(delta=1e-5)` → `.cgf().epsilon_at(...)`
- Lines 85, 128, 143, 182, 200, 213, 217, 227, 241: all `proc.epsilon_at(...)` calls
- Lines 262-282: **"Privacy metrics" section** — rewrite: metrics are on Pld, not DpProcess. Add table showing CgfPld vs PmfPld metric availability. Update example.
- Lines 350, 352: `acct.epsilon_at(delta)` → `acct.cgf().epsilon_at(delta)`
- Lines 368-395: **"Discretization" section** — complete rewrite:
  - Remove `set_discretization` / `get_discretization` references
  - Remove `DiscretizationConfig` import
  - Show `.pmf(discretization=1e-5)` pattern instead
  - Remove `training.epsilon_at(delta=1e-5, discretization=1e-5)` (kwargs no longer on epsilon_at)

#### 1b. `docs/api/accounting.md` (~20 instances)
- Line 15: `training.epsilon_at(1e-5)` → add `.cgf()`
- Lines 29-31: `DpProcess` docstring says `pld()` → should say `pmf()` and `cgf()`
- Lines 34-42: Privacy metrics table — **remove or move to CgfPld/PmfPld section**. DpProcess no longer has these methods.
- Lines 55, 68: Examples with `total.epsilon_at(1e-5)`
- Lines 71-106: **`DiscretizationConfig` section** — rewrite: class is internal; show `.pmf(**kwargs)` pattern
- Lines 97-106: `set_discretization` / `get_discretization` — **delete**
- Lines 110-113: References to `set_discretization()` in mechanism section intro
- Lines 315-317: `cached()` docstring references `pld()` → `pmf()`
- Lines 345-396: **Accountant section** — `acct.epsilon_at()` → `acct.cgf().epsilon_at()`. Add new section showing CgfPld/PmfPld classes with their method tables.
- Lines 386-387: Methods list on Accountant — remove direct metrics, show `pmf()`/`cgf()`

### Step 2: Landing page and index docs

#### 2a. `docs/index.md` (landing page)
- Lines 57-59: Code snippet `training.epsilon_at(delta=1e-5)`, `training.advantage()`, `training.beta_at(alpha=0.01)` → add `.cgf()` / `.pmf()` calls

#### 2b. `docs/api/index.md` (API reference index)
- Line 42: `.epsilon_at()` etc. listed as DpProcess methods → clarify they're on CgfPld/PmfPld
- Lines 147-151: Quick-reference table shows `.epsilon_at(delta)` etc. as direct methods → add note about materialization

#### 2c. `docs/getting-started/quickstart.md`
- Line 84: `accountant.epsilon_at(delta)` → `accountant.cgf().epsilon_at(delta)`

### Step 3: Mechanism reference docs (6 files)

#### 3a. `docs/mechanisms/index.md`
- Line 104: `proc.epsilon_at(1e-5)` → `proc.cgf().epsilon_at(1e-5)`

#### 3b. `docs/mechanisms/gaussian.md`
- Lines 70, 88, 133-136: Direct `.epsilon_at()`, `.delta_at()`, `.advantage()`, `.beta_at()` calls

#### 3c. `docs/mechanisms/rectified-gaussian.md`
- Lines 96, 141, 145: Direct `.epsilon_at()` calls

#### 3d. `docs/mechanisms/truncated-gaussian.md`
- Lines 94, 146, 155: Direct `.epsilon_at()` calls

#### 3e. `docs/mechanisms/band-mf.md`
- Lines 101, 148, 152: Direct `.epsilon_at()` calls

#### 3f. `docs/mechanisms/blt.md`
- Lines 150, 159: Direct `.epsilon_at()` calls

#### 3g. `docs/mechanisms/dense-mf.md`
- Lines 120, 124, 153: Direct `.epsilon_at()` calls

### Step 4: Other user guide pages

#### 4a. `docs/user-guide/dp-concepts.md`
- Lines 193, 221, 235, 248: Direct metric calls on DpProcess

#### 4b. `docs/user-guide/index.md`
- Line 69: `acct.epsilon_at(1e-5)` → `acct.cgf().epsilon_at(1e-5)`

### Step 5: Package README

#### 5a. `packages/opaque-accounting/README.md`
- Lines 52, 56-57, 63: Python examples with `step.epsilon_at(...)` / `training.epsilon_at(...)` / `total.epsilon_at(...)` → add `.cgf()`
- Lines 102-116: Pld type table — add note about CgfPld/PmfPld enum split and which metrics each supports
- Keep Rust examples as-is (Rust API unchanged)

### Step 6: Rust comments (LOW PRIORITY)

#### 6a. `src/pld/mod.rs:218-220`
- Remove comment about "old Mechanism trait" / "legacy API"

#### 6b. `src/numerics/gaussian.rs:4`
- Remove "and legacy APIs" from module doc comment

### Step 7: Type stub cleanup (LOW PRIORITY)

#### 7a. `opaque_accounting/_accounting.pyi`
- Lines 6: Module docstring says exports `DiscretizationConfig` — add note that it's internal
- Lines 9-15: Example is fine (documents the low-level Rust API correctly)
- No structural changes needed — this file documents the FFI layer, not the public Python API

---

## General Rules for All Updates

1. **Default to `.cgf()` path** in most examples (faster, no grid)
2. Use `.pmf()` only when demonstrating `beta_at` or `risk_at` (which need the full distribution)
3. For comparison snippets that print multiple metrics, show both paths:
   ```python
   cgf = training.cgf()
   eps = cgf.epsilon_at(delta=1e-5)
   # For beta/risk, materialize to PMF:
   pmf = training.pmf()
   beta = pmf.beta_at(alpha=0.01)
   ```
4. For the Accountant, use `acct.cgf().epsilon_at(...)` in training-loop examples
5. Calibration examples are unchanged (calibrate works internally via budgets, no user-visible PLD)
