# Horizon allocation and DP accounting — implementation plan

Contributor-oriented plan for random allocation, k-out-of-t, and the
`DpHorizonProcess` / `per_step` refactor. User-facing behavior is documented
in the [accounting](../user-guide/accounting.md) and
[sampling](../user-guide/sampling.md) guides; this file tracks architecture,
phases, and merge criteria.

---

## 1. Problem statement

Opaque was built around **per-step composition** for DP-SGD:
`step = poisson(gaussian(σ), q)` and `training = step * T`, because Poisson
subsampling has clean `* k` semantics.

**Random allocation (Scheme B, redrawn each epoch)** and **global k-out-of-t**
are **whole-horizon** mechanisms: privacy depends on the full schedule
(`t` steps, allocation rule), not on i.i.d. repeated atoms. DP-FTRL already
worked this way (formerly `DpFtrlProcess` + an optional per-step view).

Without a shared abstraction, random allocation was forced into ad hoc trainer
paths (custom accounting units, duplicated `per_step` in `opaque-dpsgd`,
inconsistent semantics vs DP-FTRL).

**Goal:** One correct model for horizon-bound processes, one generic
`per_step` adapter, Poisson unchanged as the native per-step special case, and
trainer integration that does not special-case accounting beyond configuration
and calibration.

**Non-goals for this effort:**

- Converting all of DP-SGD Poisson to horizon-only accounting.
- Remaining deep accounting research (tighter k-out-of-t bounds, MC paths,
  cross-validation against `dp-accounting`, etc.) — tracked in audit docs, not
  this PR.
- Backward compatibility for `DpFtrlProcess` naming or
  `opaque.dpftrl.accounting.per_step` — breaking rename/move is intentional.

---

## 2. Architectural decision

After comparing four directions (true per-step RA atom, FTRL-style whole process
with fake per-step, unify Poisson on horizon, stay heterogeneous), the chosen
model is:

| Layer | Role |
| --- | --- |
| **`DpHorizonProcess`** | Declares `n_steps`; implements **`pld_at(K)`** for prefix privacy (exact or conservative). Full run = `pld_at(n_steps)` ≡ `pld()`. |
| **`PerStep` + `opaque.accounting.per_step(proc)`** | Wraps a horizon process so `Accountant` can use **`acc \|= step`** and **`step * K`**. `Repeated(PerStep(proc), K).pld()` must match **`proc.pld_at(K)`** (contract-tested). |
| **Poisson (DP-SGD)** | Stays a **native per-step** `DpProcess` with `* k` composition — no forced horizon wrapper. |
| **Trainer** | Builds horizon processes where needed, wraps with **`acc.per_step`**, calibrates with **`mechanism(σ) * remaining_steps`**, charges **one accountant step per optimizer step** (no separate “epoch unit” counter). |

**Scheme distinction (unchanged, but explicit in docs):**

- **Redrawn random allocation** (`RandomAllocation` + `RandomAllocationSampler`) —
  Scheme B, `opaque-dpsgd`.
- **Fixed balls-in-bins** (`BallsInBins` + `BallsInBinsSampler`) — Scheme A,
  `opaque-dpftrl`.
- **Global k-out-of-t** (`KOutOfT` + `KOutOfTSampler`) — uniform `k`
  participations over `t` steps; accounting + sampler in `opaque-dpsgd`.

---

## 3. Work already merged (prior PRs)

These are done on `main` / sibling branches; context only:

1. **Safe-only native random allocation** — Remove public `upper` / optimistic
   selector; conservative geometric convolution and analytic discretization
   tails; audit items OPQ-183–186 recorded in
   [audit-master-list.md](audit-master-list.md).
2. **Sampler schedule fixes** — Partial final epoch for
   `RandomAllocationSampler`; empty bin slots for `BallsInBinsSampler` aligned
   with accounting.
3. **DPTrainer: Gaussian + `random_allocation`** — Config validation, sampler
   dispatch, calibration grid-cap handling, privacy guarantee tests (independent
   PR stack).
4. **DPTrainer: `mf_identity` + `balls_in_bins`** — Separate pairing where
   applicable.

---

## 4. Horizon refactor — phases and status

Status key: **Done (merged)**, **Done (branch)**, **Not done**, **Partial**.

### Phase A — Core abstractions (`opaque-accounting`)

Move FTRL-specific base to cross-cutting core; single `per_step`.

| Task | Status |
| --- | --- |
| Add `opaque.api.accounting.core._horizon.DpHorizonProcess` with `pld_at` / default `pld()` | Done (branch) |
| Add `opaque.api.accounting.core.composition._per_step`: `PerStep`, `per_step()` | Done (branch) |
| Export `per_step` from `opaque.accounting` | Done (branch) |
| Façade `opaque.accounting.types` with `DpHorizonProcess` | Done (branch) |
| Delete `opaque-dpftrl` `_base.py` (`DpFtrlProcess`) and `dpftrl/composition/_per_step.py` | Done (branch) |
| Update registry / serialization to skip abstract `DpHorizonProcess` | Done (branch) |

**Acceptance:** `tests/contracts` façade discipline; dpftrl `test_per_step*.py`
import core types; no remaining `DpFtrlProcess` in source (stale references may
remain in `auditraw.json` only).

---

### Phase B — DP-FTRL migration to `DpHorizonProcess`

All FTRL amplifications are horizon processes; trainer uses
`opaque.accounting.per_step`.

| Task | Status |
| --- | --- |
| `CyclicPoisson`, `BMinSep`, `BallsInBins` subclass `DpHorizonProcess` | Done (branch) |
| `build_step_mechanism_factory` uses `opaque.accounting.per_step` | Done (branch) |
| Remove `per_step` from `opaque.dpftrl.accounting` exports | Done (branch) |
| Tests: `test_per_step.py`, `test_per_step_invariants.py`, namespace tests | Done (branch) |
| Docs: `accounting.md`, `dp-ftrl.md`, reference pages | Done (branch) |
| `examples/train_dpftrl.py` | Done (branch) |

**Acceptance:** K-prefix invariants hold for FTRL amplifications × mechanisms;
`per_step(proc) * K` ≡ `pld_at(K)` within discretization tolerance.

---

### Phase C — Redrawn random allocation as horizon + exact prefix

`random_allocation(..., n_steps=)` is a `DpHorizonProcess`; partial epochs
exact, not “charge full epoch”.

| Task | Status |
| --- | --- |
| Python `RandomAllocation(DpHorizonProcess)` with `pld_at` | Done (branch) |
| Rust: `random_allocation_gaussian_prefix_pld` (1-out-of-t partial epoch) | Done (branch) |
| Factory requires `n_steps`; tests + cross-stack integration | Done (branch) |
| Docs for horizon + `acc.per_step(process)` | Done (branch) |

**Acceptance:** Prefix monotonicity / endpoint tests (Rust + Python);
cross-stack state_dict and cross-validation tests pass.

---

### Phase D — Global k-out-of-t (library surface)

Expose native `k > 1` support with sampler + accountant, not only `k=1` via
bins.

| Task | Status |
| --- | --- |
| `KOutOfTSampler` (streaming uniform k-subset schedule) | Done (branch) |
| `KOutOfT` horizon process + `k_out_of_t()` factory | Done (branch) |
| Rust: `k_out_of_t_gaussian_prefix_pld` (hypergeometric prefix; conservative where needed) | Done (branch) |
| Unit tests under `packages/opaque-dpsgd/tests/` | Done (branch) |
| Reference docs for `k_out_of_t` | Done (branch) |

**Acceptance:** `cargo test --workspace`; Python k-out-of-t tests pass; exports
in `__all__` / contract tests.

**Not in Phase D:** DPTrainer `sampling_mode="k_out_of_t"` — see Phase F.

---

### Phase E — Native / shared numerics

Reuse mixture structure from parallel Poisson; keep RA convolution fast and
safe.

| Task | Status |
| --- | --- |
| `amplification/discrete_mixture.rs` helpers | Done (branch) |
| `GeomPmf` index table, directional rounding, grid cap | Done (merged + branch) |
| Refactor `parallel_poisson.rs` to shared assumptions | Done (branch) |

**Acceptance:** `cargo test --workspace`; random allocation reproducibility
tests.

---

### Phase F — DPTrainer wiring (generic horizon path)

No trainer-specific “accounting unit”; horizon modes use the same loop as
Poisson.

| Task | Status |
| --- | --- |
| Gaussian + `random_allocation`: `acc.per_step(dpsgd_acc.random_allocation(...))` in `_build_mechanism` | Done (branch) |
| Calibration: raise `param_min` when PLD grid exceeds `max_grid_size` | Done (branch) |
| DP-FTRL branch docstring → `opaque.accounting.per_step` | Done (branch) |
| Remove any `accounting_unit_steps` / epoch-multiple charging | Done (branch) |
| **`k_out_of_t` in `TrainingArguments`** (`_SAMPLING_MODES`, validation, `sampling_kwargs.total_participations`) | Not done |
| **`build_sampler` / `_build_mechanism` for `k_out_of_t`** | Not done |
| **Trainer tests** (config, smoke, `test_dp_guarantees` for k-out-of-t) | Not done |
| **Docs: `training-arguments.md`** for k-out-of-t | Partial |

**Acceptance:** Same patterns as `random_allocation`: complete schedule,
calibration, step-wise `acc |= step`, resume recalibration, stop-at-ε.

---

### Phase G — Cleanup and superseded paths

| Task | Status |
| --- | --- |
| Remove `opaque-dpsgd` duplicate `composition/_per_step` / `PerStepRandomAllocation` | Done (no `dpsgd/composition` tree) |
| Supersede stacked PR `cursor/dptrainer-allocation-per-step-d1f5` with horizon branch | Verify at PR time |
| Close redundant open PRs after horizon PR merges | Pending |

---

### Phase H — Validation, commit, PR

| Task | Status |
| --- | --- |
| `uv sync --group dev --all-packages --extra all` | Pending |
| `uv run pytest -m "not cuda and not mps and not slow"` | Pending |
| Targeted: dpsgd/dpftrl accounting, transformers validation, `tests/integration/accounting`, `tests/contracts` | Pending |
| `uv run ruff check packages/` + format | Pending |
| `cargo test --workspace` | Done (355 passed, last run) |
| Docs build (`uv sync --group docs` + `mkdocs build --strict`) | Pending |
| Commit, push `cursor/horizon-allocation-processes-d1f5` | Pending |
| Open/update stacked PR | Pending |

**Suggested PR title (Conventional Commits):**

`refactor(accounting): add DpHorizonProcess and unify allocation accounting`

**PR body should cover:** heterogeneity + `per_step`; rename/migration; exact RA
prefix; k-out-of-t library surface; trainer uses generic `per_step` for RA;
breaking removal of `DpFtrlProcess` / dpftrl-local `per_step`.

---

## 5. Semantic contracts (merge gate)

1. **`per_step` materialisation:** For every shipped `DpHorizonProcess` `P`,
   `(acc.per_step(P) * K).epsilon_at(δ)` is conservative vs
   `P.pld_at(K).epsilon_at(δ)`; equality where exact prefix is implemented.

2. **Calibration objective:** `mechanism(σ) * total_steps` for fresh runs;
   `prefix | (mechanism(σ) * remaining)` on resume — same for Poisson and
   `per_step(horizon)`.

3. **Sampler ↔ accountant parameters:**
   - RA: `num_bins`, `n_steps` match sampler and `random_allocation`.
   - BiB: `num_bins`, `n_steps` multiple (FTRL rules).
   - k-out-of-t: `total_participations`, `n_steps` match `KOutOfTSampler` and
     `k_out_of_t`.

4. **Safe-only policy:** No user-facing “upper/optimistic” PLD API for random
   allocation; internal down-rounding only where required for conservative
   composition.

---

## 6. Explicitly out of scope

Recorded in [audit-master-list.md](audit-master-list.md) and
[audit-remediation-plan.md](audit-remediation-plan.md) unless they block CI:

- Tighter k-out-of-t prefix bounds beyond current hypergeometric / block cap
  strategy.
- MC random-allocation accounting, reproducibility at scale, sample-size
  sensitivity.
- Converting Poisson DP-SGD to mandatory horizon declaration.
- Cadence / GPU feature-validation runs (optional follow-up, not merge gate).

---

## 7. Remaining execution order

1. Finish **Phase F** (k-out-of-t trainer wiring + tests + training-args docs)
   if the product requirement includes trainer; otherwise document k-out-of-t as
   **library-only** and defer F to a small follow-up PR.
2. Run **Phase H** full CI-equivalent suite and fix failures.
3. Commit on `cursor/horizon-allocation-processes-d1f5`, push, open PR, reconcile
   stack with sampler-schedules / old per-step trainer PRs.
4. Copilot review + CI green per [CONTRIBUTING.md](https://github.com/JetBrains-Research/opaque/blob/main/CONTRIBUTING.md).

---

## 8. Key code locations

| Concern | Path |
| --- | --- |
| Horizon base | `packages/opaque-accounting/src/opaque/api/accounting/core/_horizon.py` |
| Generic `per_step` | `packages/opaque-accounting/src/opaque/api/accounting/core/composition/_per_step.py` |
| RA accounting | `packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/amplification/_random_allocation.py` |
| k-out-of-t accounting | `packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/amplification/_k_out_of_t.py` |
| RA / k-out-of-t Rust | `packages/opaque-accounting/src/amplification/random_allocation.rs` |
| Per-step contract tests | `packages/opaque-dpftrl/tests/accounting/test_per_step.py`, `test_per_step_invariants.py` |
| Trainer mechanism build | `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py` (`_build_mechanism`, `_calibrate_noise`) |
| FTRL helpers | `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dpftrl.py` |

---

## 9. Branch snapshot (update when merging)

- **Feature branch:** `cursor/horizon-allocation-processes-d1f5`
- **Typical stack base:** `cursor/allocation-sampler-schedules-d1f5` or `main`
  depending on open PR stack.
- **Last known Rust result:** 355 tests passed (`cargo test --workspace`).

Update this section when phases complete or the branch name changes.
