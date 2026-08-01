# Opaque DP Library — Remediation Plan

**Scope:** ~185 verified findings + 24 pre-established issues (A–X). After de-duplication (see §0.1) there are **~170 distinct defects** across 10 packages, docs, and CI.

**Framing:** this is a differential privacy library. The failure mode that matters is not "crashes" — it is "prints a number that is wrong in the unsafe direction and the user believes it." That ordering drives the whole plan: everything that corrupts a released ε, or that releases data the accountant never charged for, comes first, regardless of how small the diff is.

---

## 0. Read this first

### 0.1 Duplicates inside the verified list — do not open 12 tickets

These are the same defect reported from different call sites. One fix each:

| Merge into | Duplicated by |
|---|---|
| `_gaussian.py` inverse-CDF truncation | "hard-truncated at ±5.1666σ", "samples a hard-truncated atom-carrying distribution", "Unbounded Gaussian path hard-truncates at ±5.16σ" (3 reports, 1 bug) |
| λ-CGD zero noise at step `n_steps` | "λ-CGD with normalized=True emits ZERO noise", "DP-λ-CGD emits exactly zero noise at step == n_steps" |
| `all_finite()` wrapper no-op | "returns True unconditionally for ClippedPytree…", "can never return False on a clipped_grad output" |
| Loss-scaler "zero privacy budget" claim | docs/user-guide/precision.md finding + `_loss_scaler.py:27-34` finding |
| AdaClip realized-vs-expected batch denominator | `_clipped_fun.py:569-586` finding + `_adaptive.py:103-114` finding + the Rust `adaclip.rs` Δ=1/2 finding are **one root cause** (§2, RC-3) |
| MF noise redrawn every step | "dense-matrix MF engine path redraws", "_tensor_mf_noise redraws the full Gaussian vector" |
| `train_dpftrl.py` one-epoch runs | "BLT and Balls-in-Bins runs execute exactly one epoch", "reuses an already-exhausted sampler object" |
| DP-FTRL doc signature drift | "Every DP-FTRL mechanism doc page…", "~35 documentation call sites" |
| `gradient_checkpointing` default | docs/alignment/trainers.md finding + docs/reference/transformers.md finding |
| NOTICE broken paths | two separate reports of the same eight paths |
| trl extra never installed | "trl extra is installed by no CI command", "All 33 TRL-compatibility tests are gated" |
| Quickstart sampler mismatch | "Quick Start calibrates a Poisson-amplified ε", "Quick Start and the DP-SGD / auditing tutorials" (the second supersedes) |

### 0.2 Two decisions to make before any code is written

1. **Release hold.** Every published wheel currently computes ε through `calibrate()` (returns budget-violating parameters as converged), `budget_exceeded` (inverted for two of three budget types), and `CachedProcess` (silently drops DP-FTRL per-step accounting). Yank or mark yanked, and publish an **ERRATA** stating which ε values from which versions must be recomputed. This is not optional for a DP library — users may have published privacy claims derived from these numbers.
2. **Claim inventory.** Approximately 20 documented guarantees are false as written (§1.3). Withdrawing a claim is a 10-minute edit and is *always* available as an interim mitigation while the underlying fix is engineered. Use it aggressively; do not let a 2-week fix gate a 10-minute retraction.

---

## 1. Phase 0 — Stop the bleeding (target: 5 working days)

Three parallel tracks. Nothing here should be blocked on design work.

### 1.1 ✅ One-line / one-function fixes that change ε *today*

Order by (impact ÷ diff size). All of these are ≤ 1 day each and several are ≤ 1 hour.

| Fix | File | Effect if not fixed |
|---|---|---|
| ✅ Invert `budget_exceeded` comparison | `opaque-accounting/.../core/_accountant.py:209-213` | Beta/Risk budget runs report "under budget" while over it |
| ✅ `CachedProcess.repeated_pld` passthrough | `.../core/composition/_cached.py` | DP-FTRL per-step accounting silently degrades to K-fold single-step composition |
| ✅ `eps_delta_pld` atom → `ceil` (safe-only) | `src/mechanisms/eps_delta.rs:43` | Declared ε cannot round below the requested value |
| ✅ `calibrate()` return the *proven-safe* bracket endpoint; one-sided acceptance; relative tolerance; raise on non-convergence | `.../core/calibration.py:341-367` | Returns noise multipliers that violate the budget with `converged=True` |
| ✅ Horizon guard in all three MF `noise_fn`s (`step >= n_steps` → raise) | `dpftrl/noise/_lambda_cgd.py`, `_engine.py:319-341` | **Zero-noise release of clipped gradients** at step *n*; unaccounted noise past horizon |
| ✅ Path-keyed PerGroup (optree ParamPath) for nested + flat pytrees | `engine/{types,pytree,clipping}` + dpsgd/dpftrl noise + optimizers BC | Nested containers previously skipped leaves; compiled keys are now unambiguous path tuples |
| ✅ Split the RNG key for quantile vs gradient noise | `transformers/.../_dp_trainer.py:1368, 3847` + 3 examples | AdaClip composition void — identical noise streams |
| ✅ `pass target_quantile=target_clip_rate` | `_dp_trainer.py:3865` + 3 examples | Adaptive clipping converges to the wrong quantile |
| ✅ Reject non-finite scores in `one_run()` | `auditing/one_run/_estimate.py:37-83` | NaN scores manufacture ε̂ from a numerically broken run |
| ✅ Cap/bound the `_mu_at` doubling + bisection | `auditing/one_run/_gdp.py:66-77` | **Infinite hang** for m>2000 with a strong attack |
| ✅ Structurally fixed collective sequence in `sync(aux)` | `engine/clipping/_distributed.py:59-82` | **One empty Poisson batch permanently desynchronizes the process group** (critical) |

**Effort: 3–4 engineer-days**, plus regression tests (another 2). These are unrelated to each other — parallelize across whoever is available.

#### Accounting remediation status (Phase 0)

- **Safe-only PLDs:** `DiscretizationConfig`, PMFs, and all Python process APIs no longer expose estimate mode. Exact atoms, coarsening, truncation, and Monte Carlo histograms take the upper-bound path, and unsupported legacy keywords fail explicitly.
- **Exact mechanisms:** `eps_delta_pld` uses ceiling placement, so a finite exact atom never falls below its declared ε; identity uses the same single safe policy.
- **Calibration / budget:** `calibrate()` returns the proven-safe bracket endpoint with one-sided acceptance; `budget_exceeded` compares in the correct direction for all three budget types.
- **CachedProcess:** `repeated_pld` relays to `inner.repeated_pld`, so `cached(per_step(...)) * K` keeps the DP-FTRL horizon PLD instead of K-fold single-step composition.
- **PyO3 error conversion:** every binding uses one exhaustive `From<PldError> for PyErr` conversion. Invalid input and incompatible operands raise `ValueError`; numerical, calibration-execution, and invalid-state failures raise `RuntimeError` with native diagnostic context.
- **Precision evidence:** the current engine precision surface remains loss-scaler-only, with existing loss-scaler coverage; no precision dispatcher, fallback, or compatibility shim was added for this accounting remediation.
- **MC confidence remains open:** RC-4/A3 is not resolved. Conservative histogram bucketing does not turn b-min-sep or Balls-in-Bins point estimates into confidence bounds.

### 1.2 Fail-closed conversions (cheap, prevents the silent-degradation class)

Convert "unknown → skip" into "unknown → raise" at three remaining dispatch sites. This is the same 5-line change repeated, and it converts eight *silent* findings into loud ones even before the real fixes land:

- `base/serialization/_dispatch.py:54-60` — MRO fallback + raise on unregistered non-container
- `engine/distributed/gradients.py:124-174` — `_reduce`/`_clone` raise on non-Tensor non-None leaf
- `engine/distributed/_state.py:266-276` — MRO walk against `_SYNC_REGISTRY`, raise on miss

**Not applicable in this checkout:** the audited precision-dispatch action references a removed/nonexistent `_dispatch` path. `opaque.api.engine.precision.__init__` exports only the current loss-scaling surface, and `_loss_scaler.py` has no silent fallback dispatch path; no compatibility shim or new engine code is needed.

**Effort: 1 day.** See RC-7 for the full fix.

### 1.3 Claims to withdraw immediately (docs-only, ~1 day total)

Every one of these is a documented guarantee that the code does not provide. Withdraw now; restore only when the corresponding fix lands.

1. `bound=` Gaussian is "post-processing on the standard (ε,δ)-Gaussian" — `docs/user-guide/noise.md:174-179`. It resamples; it is not post-processing; there is no bounded-Gaussian accountant. Mark **experimental, not covered by the recorded ε**.
2. "Skipped steps consume zero privacy budget" — `precision.md:95-99` + `_loss_scaler.py:27-34`. The skip is a data-dependent branch on the un-noised gradient.
3. "eps_K ≤ eps_N exactly" and monotone-in-K for b-min-sep — `b_min_sep/registry.rs:74-83`, `_b_min_sep/__init__.py`, `docs/reference/accounting.md:418-425`. Both measurably violated.
4. "Hard ceiling ε ≤ ln(m / −ln α) regardless of method" — `docs/user-guide/auditing.md:183-196`. False for the default μ-GDP method.
5. "Every Opaque mechanism is in the Gaussian-DP family" — `auditing.md:66-70` (issue R).
6. "First optimal-rate DP-DPO loss" for SquareChiPO — `_squarechipo.py` docstring, `trl-trainers-plan.md:318,1129`, `reference/alignment.md:47` (issue A).
7. "Every public loss is NaN-injection verified" — `opaque-alignment/README.md:25-27`.
8. "TRL numeric parity gate (σ=0, C=∞, 1e-3)" — issue F; no test imports trl.
9. "Continuous privacy regression testing" — issue G; no scheduled workflow exists.
10. All benchmark tables with no harness — `memory-optimizations.md:50-55,170-213`, `bisr.md:112-120`, the 81% Mellum-4b claim, the 22%-vs-40% contradiction (issue W), schedule-free √n, Adadelta memory, `row_norms_squared` "sub-second", `pre_clipping_transform` docstring (issue U — that one is *someone else's text*, remove or attribute).
11. `float32` inverse-CDF sampling "yields a true Gaussian" — `_gaussian.py:189-196`.
12. `adaptive_clipped_grad` "automatically detects if distributed" — `_adaptive.py:200,212`.
13. MC-derived ε values are point estimates — add a one-line caveat to `docs/reference/accounting.md` and the `pld()` docstrings until RC-4 lands.

### 1.4 ✅ Add the missing disclosures

- **`docs/limitations.md`: "Randomness and the threat model."** The PRNG is not cryptographically secure; the seed is user-chosen, printed, and checkpointed; the guarantee is void against anyone who learns it. Noise state in checkpoints must be stripped before publishing artifacts.
- **`docs/limitations.md`: "Telemetry outside the guarantee."** Every logging step publishes un-noised mean loss, mean pre-clip grad norm, clip rate, realized Poisson batch size, DPO reward means. Mirror the disclosure `DPTrainer.evaluate` already makes at `_dp_trainer.py:2716-2725`. Add `runs/` to `ignore_patterns` in `_hub.push_to_hub` **today** — that one is an actual exfiltration path.
- Fix the four alignment metric docstrings that say "not for release" while both shipped trainers release them.

### 1.5 Supply chain — do these this week, they are not code changes

- **Register all five `opaque-*` distribution names on PyPI** as placeholders. Right now `pip install opaque-engine` from the documented `--extra-index-url` command is a live dependency-confusion vector.
- ✅ Switch documented install to `--index-url` (README.md:57-59, installation.md:12-15). Note separately: the name `opaque` on PyPI belongs to an unrelated project and can never be reclaimed — plan a rename to `opaque-dp`.
- **Gate publishing on tests**: add `rust-tests`/`python-tests` to `needs:` for `publish-dev-wheels` (ci.yml) and add a test job gating `publish` (release.yml). Wheels currently ship from a red tree.
- Add `if: github.event.pull_request.head.repo.full_name == github.repository` to the self-hosted GPU leg (pr.yml:85) — a fork PR currently executes arbitrary code on a non-ephemeral runner.

**Phase 0 total: ~1 engineer-week** (parallelizable to ~3 calendar days across 3 people).

---

## 2. Root-cause clusters — fix these once, not fifteen times

This section is the point of the document. Each cluster is **one design change** that resolves the listed findings. Do not let these be worked as independent tickets.

### RC-1 — There is no binding between the sampler you run and the amplification you account for
**Resolves:** quickstart + DP-SGD tutorial + auditing tutorial + DP-FTRL tutorial sampler mismatches; `train_sft.py --no-shard`+`--truncated-batch-size`; `BallsInBinsSampler` dropping empty bins; `train_dpftrl.py` one-epoch runs; `distributed.md` unbounded sampler; `dp-sgd.md` end-to-end loop; issue O (audit coin flip changes n → changes min_sep/bins).

**Change:** samplers emit a `SamplingContract` (mode, `sample_rate`, `n_steps`, `min_sep`, `num_bins`, `max_participations`). Amplification factories *require* one; `Accountant` records it; a mismatch raises. `BallsInBinsSampler` yields empty batches for empty bins so the declared `num_bins` stays true. `parallel_poisson(poisson(...))` composes rather than either-or.

**Why one change:** every one of these bugs is "the runtime sampling distribution and the accounted distribution differ, and nothing checks." Ten doc rewrites without the contract will drift again within two releases.

### RC-2 — The deployed Gaussian is not the Gaussian the accountant prices
**Resolves:** fp32 inverse-CDF ±5.1666σ truncation (×3 reports); bounded-Gaussian collapse to a deterministic zero-noise release beyond ~5σ; bounded-Gaussian "post-processing" doc claim; bounded Gaussian reporting nominal `noise_stddev` while realized variance is smaller (breaking Adam/Adafactor bias correction); no mechanism↔accountant conformance test.

**Change:** (a) replace the inverse-CDF path with `torch.normal` in float64 → cast (matching the DP-FTRL path), or a proper discrete Gaussian (Canonne–Kamath–Steinke) with a matching PLD; (b) reimplement `bound=` as *actual* post-processing (`clamp(gaussian_sample, low, high)`), which is genuinely (ε,δ)-preserving and removes the tail-underflow entirely; (c) report the realized std.

**Why one change:** (b) makes the collapse bug, the doc claim, and the bias-correction bug all disappear at once. Do not fix the `log_ndtr`/`erfcinv` numerics — that preserves a mechanism with no accountant.

### RC-3 — AdaClip's released statistic ≠ AdaClip's charged statistic
**Resolves:** `adaclip.rs:32-33` factor-4 undercharge; realized-batch denominator (×2 reports); `fraction_noise_std` defaulted independently at two sites; RNG key collision; `target_clipping_rate` inversion; circular `test_adaclip_effective_noise`; missing "Privacy accounting" docstring section.

**Change:** release the count against a **public** denominator — `noisy_rate = (num_clipped − B/2 + N(0,σ_b²)) / expected_batch_size + 1/2` (Andrew et al. / TF-Privacy `QuantileEstimatorQuery` form). Then make `adaclip()` accept the `AdaptiveClipState` rather than a loose float, so `fraction_noise_std` and `expected_batch_size` have exactly one source of truth and the accountant validates against the runtime.

**Why one change:** the four-way undercharge, the drift, and the double-default are all "two places independently decide what the released statistic is."

### RC-4 — Monte-Carlo point estimates are presented as privacy guarantees
**Resolves:** MC PLDs with no confidence correction (`epsilon_at(δ)` returns finite for δ ≪ 1/num_mc_samples); b-min-sep non-monotonicity; MC amplification validated only by MC-vs-MC smoke tests; issue I (per-step invariant sandwich skipped for 6 of 8 MC pairs); `MAX_LINEAR_FFT_SIZE` guard inert because MC PLDs have zero tail budget.

**Change:** `samples_to_pmf` / `weighted_samples_to_pmf` must return a **high-probability upper bound**, not an empirical histogram: reserve mass above the (1−k/n) empirical quantile into `infinity_mass`; inflate buckets by one-sided Clopper-Pearson / empirical-Bernstein at a caller-visible confidence level; residual → `infinity_mass`. Surface confidence level + resolution floor through `DiscretizationConfig`.

**Why one change:** the UCB construction makes the monotonicity claim **true by construction**, gives the FFT guard a non-zero tail budget to work with, and makes the sandwich tests in issue I passable rather than skippable. Fixing the monotonicity claim by memoizing a running max is a workaround that leaves the underlying unsoundness.

### RC-5 — MF noise engines have no horizon guard and no correlation test
**Resolves:** λ-CGD zero noise at step n (×2); streaming MF no horizon guard; `_column_norm` returning 0/NaN; dense engine redrawing base Gaussians (×2 reports — zero cross-step correlation, i.e. *not matrix-factorization noise at all*); BISR truncating C to `bandwidth` instead of `n_steps`; `_momentum_workload_coef` applying the LR schedule along the lag axis.

**Change:** thread `n_steps` into `_matrix_factorization_noise`/`_make_raw_mf_noise` and raise in all three `noise_fn`s. Derive base z_j from a per-**column** key `fold_in(key, j)`, not a per-step key. Add an **empirical covariance test**: sample the noise sequence and assert Cov ≈ σ²C⁻¹(C⁻¹)ᵀ, run against dense *and* streaming engines. That single test catches all four correlation/truncation bugs and prevents the next one.

### RC-6 — Strategy identity is not a fingerprint, but PLDs are cached on `self`
**Resolves:** `lr_schedule` excluded via `compare=False` while `pld` workers are `lru_cache`d on `self` → **two different BLT mechanisms share one cached ε** (`_blt.py:122`, `_band_mf.py:115`, 3 consumers); `set_discretization()` silently ignored for already-computed PLDs; `calibrate()` cleanup docstring overstating what is released.

**Change:** introduce `mechanism_fingerprint()` covering the materialized schedule and the resolved `DiscretizationConfig`; key every PLD cache on it; bump a module-level generation counter in `set_discretization()`. One cache-key redesign, three findings.

### RC-7 — Dispatch is by exact type and by runtime value, with "unknown → silently skip" as the default
**Resolves (8 findings + issue H's blast radius):** `state_dict()` dropping every `torch.Tensor` subclass (including `nn.Parameter` from `make_functional`); `all_finite()` returning True for the library's own clipping wrappers (×2); `sum_gradients`/`reduce_pytree` no-op on `SecondMomentClippingOutput` (**each DDP rank trains on its own shard, silently**); `sync()` missing every `ClippedGradAux` subclass; `_split_aux_fields` classifying by value; `sync_object` skipping non-numeric fields; tensor deserialization accepting shape mismatch; per-group clipping no-op on nested pytrees.

**Change — one policy, applied at every registry:**
1. exact-type miss → walk `__mro__` before giving up;
2. still a miss on a non-container, non-primitive leaf → **raise `TypeError`**, never skip;
3. anything that enumerates fields for a collective enumerates the **dataclass schema**, never the runtime values.

Then register the four wrapper NamedTuples/dataclasses properly (or make them optree nodes) and every one of these resolves.

### RC-8 — Collective counts are data-dependent, and no multi-rank code path is exercised in CI
**Resolves:** `sync(aux)` empty-batch desync (critical); `_split_aux_fields` None-on-one-rank; `sync_object` value-derived field map; `sync_perf_tracker` per-rank stage dicts; `reduce_scalar` hard-coded fp32 making "exact" equality a 24-bit comparison; `gather_for_metrics` with uneven per-rank lengths; `sync_object`'s TypeError-retry re-issuing collectives; issue H (√W scaling vs asserted-equal seeds).

**Change:** the fix for each is small; **the reason all eight exist is G1 (§4)** — no multi-rank collective code executes in CI. Build the gloo lane *first*, then fix. Also give `reduce_scalar` a `dtype` param defaulting to float64 plus an integer-exact path, then set `atol=rtol=0` on the equality guards.

### RC-9 — The library has no definition of "what leaves the mechanism"
**Resolves:** per-step un-noised telemetry from `DPTrainer` and both alignment trainers; the four "not for release" metric docstrings; TR-DPO writing per-example private reference logprobs to `tempfile.gettempdir()` with default perms (+ issue Q); the loss-scaler data-dependent skip; `runs/` pushed to the Hub.

**Change:** a single `Release` boundary concept: anything that crosses it is either (a) noised and composed into `ctx.accounting`, or (b) behind `allow_unaccounted_telemetry=True` with a loud warning, or (c) not emitted. Then apply it at the ~6 sites. Also: cache dir 0700, files 0600, and `write_cache=False` for TR-DPO.

### RC-10 — The auditor was never calibrated under the null
**Resolves:** issues B (v_k = 0.5 truncation inflating ε̂ for every m>2000 audit), C (post-hoc threshold selected with secret labels, Type-I 4–5× nominal), D (no two-sided validation); NaN scores; +inf scores corrupting TPR/FPR denominators; `attack_auc` returning convex-hull AUC (biased above 0.5 under the null, ~0% CI coverage); `gdp(grid_size)` returning ε̂=0 for a leaky audit; `_mu_at` hang; the false "hard ceiling" claim; no canary construction (README says "canary injection", code only subsets natural rows); no key-domain separation between the audit RNG and the mechanism RNG; the unstated score-ordering contract; the "Bonferroni" comment describing a correction that does not exist.

**Change:** treat the estimator as unsound and rebuild the statistical core: conservative μ-dependent bound for truncated ranks (not the constant 0.5); threshold selection that does not consult labels (or an explicit multiplicity correction); denominators from `n_in`/`n_out`; raw ROC for AUC; grid size as a function of n with a convergence guard. **Then** build the Type-I calibration suite (G6) — 200 null runs, assert empirical false-positive rate ≤ α + binomial slack — and only then is any ε̂ this library prints meaningful.

### RC-11 — Docs are hand-maintained against a refactored API
**Resolves:** ~35 DP-FTRL call sites + 2 public factory docstrings; `rng.md`'s six broken `gaussian_noise` calls; `transformers.md` tables (missing `chunked_nll`, `use_performance_kernels`; two invented behaviours); `gradient_checkpointing` default in 3 places; `privacy_target_delta` documented default; `.pyi` missing `tail_mass_truncation`; `CONTRIBUTING.md`, `README.md`, `AGENTS.md` rule 6, `.github/WORKFLOWS.md`, NOTICE paths, alignment `.npz` format, `docs/development/opaque-alignment-plan.md` (doesn't exist).

**Change:** generate what can be generated (tables from `dataclasses.fields`, `.pyi` from pyo3 signatures) and **execute what can't** (G4). The manual sweep without the gate buys you one release.

### RC-12 — CI cannot fail on the things that matter
See §4. Eight findings, all "the test exists but never runs, or the lane is `continue-on-error`."

### RC-13 — Citations
~12 verified findings + issue E (14 of 20 alignment loss files). Pattern is uniform: correct prose, wrong arXiv ID; or correct ID, fabricated title; or invented author lists. Two of them (`_lambda_cgd.py`/`lambda_cgd.rs`, `bisr.rs`/`bisr.md`) put fabricated titles **in the accounting crate itself**. One fix (G5): a single `references.bib`, all citations rendered from it, CI resolves each ID against arXiv metadata.

### RC-14 — Performance claims with no harness
Issue J (no benchmark harness exists), issue W, the memory-optimizations tables, the BISR bandwidth table, schedule-free √n, Adadelta memory, `row_norms_squared` "sub-second" (actually O(bands·n²), eager at construction), `adaptive_clipped_grad(return_aux=False)` materializing per-example gradients and discarding them (defeats `microbatch_size`), `MAX_LINEAR_FFT_SIZE`. **Dependency: the harness must exist before any of these numbers can be restated.**

### RC-15 — Packaging / supply chain
PyPI name collision + dependency confusion; unpinned metapackage (`set_build_versions.sh:144-153` is a no-op sed); non-manylinux wheels + no sdist; `__pycache__` in the wheel; missing LICENSE/NOTICE in sub-wheels; unattributed Unsloth `_utils.py`; `opaque-transformers` pyproject with no license/classifiers and a contradictory `requires-python`; Dependabot blind to 10 manifests + Cargo.lock + uv.lock; dead `unsafe_code` workspace lint.

### RC-16 — Accounting core API returns unsafe values
`calibrate()` (two-sided tolerance, direction constant, non-convergence returns), `budget_exceeded`, `CachedProcess`, `set_discretization`, `_process_codec` recursion (RecursionError at ~2500 steps — **checkpointing an accountant fails mid-run**), non-dataclass Budget serialization, `Pld.self_compose(0)` panic through PyO3, u32 count truncation, and `poisson(1.0)`/`parallel_poisson` validation. The optimistic estimate mode was removed as a fail-closed API change.

---

## 3. Work batches — grouped by file/subsystem so related fixes land together

Effort is engineer-days for implementation **plus** the regression tests named in the findings.

### Track A — Accounting (blocks everything that prints an ε)

**A1. `opaque-accounting` core Python** — 5d
`core/calibration.py`, `core/_accountant.py`, `core/composition/_cached.py`, `core/discretization.py`, `core/_process_codec.py`, `opaque_accounting.pyi`
Covers RC-16 + RC-6's generation counter. Includes: iterative serialize/load/`__eq__` (mirror `iter_hash`), idempotent `cached()`, extend `test_iter_hash.py` to assert `state_dict`/`==` at depth 10,000.
*Blocks: everything.*

**A2. `opaque-accounting` Rust mechanisms + numerics** — 4d
`mechanisms/eps_delta.rs`, `mechanisms/identity.rs`, `discretization/connect_the_dots.rs`, `numerics/fft.rs`, `pld/pmf/dense.rs`
Ceiling rounding, unconditional conservative PMF construction, `PyResult` on `self_compose`, and the u32 guard. The optimistic estimate mode was deleted rather than retained as a compatibility no-op.

**A3. Monte-Carlo upper-confidence-bound PLDs** — 10–12d ⚠ hardest single item
`amplification/balls_in_bins/monte_carlo.rs`, `identity.rs`, `mc.rs`, `b_min_sep/registry.rs`, Python wrappers, `docs/reference/accounting.md`
RC-4. Needs statistical design review, not just coding. Deliverables: UCB PMF construction, confidence level plumbed through `DiscretizationConfig`, closed-form/quadrature cross-check for BnB-Identity at small b,E at the **default** 100k budget, and a test asserting ε *increases* as `num_mc_samples` decreases.
*Blocks: b-min-sep monotonicity restoration, issue I sandwich tests, any BnB/b-min-sep ε claim.*

**A4. DP-SGD amplification API hygiene** — 1d
`poisson(1.0)` special-case, `parallel_poisson` delegating to `poisson()` + `num_workers` validation, `TestParallelPoissonCrossValidation` made non-vacuous against dp_accounting or the Rust golden vectors.

### Track B — Mechanisms (the noise that actually gets released)

**B1. `_gaussian.py` sampler + bounded mode** — 5d
RC-2, all of it, plus the empirical-max regression test (≥1e8 draws, assert not pinned) and the realized-std reporting.
*Blocks: any DP-SGD ε claim.*

**B2. AdaClip end-to-end** — 8d
`transformations/adaclip.rs`, `dpsgd/clipping/_adaptive.py`, `engine/clipping/_clipped_fun.py:569-586`, `accounting/dpsgd/mechanisms/_adaclip.py`, `_dp_trainer.py:3832-3847`, `examples/train_{causal_lm,sft,dpo}.py`, `tests/accounting/test_cross_validation.py:393-421`
RC-3. Ship with the MC sensitivity test the finding names: empirically estimate the per-record sensitivity of the released rate, assert ≤ what `adaclip()` charges.

**B3. DP-FTRL noise engines** — 10d
`_lambda_cgd.py`, `_engine.py`, `_bisr.py`, `_second_moment.py`, `_blt.py`, `_band_mf.py`, `_mf_gaussian_noise.py`, `sampling/_balls_in_bins.py`
RC-5 + RC-6 + the second-moment budget split (use deployed-participation sensitivities; the extra L-BFGS solve disappears) + lazy/closed-form `row_l2_at` + BnB empty-bin emission.
The **covariance test** is the deliverable that matters here.

**B4. Clipping correctness** — 4d
`engine/clipping/_pytree.py` (nested per-group; bf16 scale rounding — multiply by `nextafter(s, 0)` so the stated `‖out‖ ≤ C` guarantee actually holds), `engine/types.py` `PerGroup` immutability + fingerprint-based MF latch, `_clipped_fun.py` microbatch accumulator dtype + the `return_aux=False` per-example materialization.

### Track C — Distributed / engine

**C1. gloo 2-rank CPU lane** — 4d — **do this before C2**
See G1. Reuse `packages/opaque-transformers/tests/distributed/_ddp_runner.py`.

**C2. Distributed correctness** — 6d
`engine/distributed/_state.py`, `engine/distributed/gradients.py`, `engine/clipping/_distributed.py`, `engine/profiling/_distributed.py`, `_dp_trainer.py:2596-2604`
RC-7 + RC-8. Land the fail-closed policy (§1.2) as the first commit of this batch, then each individual fix is small and *verifiable* on the new lane.

**C3. Serialization & precision** — 3d
`base/serialization/_dispatch.py`, `engine/serialization/_structural.py`, `engine/precision/_loss_scaler.py`
RC-7 remainder. `all_finite` needs the per-example finite mask surfaced from `clipped_fun` **before** `clip_pytree`'s `nan_to_num` — otherwise the check is structurally impossible, not just mis-dispatched.

### Track D — Auditing (self-contained; can run fully parallel)

**D1. Estimator rebuild** — 8d — `one_run/_gdp.py`, `_estimate.py`, `_roc.py`, `_eps_delta.py`
**D2. Type-I calibration + negative controls** — 5d — G6
**D3. Canary construction + key separation + façade + docs** — 4d — `_coin_flip.py`, `attacks/_helpers.py`, `docs/user-guide/auditing.md`, `auditing/types` re-exports
*D1 blocks D2 blocks any "we audited this" statement anywhere in the repo or docs.*

### Track E — Optimizers

**E1. Numerical correctness** — 5d
`_adafactor.py` (`eps_root` absolute floors firing for 100% of coordinates — Adafactor is currently RMS-normalized SGD with a no-op bias correction), `_adam.py` second-moment scale mismatch (Σgᵢ² vs (Σgᵢ)², ~√B effective-LR difference despite being documented as interchangeable), `phi` dropped on checkpoint restore across 6 optimizers, `_adadelta.py` allocation, `_chain.py` `update_rms_clip` scope.
**E2. De-vacuum the tests** — 3d — `tests/test_adam.py` low-SNR regime, BC state_dict round-trip, real Adagrad gradients, schedule across save/restore.
**Note:** E1's Adafactor fix invalidates the empirical claims at `docs/user-guide/optimizers.md:221,265` — re-measure (needs Track G harness) before restating.

### Track F — Trainers / alignment / patches

**F1. `DPTrainer`** — 7d — RNG split, telemetry boundary (RC-9), uneven eval gather, fractional epochs, `load_best_model_at_end`, `from_hf` fused-optim, stop-at-ε per-step evaluation, SFT `dft` eval loss, unwired `privacy_noise_radius`/`log_level*` (wire or delete).
**F2. Alignment plumbing** — 6d — chat-template Gemma span detection (**silently marks user/system tokens as assistant** — a training-data correctness bug with privacy consequences), ref-logprob cache fingerprint, `compute_ref_logprobs_for_dataset` sharding, `null_ref_context` adapter, fused/eager token-count dtype, metric docstrings.
**F3. Alignment losses (issues A, E, K–P)** — 8–10d — SquareChiPO DP construction (or permanent claim withdrawal), BCO delta contract, MPO normalization, f-divergence α inversion + cap, LD-DPO completion-relative positions, DiscoPOP fp16 clamp, 14 citation fixes. **Parity tests against TRL at σ=0/C=∞ are the acceptance criterion** (issue F).
**F4. Patches** — 8d — sliding-window attention collapsed to full causal (**silent numerical divergence for every sliding-window family**), Gemma2 softcapping dropped, CE backward overwriting `outputs.logits`, checkpoint shim disabling `torch.func` process-wide, LoRA in-place dX / lite-vmap reshape / `up_proj` bias / `in_dims` validation, router dropout+batchify ordering, `fused_moe.py` false claims (issue V).
**F5. Patched-vs-unpatched parity suite** — 4d — see G7; this is what makes F4 verifiable.

### Track G — Infrastructure (see §4 for the mapping)

**G-build. Benchmark harness** — 6d, then 3d to restate/retract every number.
**G-ci. CI overhaul** — 5d.
**G-docs. Snippet execution gate + doc sweep** — 7d.
**G-cite. Bibliography + arXiv check** — 4d.
**G-pkg. Packaging/supply chain** — 4d.

---

## 4. Dependency order — what must exist before what can be tested

```
                    ┌─ A1 (accounting core) ──────────────┐
Phase 0 ────────────┤                                     ├──> any ε claim
                    └─ A2 (Rust mechanisms) ──────────────┘
                                 │
                                 ├──> A3 (MC-UCB) ──> b-min-sep/BnB claims, issue I sandwich tests
                                 │
   G2 (conformance harness) ─────┼──> B1 (gaussian) ──> DP-SGD ε claims
                                 └──> B2 (AdaClip)  ──> AdaClip ε claims

   C1 (gloo lane) ──────────────────> C2 (distributed fixes)   [C2 is untestable before C1]

   D1 (estimator) ──> D2 (Type-I suite) ──> any audited-ε statement, issue D, issue O

   G-build (harness) ────────────────> every performance claim (RC-14, W, J)

   G-docs (snippet gate) ────────────> the ~40-site doc sweep (RC-11)
                                       [sweeping first = drift returns next release]

   RC-1 SamplingContract ────────────> quickstart/tutorials/examples rewrites
   RC-7 fail-closed policy ─────────> C2, C3, B4  [land the policy commit first]
```

**Five hard orderings, stated plainly:**

1. **The auditor before any audited-ε claim.** D1 must land before D2, D2 before anything in docs or README says a number was empirically validated. Issue O also means the audit harness in `train_dpftrl.py` is currently comparing two *different mechanisms* — fix that with RC-1 before trusting any output.
2. **The gloo lane before the distributed fixes.** Every RC-8 fix is a 10-line change that is impossible to verify today. Build the lane, watch it fail on the current code, then fix.
3. **The benchmark harness before any performance number.** Do not restate the memory-optimizations tables, the BISR table, the Adafactor comparison, or the second-moment overhead from memory.
4. **MC-UCB before b-min-sep/BnB claims and before issue I's sandwich tests.** The tests are skipped for 6 of 8 pairs *because* the estimates are non-conservative point estimates.
5. **The conformance harness (G2) before signing off B1/B2.** Otherwise you are trading one unvalidated mechanism for another.

---

## 5. Infrastructure that prevents recurrence

Each gate is listed with **the issue class it would have caught**, so the investment is justifiable.

| Gate | Build cost | Catches |
|---|---|---|
| **G1. gloo/CPU 2-rank blocking lane** — existing DDP scenarios + a full DP step (`sum_gradients_` → `sync(clip_state, aux)` → `sync(noise_state)`) with (a) one rank drawing an empty Poisson batch, (b) `second_moment=True`, (c) uneven eval shard, (d) rank-0-only profiler stage | 4d | All 8 RC-8 findings + issue H + the `sum_gradients` no-op + `gather_for_metrics`. Today **zero** multi-rank collective code executes in CI. |
| **G2. Mechanism↔accountant conformance suite** — for each deployed noise path, empirically estimate the hockey-stick divergence (or realized σ / released-statistic sensitivity) between neighbouring inputs and assert it is dominated by the PLD the accounting factory produces. Include bounded mode with the centre several σ outside the interval, and adaptive with a Poisson batch < `expected_batch_size` | 6d | RC-2 (all), RC-3 (all), bounded-Gaussian stddev, the circular `test_adaclip_effective_noise`, "no test connects the deployed mechanism to the accountant that charges for it". **This is the single highest-value new test suite in the repo.** |
| **G3. Sampler↔accountant contract** (RC-1) + a test that the sampler drives exactly `n_steps` batches per configured run | 3d | All 9 RC-1 findings + issue O. |
| **G4. Executable docs** — `pytest --doctest-glob` / mkdocs codeblock exec over `docs/**/*.md`, `nbmake` over CPU-runnable notebooks, plus signature-binding of every factory call in every docstring | 7d | All of RC-11 (~40 sites), the notebooks published with `execute: false`, the `rng.md` calls that raise, the DP-FTRL factory docstrings. |
| **G5. `references.bib` + arXiv metadata resolution in CI** | 4d | All of RC-13 + issue E (14 files). Three fabricated titles currently live in the Rust accounting crate. |
| **G6. Auditor Type-I calibration suite** — 200 null runs, assert P(ε̂>0) ≤ α + binomial slack; m=5000 case to exercise rank truncation; assert truncated ≤ untruncated at higher `_MAX_EXACT_RANKS` | 5d | RC-10 (all), issue D. |
| **G7. Structural property tests** — round-trip `state_dict` over `dict(nn.Linear.named_parameters())`; `all_finite(clipped({'w':[inf]}))` is False; `sync`/`reduce_pytree` over every registered wrapper; patched-vs-unpatched forward+grad parity per model family (eager *and* sdpa, and a config with `sliding_window < seq_len`) | 6d | All 8 RC-7 findings + F4's sliding-window and Gemma2 bugs + the CE-logits overwrite. |
| **G8. CI gating fixes** — tests in `needs:` for both publish paths; blocking `ddp-correctness` leg; CPU reference lane for Triton kernels via the existing functional fallbacks; `trl` in the dev group; python 3.12 axis; a `--resolution highest` (torch 2.12) leg and a `transformers==4.57.*` leg; assert the CUDA leg executed >0 tests; assert no test module unexpectedly skips at import | 5d | RC-12 (8 findings) incl. "wheels ship from a red tree", "33 TRL tests never run", "version-adaptive branches never execute", "12 silent skips read as green". |
| **G9. Nightly privacy-regression workflow** (issue G — currently no scheduled workflow exists at all) — golden ε vectors vs `dp_accounting`/riskcal across every mechanism × amplifier pair, fail on drift beyond a tight tolerance | 4d | The entire "ε silently changed" class; would have caught the `eps_delta` rounding, `CachedProcess`, and `budget_exceeded` findings on the commit that introduced them. |
| **G10. Benchmark harness** (issue J) with committed results + hardware/commit in every table caption | 6d | RC-14 (all), issue W. |
| **G11. Extend `tests/contracts/`** — every NOTICE path resolves; documented defaults vs `dataclasses.fields`; `.pyi` vs pyo3 signatures; `test_all_exports_match` promoted from opaque-alignment to all 10 wheels (AGENTS.md rule 6 currently describes a file covering 1 of 10) | 3d | ~10 docs-drift findings, cheaply and permanently. This directory already exists — extending it is the lowest-friction gate in the list. |

**Total infrastructure: ~53 engineer-days.** It is roughly a third of the total programme and it is what stops this report from being written again in six months.

---

## 6. Sequence and milestones

Assumes **3 engineers**: one on accounting/mechanisms (statistical background required for A3), one on engine/distributed/infra, one on trainers/alignment/patches/docs. Calendar estimates assume normal review overhead.

### M0 — "Nothing is silently wrong" (week 1)
Phase 0 (§1) in full: the 12 one-liners, the fail-closed conversions, all 13 claim withdrawals, the two new `limitations.md` sections, PyPI name registration, release gating, fork-PR guard, `runs/` ignore.
**Exit:** ERRATA published; no shipped path returns a budget-violating parameter with `converged=True`; no zero-noise release path; no un-guarded empty-batch DDP desync.
*≈ 5 engineer-days across 3 people.*

### M1 — "The accountant is trustworthy" (weeks 2–4)
A1, A2, A4 · G9 (nightly golden-vector regression) · G11 (contract tests) · C1 (gloo lane) · G5 (bibliography) + the citation sweep.
**Exit:** ε from any non-MC path is reproducible, checkpointable at any depth, and cross-validated nightly against `dp_accounting`. A blocking multi-rank lane exists and is red.
*≈ 22 engineer-days.*

### M2 — "The mechanism matches the accountant" (weeks 3–7, overlaps M1)
G2 (conformance harness) → B1 (Gaussian) → B2 (AdaClip) · B4 (clipping) · C2+C3 (distributed + serialization, on the now-green-able lane) · RC-1 `SamplingContract` (G3) and the quickstart/tutorial/example rewrites.
**Exit:** every deployed noise path has an empirical conformance test against the PLD that prices it. The documented copy-paste training loops produce a valid ε.
*≈ 34 engineer-days.*

### M3 — "The auditor works" (weeks 4–8, fully parallel)
D1 → D2 (G6) → D3.
**Exit:** Type-I error at nominal level across 200 null runs; no hang; no NaN-manufactured ε̂; the "hard ceiling" and Gaussian-DP-family claims corrected; issue O resolved so the audit compares one mechanism to itself.
*≈ 17 engineer-days.*

### M4 — "DP-FTRL and MC amplification are sound" (weeks 6–10)
B3 (MF engines + covariance test) · A3 (MC-UCB) → restore the b-min-sep monotonicity claim by construction → un-skip issue I's sandwich tests at tight tolerance.
**Exit:** MF noise is provably correlated as specified; MC-derived ε values are upper confidence bounds with a documented confidence level and resolution floor; `epsilon_at(δ)` returns INFINITY below the MC floor rather than a finite number.
*≈ 22 engineer-days. A3 is the schedule risk — start the statistical design in M2.*

### M5 — "Trainers, alignment, patches" (weeks 7–12)
F1 · F2 · F3 (with the TRL parity gate that issue F says already exists) · F4 · F5 (G7 parity suite) · E1 · E2.
**Exit:** no un-accounted telemetry without an explicit opt-in; chat-template role spans validated by acceptance test; sliding-window and softcapping numerics match upstream to a stated tolerance; the SquareChiPO claim is either implemented or gone.
*≈ 44 engineer-days.*

### M6 — "It can't regress" (weeks 9–13, overlaps M5)
G4 (executable docs) → the RC-11 sweep · G8 (CI overhaul) · G10 (benchmark harness) → restate or delete every performance number · G-pkg (RC-15: manylinux, sdist, licenses, NOTICE, Unsloth attribution, transformers pyproject, Dependabot, `__pycache__`, `set_build_versions.sh`).
**Exit:** a docs snippet that does not run fails the build; a wheel cannot publish from a red tree; every published number has a script behind it.
*≈ 31 engineer-days.*

---

### Totals

| | Engineer-days |
|---|---|
| Phase 0 / M0 | 5 |
| Correctness work (M1–M5, excl. infra) | ~118 |
| Infrastructure (G1–G11) | ~53 |
| Docs sweep + citations + claims | ~19 |
| **Total** | **~195 engineer-days ≈ 39 engineer-weeks** |

**With 3 engineers: ~13 calendar weeks** to M6, with the critical safety content (M0–M2) landing in the first 7. With 1 engineer: ~9 months, and I would not run this library in production during that period without M0–M2 complete.

### The three things to cut if the schedule compresses

Cut **scope**, never **gates**. In order of what is safest to defer:
1. F3 (alignment loss corrections) — withdraw the claims, mark the losses experimental, defer the implementations.
2. G10 + all performance restatements — delete the numbers instead of re-measuring them.
3. F4's non-numerical patch issues (LoRA vmap `in_dims`, router ordering) — keep the sliding-window and softcapping fixes, they are silent numerical divergence.

**Do not cut:** G1, G2, G6, G9. Those four gates are the difference between fixing 170 bugs and fixing 170 bugs again.
