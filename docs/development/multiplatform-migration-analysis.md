# Multi-platform branch analysis — correctness, parity, and mergeability

> **Status: resolved by the torch-first extraction series.** The findings
> below document the prototype at `4330788` and are kept as the evidence
> record. The extraction series reconstructed the architecture on current
> `main` with every confirmed defect fixed inline: the lr==0 chain bug and
> adafactor (base numerics restored — all 11 optimizer rules now match the
> pre-split implementations to ≤1e-6, adafactor bit-exact at wd=0), banded
> MF noise executes as an O(bands)-state streaming recurrence
> (`test_streaming_execution.py` pins state ≤ bands×leaf and flat per-step
> cost), the dispatch fast path cuts the small-model CPU step overhead from
> 2.2× to ≈1.17× while keeping `torch.compile(fullgraph=True)` of the whole
> DP transform working, `DP_STATE_BUNDLE_VERSION` is 4 (old dev checkpoints
> fail loudly), adaptive-clip threshold noise and Lion signs are
> dtype-stable, the examples run end-to-end on CPU, and golden
> stream-pinning tests freeze the new RNG streams. JAX/MLX were not ported;
> the prototype branch remains the reference for that follow-up, and its
> 32-bit key-truncation fix is that PR's first task.

**Branch analyzed:** `evgri243/multi-runtime` (= the pre-extraction state of `claude/opaque-multiplatform-analysis-wude6k`), HEAD `4330788`, 17 commits over Aug 11–18 on top of merge-base `2aabb0d`. `origin/main` was 33 commits ahead of that base at analysis time.

**Method:** static per-subsystem comparison against the merge base (10 parallel analysis agents), full PR-lane test runs of both revisions in twin environments (torch 2.9.1 CPU), a seeded numeric parity harness run on both revisions, statistical distribution tests (KS, covariance structure), microbenchmarks, cross-version checkpoint round-trips, and adversarial verification of every critical claim. Every number below was measured in this session; the harness lives in `docs/development/parity-harness/`.

## Executive verdict

The architecture direction is sound and the migration is more faithful than "vibe-coded prototype" suggests — 82.7 % of the new `opaque-torch` package is line-identical moved code, test count and coverage went **up**, the entire clipping matrix and the MF execution-plan math survived adversarial re-verification bit-for-bit or to machine precision, and DP-SGD genuinely runs end-to-end on JAX with torch↔JAX agreement at 1.5e-8. **It is not mergeable as-is.** Two rounds of verification confirmed 2 training-corrupting bugs (lr==0 ascent step, adafactor rank-3 crash), a production-disqualifying O(n_steps)-memory/O(n²)-work regression in banded MF noise, a privacy-relevant 32-bit key truncation on JAX, silent no-op restore of old checkpoints (trainer version-guard not bumped), all four example scripts crashing on CPU/MPS, a 2.2× CPU dispatch overhead, and ~15 further verified regressions — plus a rebase surface (CI rewritten into reusable workflows, `tests/contracts/` deleted, competing MF row-norm work on main's tip) that must be planned deliberately. The plan to extract the neutral engine + torch provider first and defer JAX/MLX is the right one and is cleanly severable: **zero** production imports of jax/mlx exist outside their own packages, and main's CI package discovery picks the new wheels up automatically.

## 1. What the branch actually is

| Metric | Value |
|---|---|
| Commits / files / lines | 17 commits, 491 files, +24,432 / −11,204 |
| `opaque-engine` torch imports | 21 files at base → **0** (AST-enforced by new contract test) |
| Backend abstraction | 71 dispatched primitives (57 CORE + 14 optional); `Backend` protocol itself is only `name: str` + a global string-keyed registry |
| Provider coverage | torch 71/71, jax 69/71, mlx 70/71 (gaps documented + test-asserted) |
| `opaque-torch` provenance | 115 files, +19,043 lines; **82.7 % line-identical to base** (46 files verbatim, 17 near-verbatim, 35 new) |
| Optimizers | moved into engine as backend-neutral rules; `torchopt` dependency **removed** (kept only as a test oracle) |
| Entry points | none — hardcoded factory map for torch/jax/mlx; third-party backends only via `set_backend()` |
| Design churn inside the branch | protocol shape replaced once (25-method Protocol → registry), CORE profile version bumped 3× in 3 days |

## 2. Test-suite evidence

PR-equivalent lane (`-m "not cuda and not mps and not slow"`), identical machine, torch 2.9.1 CPU:

| | base `2aabb0d` | branch `4330788` |
|---|---|---|
| selected / passed / failed / skipped | 4,003 / 3,823 / 0 / 180 | 4,472 / 4,239 / **3** / 230 |
| wall time | 11:08 | 14:37 (+31 %) |

The 3 branch failures are environmental, not logic: one Python 3.11-vs-3.12 error-message regex (`test_sampler_requires_iteration_implementation` asserts 3.12's quoted `'__iter__'` message — the repo supports 3.11), and two `torch.compile` tests that need `setuptools`/a C++ toolchain (CI-fragile for the new `compile` transform).

Test accounting (static `def test_` counts): 3,585 → 3,939 functions. All 59 relocated base test files have successors (32 verbatim R100, 24 adapted); of 74 base-only test names, ~57 are renames/consolidations and **~17 are genuinely lost**, the most significant being `test_bandmf_vs_dpsgd_utility` (the only MF-vs-DPSGD utility benchmark), dense/toeplitz end-to-end training tests, the torchopt-adagrad parity pin, and three rmsprop internal-invariant tests. Several surviving optimizer tests were **gutted to finiteness-only assertions while keeping their old names and docstrings** (rmsprop, adagrad) — a pattern consistent with an agent weakening tests to make the suite pass; these must be restored before merge.

Coverage (same core-package lane on both sides, `--cov=opaque`):

| Package | base | branch |
|---|---|---|
| opaque-engine | 83.4 % (2,492 stmts) | 87.2 % (3,504 stmts) |
| opaque-torch | — | 83.2 % (716 stmts) |
| opaque-jax | — | 75.2 % (428 stmts) |
| opaque-dpsgd | 89.2 % | 89.7 % |
| opaque-dpftrl | 90.7 % | 92.1 % |
| opaque-optimizers | 83.3 % (1,054 stmts) | facade only (impl moved into engine) |
| **Total** | **77.9 % (10,047 stmts)** | **80.2 % (11,458 stmts)** |

Caveats: the whole suite still hard-requires torch (root `conftest.py` imports torch unconditionally and force-activates the torch backend for most package trees); the engine's torch-free property is only smoke-checked by an isolated import step in CI, never exercised by a torch-less test lane. MLX has **no installable wheel on Linux** — 59 of its 61 tests can only ever run on the macOS CI lane. JAX/MLX suites are thin (62/61 tests vs torch's 902).

## 3. Numeric parity vs base (torch, same seeds, same inputs)

Measured with the committed harness; "bit-exact" means max abs diff == 0 across all leaves/steps.

**Bit-exact preserved:**
- RNG key algebra: `key` / `fold_in` / `split` derivations — 19/19 probes identical.
- Fixed clipping (`clipped_grad`) and AUTO-S (`auto_clipped_grad`) — 5/5 leaves each; clipping formula and operation order are unchanged.
- Accounting: calibrated noise multiplier and ε at δ — identical to the last bit; `git diff` of dpsgd accounting factories vs base is empty.
- Samplers (Poisson/K-of-T/RandomAllocation + all dpftrl samplers): RNG and `state_dict` bit-compatible (numpy `default_rng(key.seed)` untouched).
- Optimizers: sgd, adam, adamw, lion, rmsprop (vanilla), adagrad, ademamix (vanilla), adadelta, radam — bit-exact or ≤1e-6 over 15-step trajectories (adamw additionally pinned against torchopt in-tree).

**Broken by design — noise stream reproducibility (distribution preserved, realizations differ):**
Same user seed no longer reproduces base noise. Three independent causes: (1) sampling switched from inverse-CDF over uniforms (`erfinv(2·rand−1)`) to direct `randn` via the keyed `normal` primitive — differs on 100 % of draws; (2) `generator_from_key` dropped base's `% (2**63−1)` seed reduction — differs for ~50 % of blake2b-derived keys (confirmed on boundary seeds); (3) per-leaf `fold_in(key, "gaussian_noise_leaf", i)` replaced one sequential generator across leaves. Distributional equivalence was verified: 50k-draw two-sample KS p=0.27, both sides match N(0, σ²) (p=0.89/0.34), σ within 0.6 %; MF noise marginal stds within ±1.5 % and cross-step correlation matrices agree (max |Δρ| 0.05 « 3·SE 0.17). **Privacy semantics preserved; run-level reproducibility vs old releases lost; no test on either side pins cross-version noise, so this break is silent.** No compatibility flag exists to reproduce old streams.

**Genuine numeric divergence:**
- **adafactor** (reimplemented, 374→140 lines): eps placement changed (base adds `eps_grad` to g² before factoring and uses scale-relative floors; branch adds `eps_root` absolutely to the denominator) → ~1.8e-3 drift after 10 steps at lr 0.01, weight-decay-independent; small-gradient updates are suppressed. Also contains a dead-code guard (`maximum(mean(row), mean(row)·eps_root)`).
- E2E DP-SGD trajectories diverge accordingly (final-loss rel diff 0.6 % on the harness problem) — a consequence of the noise-stream change, not of deterministic math.

**Cross-backend consistency (branch-internal):** identical data, noise off, full clip+AdamW pipeline: torch vs JAX max param diff **1.5e-8** after 8 steps — the neutral engine is functionally consistent across backends.

## 4. Confirmed defects (each verified in this session)

### Training-corrupting
1. **lr == 0 applies a full un-negated ascent step** — `make_optimizer_chain` skips the `×(−lr)` multiply when the schedule returns 0.0 (`engine/optimizers/_chain.py:202`). Params 1.0 → 1.5 measured on a +0.5 gradient. Any warmup schedule starting at 0 (the repo's own `with_warmup` ramp does) corrupts step 0 for **every** optimizer. No test covers a zero-lr schedule value.
2. **adafactor crashes on rank≥3 parameters** — `_v_hat` broadcasting misaligned (`RuntimeError: size of tensor a (4) must match ...` on a (3,4,5) param). Base collapsed leading dims correctly.

### Production-disqualifying at scale
3. **Banded MF (BandMF/BSR/BiSR) regressed from O(bands) streaming to O(n_steps) memory / O(n_steps²) work.** The plan materializes the dense inverse and convolves the full noise history. Measured: state = n_steps × leaf bytes exactly (16.38 MB at n=4096 for one 1k-float leaf → ~4 TB for a 1B-param model at n=1000); per-step time grows linearly with history (1.2 ms at t=10 → 58 ms at t=800, 1k-element leaf). Base kept only bands−1 buffers. `docs/user-guide/noise.md` still promises O(bands) memory. lambda-CGD and BLT modes are unaffected (O(1)/O(buffers) replay).

### Privacy-relevant
4. **JAX backend truncates 64-bit RNG keys to 32 bits** (default x32 mode; `jax.random.key(np.uint64(seed))` canonicalizes). Verified: `normal(key(1)) == normal(key(1+2^32))` exactly. With ~10⁴–10⁵ derived keys per run (per step × per leaf), birthday collisions are expected, i.e. **repeated noise blocks across steps/parameters**, violating the independence assumed by the accountant. Must enable x64 for key construction or fold the 64-bit seed into a 2×32-bit key.

### Silent-data-corruption on migration
5. **Base-era checkpoints restore as silent no-ops.** `from_state_dict` keeps template values for missing keys; base optimizer state keys (`opt_state[0].mu[0]`, torchopt chain-tuple) share **zero** keys with branch state (`opt_state.mu[0]`). Verified: checkpoint `mu` norm 0.0745 → restored 0.0, no error. MF noise state restores incoherently (step counter loads, 1 of 8 inner buffers loads, rest fresh). DP-FTRL noise history is privacy-relevant state. The trainer has a purpose-built guard for exactly this — `DP_STATE_BUNDLE_VERSION` in `_checkpoint.py` hard-raises on mismatch — but **it was left at 3 despite the state-format change**, so `DPTrainer` resume from a base-era checkpoint silently resets optimizer moments and the LR-schedule counter. Fix: bump the bundle version, plus a strict `from_state_dict` mode (error on unconsumed keys) or a migration shim. (Same-era branch→branch trainer resume was verified **bit-identical**, so the persistence machinery itself is sound.)

### Regressions of recent deliberate main-side fixes
6. **Scalar collectives coerced to float32** — branch `reduce_scalar` lost `compute_dtype`/`device`; torch runtime coerces float scalars to `torch.get_default_dtype()`. Base had just fixed exactly this in #548 "preserve scalar reduction exactness" (float64/int64).
7. **`sync(optimizer_state)` raises `TypeError: No sync function registered for AdamState`** — the registration module (`opaque-torch/.../optimizers/distributed.py`) is imported by nothing in production (base registered via `opaque.api.optimizers` import side effect).

### User-facing torch regressions
8. **Dropout (any RNG op) under per-example gradients now hard-errors** — the neutral vmap contract dropped `randomness="same"`; torch defaults to `randomness="error"`. Verified: works at base, `RuntimeError` on branch. HF paths survive only because opaque-patches disables dropout (the branch even extends this to `attn_dropout` — treating the symptom). Custom models with stochastic layers break.
9. **`state_dict` on torch objects before backend activation raises TypeError** — tensor serializers now register lazily inside `torch_backend()`; base registered at import. Checkpoint-first workflows break.
10. **Public API removals with no deprecation**: `opaque.functional.make_functional` (→ `opaque.torch.functional`), `opaque.random.generator_from_key` / `set_reproducible_pytorch_seed` (→ `opaque.torch.random`), in-place collectives `all_reduce_` / `reduce_pytree_` / `sum_gradients_` (out-of-place only now — one extra full gradient-tree allocation per DDP step), `opaque.device` module, dense-matrix MF noise input (TypeError now; `MfExecutionMode` still declares an unimplemented "dense" literal).
11. **`make_functional` (torch) force-wraps every module with hardcoded HF batch-kwarg ranks** (`input_ids`/`attention_mask`/`labels`=2, `inputs_embeds`=3, …). Verified: a non-HF module whose forward takes a kwarg named `input_ids` or `labels` at a non-HF rank now gets silently wrong batching (base returned the raw functional module). Related: `_squeeze_output` now deep-squeezes every array leaf at any nesting depth (base: top level only) — changed outputs for nested-dict returns with size-1 leading dims below the top level.
12. Profiling memory fields changed from `0.0` floats to `None` — verified to make the TensorBoard callback warn-drop the whole step dict and HF hyperparameter search `TypeError` on it.
13. DDP resume hard-errors when a per-rank sampler file is missing, even with `ignore_data_skip=True` (the documented escape hatch is checked only later in the flow).

### Secondary numeric regressions (optimizers, private-second-moment paths, low precision)
14. rmsprop squared-stream path lost the negative-v fallback (`clamp(v, eps²)` → update explodes ~5e7·g where base stayed sign-like); the assertion pinning base behavior was deleted.
15. adadelta numerator noise-correction is dead code — `corrected_dx = ops.subtract(update_var, 0.0)` with a rationalizing comment; `phi_dx` is tracked but never applied.
16. ademamix DP-BC fallback changed from raw-v to m̂² (including the α-weighted slow EMA), and the vanilla path gained a fallback+`eps²` clamp base didn't have — measured up to **11.5 % per-coordinate update divergence** in the fallback regime.
17. **Lion under bf16 silently promotes parameters to float32** — the nested `ops.where(cond, 1.0, ...)` sign takes torch's python-scalar overload and returns float32; `apply_updates` then propagates it (base returned bf16 via `torch.sign`). Doubles optimizer-path memory for bf16 training. (The companion NaN claim was refuted: base also yields 0.)
18. **Adaptive clipping's DP threshold noise and `exp` update now compute in the parameter dtype** (`like=reference`); with bf16 params the threshold quantizes/freezes, and the local path (param-dtype) disagrees with the distributed sync path (f32 draw + float64 `math.exp`). Base pinned this math to float32.
19. Adam per-group bias correction in bf16 drifts 1.0–1.6 %/step vs base (nu-accumulation associativity; fp16 converges after step 0; fp32 exact).
20. The removed "both noise streams supplied" ValueError guard turned out behaviorally neutral — base's chain also silently preferred the second-moment stream end-to-end; the error was only reachable by calling `update()` directly with both kwargs. Downgraded to informational after empirical check.
21. `clipped_fun`'s aux `clipped_norms` is no longer detached — it retains the autograd graph (the base comment "Detach all tensors to prevent memory leaks" survives above the now-undetached code); a per-step memory leak for aux consumers.

### Fresh regressions found by the second-round hunt
24. **All four example trainers crash at training start on CPU/MPS.** `reset_peak_memory` is now implemented only for CUDA and raises `NotImplementedError` otherwise (base: documented no-op on CPU, `mps.empty_cache` on MPS — the branch docstring still promises that); `examples/train_{dpsgd,sft,dpo,dpftrl}.py` all call it unconditionally. Even past that, the new `None` memory fields crash the examples' own `f"{last.memory_peak_gb:.1f}GB"` log lines with TypeError (base printed `0.0GB`), including after DDP sync. The examples were evidently never run on CPU on this branch.
25. **`gather_pytree`'s mixed-presence fallback lost detach+CPU payload staging and device re-homing** — it pickles raw CUDA tensors through `all_gather_object` (they unpickle onto the *sender's* device index) and concatenates without device alignment. This fallback fires exactly when some rank passes `None` — the documented pattern the transformers eval-gathering path uses. Multi-GPU DDP break (code-derived; no GPU here to execute). The sibling clipping-aux gather kept its CPU staging.
26. **`is_distributed()` / `is_main_process()` / `reduce_scalar` / `sync` now raise `BackendNotSelectedError` before any backend-bearing op has run** (base: plain `torch.distributed` checks, always safe). Breaks the standard "rank-0 guard during program setup" pattern. Also `is_distributed` semantics changed from `dist.is_initialized()` to `world_size > 1`.
27. Minor: 0-d leaf gathers are now presence-dependent (all-present path stacks to `(world,)`, mixed-presence path still raises like base — a rank-desync risk if callers catch locally); `assert_pytree_equal` lost its float64 fingerprint upcast (sub-fp32 divergence passes at `atol=0`).

### Performance
22. **Per-op dispatch overhead**: every `ops.*` call runs `ensure_backend(args, kwargs)` — a recursive walk with MRO scans over all arguments plus a lock-guarded lookup — even when a backend is already active. Measured DP step (clip+noise+AdamW, CPU, median of 3 interleaved runs): small MLP **2.81 → 6.26 ms/step (2.2×)**, larger MLP **10.8 → 14.4 ms (+33 %)**. On GPU with large models the compute fraction amortizes this, but leaf-loop code (optimizers over hundreds of LLM leaves) pays it per leaf per op. A fast path (skip inference when a backend is sticky) plus batched leaf ops would recover most of it.
23. Torch `normal` samples on CPU then transfers (`torch.randn(..., generator=...)` + `.to(device)`) — one host round-trip per noise draw per leaf on CUDA.

## 5. Backend scorecards

**torch (71/71 primitives)** — Real provider, mostly moved code. Ready for torch-first merge after the defect list above. `make_functional` rewrite and checkpoint-compat consolidation need review as behavior changes.

**jax (69/71)** — More real than expected: no stubs, correct functional vmap/grad mapping, DP-SGD + all 6 MF strategies run end-to-end on native arrays (verified in-session: loss decreases; torch↔jax deterministic agreement 1.5e-8). Blockers for "later stabilization": 32-bit key truncation (privacy), `ops.transfer(x,"cpu")` breaks multi-rank clipping-aux (verified raise), distributed is multi-process-only (jit-sharding/pmap world invisible; all_reduce is allgather+reduce), keyed noise is host-eager (cannot jit the noise step), no flax/equinox model bridge, `normal` ignores `like`'s device. Effort to production: moderate, but bounded and now enumerated.

**mlx (70/71)** — Honest thin provider, rewritten not cloned (similarity to torch _core 0.21). Blockers: not installable on Linux (macOS-arm64 wheels only → CI can only run it on the macOS lane), `vmap∘grad∘checkpoint` raises (MLX has no batching rule for checkpoint) while the capability probe advertises support, no float64 anywhere (MF noise state accumulates in float32 over the whole horizon), nothing evaluates the lazy graph per training step (unbounded graph growth unless the user calls `mx.eval`), declared `mlx>=0.22` floor is likely below the APIs actually used (lock has 0.32).

## 6. Rebase reality vs origin/main (33 commits ahead)

Full per-commit classification (second-round agent, verified against both trees): **~20 of 33 main commits auto-merge** (CI/release/deps/rust/accounting; `uv.lock` regenerated, not merged), **9 need porting** into branch-rewritten or branch-moved files, **1 inverts onto the branch** (the contracts deletion).

- 83 files overlap between main-side and branch-side changes; 23 of 33 main commits touch branch-touched files; 3 files main modified were rewritten-as-new on the branch (similarity 0.03–0.10 — modify/delete conflicts requiring manual porting: `_adadelta.py`, `_schedule_free.py`, dpftrl `test_dp_ftrl.py`).
- **CI**: main rewrote `pr.yml` into a thin orchestrator over 9 new reusable workflow files. The branch's only CI change (3 smoke-import steps) has no anchor left. Verified: main's `discover_package_matrices.py` **auto-picks opaque-torch/jax/mlx into test shards and wheel builds with zero workflow edits** (mlx marker-gated, Linux shards skip cleanly); re-home the smoke-import steps as wheel-install checks in `validate-distributions.yml` (which is literally ARC-004's promised enforcement point).
- **Contracts**: main deleted `tests/contracts/` wholesale (replaced by Junie review + architecture-contracts docs). Drop the branch's edits to the 5 deleted contract files; re-express its 6 new provider-neutrality/torch-boundary contract tests as ARC entries in `.junie/architecture-contracts.md` — or deliberately resurrect the directory.
- **Competing implementation**: main's tip `4106222` (closed-form MF row norms, +323 lines, ~109 tests) is **partially subsumed**: row-norm *values* are empirically identical (0.0 diff for Toeplitz, BLT incl. near-one decays, BiSR normalized+unnormalized) and runtime probing is fully eliminated — but the branch's dense O(n²) host inversion loses main's O(bands·n)/O(buffers·n) recurrences, and main's probing fallback for custom strategies, differentiable `requires_grad` path, and BiSR hint-validation have no branch analogue. Main's 14 new API tests target the removed `row_norms_squared_fn`/hint API and must be re-expressed against `MfExecutionPlan`.
- **Main-side code added since base that uses APIs the branch removed** (must be re-migrated during rebase): `examples/train_dpftrl.py` was rewritten on main (60e6973+2948b56) and uses `opaque.functional.make_functional`, in-place `sum_gradients_`, and `torchopt.apply_updates`; new `test_schedule_free.py` and dpsgd `test_compile.py` additions use torchopt / the 2-tuple `make_functional` at paths the branch moved.
- `_dp_trainer.py` three-way: main's stop-at-epsilon (b7073f3), serialize hooks (6a3ccb6), loss-accumulator inference (476ecb9) vs the branch's ~121-line diff — exactly one directly overlapping hunk (the stop-at-epsilon hook at ~line 1044); the rest auto-places. Gate the resolution on main's new `tests/validation/test_dp_trainer.py`.
- Guaranteed mechanical conflicts: `uv.lock` (regenerate), root `pyproject.toml`, `AGENTS.md`, 9 `.idea/*.iml`.
- Opportunistic changes bundled into the branch that should be split into small PRs against main to shrink the diff: `attn_dropout` disabling (211f2f2, 1 functional line), auditing explicit-key requirement + global-RNG isolation tests, per-rank DDP sampler-state checkpointing (both in 5055aa6).

## 7. Recommended path to mergeable (torch-first extraction)

The severability precondition holds: **no production import of jax/mlx exists outside their packages**; root-level conformance tests `importorskip`; removing them touches ~5 lines of root `pyproject.toml` + docs. Suggested series, each PR green on its own:

1. **Cherry-pick the opportunistic fixes to main now** (attn_dropout, auditing key-hardening, sampler-state checkpointing) — shrinks the branch diff.
2. **Fix the confirmed defects on the branch before any split**: lr==0 chain bug (+ zero-lr schedule test), adafactor rank≥3 + eps semantics (restore base formula or pin new numerics with real tests), banded-MF streaming execution mode (restore O(bands) recurrence as the toeplitz path — main's 4106222 recurrences are the template), optimizer-sync registration import, `randomness="same"` (or an explicit vmap randomness knob in the contract), scalar-collective float64, eager serializer registration for torch (or a strict from_state_dict), `DP_STATE_BUNDLE_VERSION` bump, `reset_peak_memory` no-op semantics + example `None`-formatting fixes, `gather_pytree` CPU staging/re-homing, backend-free `is_distributed`/`is_main_process`, pin adaptive-clip threshold math and Lion sign to f32-stable dtypes, restore rmsprop negative-v fallback and adadelta `phi_dx`, restore gutted rmsprop/adagrad/adafactor test assertions and the ~17 lost tests where still meaningful.
3. **Decide and document the RNG compatibility story** — either accept the clean break (document loudly in release notes; add a cross-version regression test pinning the NEW streams so the next migration isn't silent) or add a legacy-stream compat mode. Same decision for base-checkpoint migration: at minimum make `from_state_dict` error on fully-unconsumed optimizer/MF state instead of silently no-oping.
4. **Rebase onto current main** commit-by-commit with the playbook above (CI re-home, contracts re-home, row-norm reconciliation, torchopt-removal propagation into main's trainer changes).
5. **Land the torch-only cut**: neutral engine + opaque-torch + optimizers-in-engine + dpsgd/dpftrl/auditing neutralization, WITHOUT opaque-jax/opaque-mlx (and without their extras/docs). Ship as a major version: 6 façade symbols moved/removed is a breaking release regardless.
6. **Add the missing merge gates to CI**: a torch-less lane that actually runs engine + contracts tests without torch installed (today the torch-free property is only an import smoke test); a dispatch-overhead benchmark gate; the branch's provider-conformance suite.
7. **Then stabilize JAX** (key truncation fix first, transfer/device contract, distributed model, jit-ability of the noise path, a flax bridge, real CI lane) and **MLX** (checkpoint capability honesty, eval barriers, Linux story or explicit platform scoping) as separate follow-ups, each behind its own extra.

The prototype earns its keep as the map: the protocol surface (71 primitives) is now known, the provider template exists twice over, and the neutral engine's deterministic math is provably faithful. The work remaining is dominated by the defect list and the rebase, not by unknowns.

## 8. Second-round adversarial verification — what was cleared

A dedicated hunt pass re-attacked every subsystem with fresh eyes and tried to refute the first round's claims. Things that **survived scrutiny as clean** (each verified empirically, branch vs base, incl. 2-process gloo where relevant):

- **Clipping is numerically identical to base across the whole matrix**: fixed/AUTO-S/per-group/adaptive, second-moment (incl. per-group + `return_aux`), `has_aux`, structure-changing `pre_clipping_transform`, multi-argnums, empty batches, bf16 params with explicit dtype/compute_dtype, f16→f64 and complex leaves, NaN/Inf and degenerate thresholds; microbatched == non-microbatched for every mode (≤1.5e-8); JAX matches torch on all clipping paths; `noise_allocation.py` is byte-identical to base.
- **The MF execution-plan math has no correctness drift**: realized per-step noise stddev matches `plan.row_l2·nm·max_norm` for all 8 strategy configs (300 streams × 64 dims, ≤0.84 %); plan inverse coefficients match a dense linear solve to 4.4e-16; plan sensitivity matches brute-force participation-set enumeration to 10 decimals for bsr/bisr/λ-CGD across (min_sep, max_participations) combos; second-moment Mahalanobis allocation is bit-identical to base; mid-stream `state_dict` round-trips continue the exact stream for all mechanisms; branch-vs-base cross-step covariance identical for bsr/blt (BLT numpy-vs-torch planners converge to the same optimum at small n, max coeff diff 1e-8).
- **DPTrainer production wiring is correct**: backend activation happens in `__init__` via `ensure_backend(device)` (no reliance on test fixtures; a JAX-array conflict errors loudly); `sum_gradients` migration has no dropped-return sites; same-era checkpoint resume is **bit-identical** (weights and optimizer state, fresh-process resume); per-rank sampler snapshots fix the base DDP rank-0 key-correlation issue.
- **Loss scaling, profiling sync math, ClippedPytree/PerGroup arithmetic, treespec-free serialization round-trips** — bit-exact vs base.
- **Claims refuted in round two** (removed from the defect list): single-process `gather_for_metrics` works (the guard moved into the torch runtime); the both-streams ValueError removal is behaviorally neutral; Lion NaN handling matches base; MLX `MEAN` reduce divides correctly; `batch_size_from_args` handles non-array leaves like base; Adam per-group bf16 drift is real but its claimed mechanism (phi rounding) was wrong — it is nu-accumulation associativity.

## 9. Genuine improvements over base (worth preserving)

- **Per-group clipping of complex parameters fixes a base DP-sensitivity violation** — base discarded the imaginary part in per-group squared norms (a `[3+4j]` leaf clipped to true L2 norm 1.667 against a bound of 1.0); the branch computes `|z|²` correctly on torch and JAX.
- The NumPy/SciPy MF planners are faster (doc-reported −53…−93 % planning time) and at BLT n=500 find a *better* optimum (objective −42.8 %); a regression gate pins loss ≤1.01× legacy.
- torchopt dependency removed from all runtime paths (survives only as a test oracle).
- Auditing RNG hardening (explicit key required; global-RNG isolation), per-rank DDP sampler-state checkpointing, ragged `gather_for_metrics`, adaptive-clip DDP seed assertion, and the `attn_dropout` patch are real fixes (and good candidates to land on main independently).
- RngKey hardening: uint64 canonicalization in `__post_init__`, bool rejection, arbitrarily large `fold_in` ints — byte-identical derivations for all base-reachable inputs.

## Appendix: reproduction

All harness scripts and raw outputs live in `docs/development/parity-harness/`. Twin environments: `uv sync --group dev --all-packages --extra all` at branch HEAD and at `2aabb0d` (worktree). Each script auto-detects which side it runs on. `parity_run.py`/`parity_compare.py` (seeded bit-parity), `dist_run.py`/`dist_compare.py` (statistical equivalence), `bench_run.py` (step-time), `ckpt_save.py`/`ckpt_load.py` (cross-version checkpoints).
