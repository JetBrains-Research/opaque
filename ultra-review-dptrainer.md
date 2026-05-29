# Ultra-review — `feat/dptrainer-main-integration`

**Reviewer:** automated deep review (privacy-first, no-surprises lens), standing in for the repo owner.
**Scope:** full branch diff vs `main` (~24.5k insertions, 86 files) — the HF-style `DPTrainer` and its supporting modules under
`packages/opaque-transformers/src/opaque/api/transformers/trainer/`, the example, docs, and the test suite.
**Method:** the DP hot path traced by hand against the Opaque DP-engine contract; periphery reviewed module-by-module; two critical
claims confirmed empirically in the worktree venv.

Line numbers are anchors against the branch at review time and will drift.

---

## Verdict

The engineering is genuinely good. The DP-SGD / DP-FTRL math in the **hot path is correct**: `clip → (DDP sum) → finite-check →
noise → optimizer` ordering is right; the accountant composes exactly `total_steps` mechanisms; resume uses proper heterogeneous
composition; DP-FTRL noise and accounting are coupled through a single amplifier so they cannot desync; the DDP step is correct
central DP-SGD (sum-then-shared-noise-then-pure-optimizer). HF-parity plumbing (checkpoint cadence, eval flags, callbacks, LR
schedules, reporting) is careful and well-tested.

**The dominant structural problem:**

> **The test suite does not guard the privacy guarantee.** It rigorously tests HF-parity plumbing, but a bug that silently
> *weakened DP* — wrong σ, miscomposed accountant, unclipped gradients, noise scaled by the wrong sensitivity, eval spending
> budget — would pass the entire suite green. For a vibe-coded DP trainer this is the #1 risk: the code looking correct today does
> not protect against the next AI-generated refactor silently breaking it.

Two **confirmed silent defects** also exist (fp16 overflow detection is dead; resuming from a `save_only_model` checkpoint reuses
the noise stream). Both are ~one-line fixes; the test gap is the structural work.

Severity legend: **CRITICAL** = silent privacy/correctness break · **HIGH** = crash/hang or contract violation in normal use ·
**MEDIUM** = real bug, narrow trigger or non-silent · **LOW/UX/doc** · **NIT**.

---

## META — the test suite cannot catch DP regressions (highest priority)

Rigorous on HF-parity plumbing, but the privacy guarantees are essentially unguarded. Nothing in the suite:

- ever runs the noise-calibration solver — every "target ε" test also pins a fixed `privacy_noise_multiplier`;
- checks reported ε against an independently-computed reference, or that the accountant composed exactly N mechanisms;
- verifies clipping actually bounds a per-example gradient norm (tests only assert the *configured* threshold is echoed in metadata,
  e.g. `test_training.py::test_clip_norm_from_clip_state`);
- verifies realized noise ≈ `noise_multiplier × clip_norm` (tests only assert "noise changed the gradient");
- asserts eval consumes zero budget;
- asserts resume gives `prefix + remaining == target ε` for the **calibrated** case (only directional `ε grew` checks exist);
- asserts DDP ε == single-process ε, or cross-rank bit-identical parameters;
- exercises the fp16-overflow path, truncated-Poisson accounting, or per-group clipping metric math.

Most "DP math" tests exercise the underlying `opaque` library directly (hand-rolled `clipped_grad`/`gaussian_noise` + manual SGD),
not `DPTrainer.training_step`.

**Highest-value tests to add:** calibrated-ε vs reference; accountant composition count (and prefix+remaining on resume);
clipping bounds real norms; realized σ ≈ σ·C (scalar + per-group); eval spends no budget; DP noise reproducibility at σ>0;
DDP ε-equivalence + cross-rank param bit-identity; fp16-overflow contract; truncated-Poisson + adaptive-clip accounting.

---

## CRITICAL

### C1 — Training-resume from a `save_only_model` checkpoint reuses the noise stream → silent privacy violation
`_dp_trainer.py:3789–3798` gates optimizer / `dp_state` / RNG saves behind `save_only_model`, but `_save_accountant` is
unconditional (`:3788`). On resume, `_read_runtime_for_resume` sees no `dp_state.pt`, `_train_once:839–843` installs only the prefix
accountant and trains on. Noise state is rebuilt fresh (`GaussianNoiseState(_step_counter=0)`, `:1163`) and the Poisson cursor
restarts at 0. Because noise for step *t* is `fold_in(key, _step_counter)`, the resumed run **reuses the exact noise vectors** from
the original run's first steps, on the same re-sampled data. An observer of both runs can subtract to cancel the noise and recover a
noiseless data-dependent gradient difference; the accountant reports a clean ε that does **not** bound the true loss. Reachable
(`save_only_model=True` is a common disk-saving choice) and silent.

**Fix:** when continuing training and the checkpoint has no DP runtime state but the prefix accountant is non-empty (real prior DP
cost), hard-error — `save_only_model` checkpoints are export-only. The empty-prefix warmup path
(`privacy_resume_without_accountant`) is the only sound continue-case, and even there the noise key should be re-folded by
`global_step_already_done`.

### C2 — fp16 dynamic loss scaling is completely non-functional; overflow is never detected (confirmed empirically)
`_dp_trainer.py:1630`: `grads_finite = all_finite(grads)` where `grads` is a `ClippedPytree`. `ClippedPytree` (types.py:229) is a
plain `@dataclass` with **no** optree registration anywhere in the repo; `all_finite` (`engine/precision/_loss_scaler.py`) walks
`optree.tree_flatten` leaves and only checks `torch.Tensor` leaves. Empirically in the worktree venv:
`optree.tree_flatten(ClippedPytree)` → **1 opaque leaf**; `all_finite(ClippedPytree-with-NaN)` → `True` (bug);
`all_finite(cp.pytree)` → `False` (correct).

Consequence for any `fp16=True` run: the scaler never backs off, the overflow short-circuit (`:1635–1651`) is dead,
`state.fp16_overflow_steps` stays 0, and NaN/Inf gradients flow straight into the noise mechanism and the optimizer update. Under
DDP the "any rank's overflow trips every rank" invariant is moot because it never fires anywhere.

**Fix:** `all_finite(grads.pytree)` (or register `ClippedPytree`/`NoisedPytree` as pytree nodes flattening to their inner tensors).
Regression test: force a real overflow (huge scale), assert the scale halves and `fp16_overflow_steps > 0`.

---

## HIGH

### H1 — DDP eval gather can deadlock on empty/uneven shards and on `eval_do_concat_batches=False`
`_dp_trainer.py:2212–2215` + `_eval.py` finalize/gather. `local_shard` can hand leading ranks an empty slice when
`len(eval_dataset) < world_size` or uneven; an empty rank yields `None` for preds/labels/inputs/losses and *skips* the
corresponding `gather_pytree` collective while other ranks enter it → NCCL desync/hang. Same class of bug when `finalize` returns
per-batch *lists* (`eval_do_concat_batches=False`): one `all_gather_object` per element, unequal counts across ranks.
**Fix:** agree on presence flags up front and gather on a fixed schedule; size-prefixed all-gather instead of per-element object
gather; disallow/pad `eval_do_concat_batches=False` under DDP.

### H2 — `cosine_with_restarts` crashes unless `(num_training_steps − warmup)` is divisible by `num_cycles`
`_scheduler.py:197–203` passes `decay_steps` into `with_restarts`, which raises when `decay_steps % num_cycles != 0`. e.g.
`num_training_steps=1000, warmup_steps=101, num_cycles=4` → 899 % 4 → crash at build time. Tests only use evenly-divisible cases,
so it's masked. HF handles any combination via fractional progress.
**Fix:** round the restart window up to a multiple of `num_cycles` (or make `with_restarts` tolerate a remainder); add a
non-divisible test.

### H3 — Functional vs `nn.Module` eval paths do not produce identically-shaped outputs (violates documented contract)
The standard batched path collects all surviving output tensors and applies `ignore_keys`; the per-example path
(`include_for_metrics=["loss"]`, `_dp_trainer.py:1976–1982`) returns only `output.get("logits")` and **ignores `ignore_keys`**. For
multi-output models (seq2seq/vision) enabling per-example losses silently changes `EvalPrediction.predictions` shape and bypasses
`ignore_keys`, breaking `compute_metrics`.
**Fix:** route both paths through the same output-collection/`ignore_keys`/collapse logic, or document per-example as logits-only and
warn when aux outputs are dropped.

### H4 — Sampler resume is O(consumed) replay, contradicting the documented O(1) cursor
`opaque-dpsgd/.../sampling/_poisson.py:226–229` replays `_sample_step()` `consumed` times to reposition the numpy generator; the
trainer docstring (`_dp_trainer.py:1291–1293`) and the design doc claim O(1) "no batch replay." Functionally correct but a resume
deep into training pays O(consumed × dataset).
**Fix:** PCG64 `advance()` for a true O(1) jump, or correct the docs.

### H5 — DDP Poisson sampler uses a non-rank-folded key → correlated cross-rank subsampling; comment is false
`_dp_trainer.py:2865` passes `key(data_seed or seed)` unchanged on every rank; `PoissonSampler` seeds purely from `key.seed`, so
every rank draws the *same* local-position mask over its disjoint shard. The sampler's own docstring prescribes `fold_in(key, rank)`.
The comment at `:2830` ("the same key on every rank so the union … is a single global Poisson draw") is **wrong** — records at the
same local offset across ranks are perfectly co-included/co-excluded; this is not an i.i.d.-Bernoulli(q) draw over the dataset.

**Privacy-accounting assessment (reviewer):** this does **not** appear to under-state example-level DP. Shards are disjoint, so any
record lives on one rank at one offset and its marginal inclusion is still Bernoulli(q) independent of the noise; subsampled-Gaussian
amplification for the differing record depends only on that marginal, not on the joint distribution of other records. So
`acc.poisson(gaussian(nm), q)` stays valid for add/remove/replace-one DP. **Still a real bug:** the realized sampling is
statistically degenerate (low batch diversity → worse SGD) and it violates the documented contract with a false reassuring comment.
**Fix:** `sampler_key = fold_in(key(...), self._ddp.rank)` when `world_size > 1` (consistently on resume), correct the comment, add a
DDP independence test. Recommend the Opaque accounting authors sign off on the "accounting unaffected" claim — treat as
needs-confirmation rather than settled.

---

## MEDIUM

### M1 — Epoch-driven resume + `ignore_data_skip=True` from a non-epoch-aligned checkpoint overshoots the target ε
When `max_steps` is unset, the only step ceiling is `max_steps>0 and global_step>=max_steps` (`_dp_trainer.py:1451,1473`). With
`ignore_data_skip=True` the sampler is rebuilt fresh (yields `total_steps`) but `start_epoch = global_step // steps_per_epoch`
re-runs the whole partial epoch. Resume at step 15 of a 30-step/3-epoch run executes 20 more steps (→35) while calibration
recalibrated for `remaining=15` → the run **exceeds `privacy_target_epsilon`** (the accountant stays honest; the run overshoots). The
normal path (`ignore_data_skip=False`) is correct because `StopIteration` bounds it.
**Fix:** add `if global_step >= ctx.total_steps: break` unconditionally, not only under `max_steps>0`.

### M2 — No write atomicity in checkpointing
Every artifact is written directly to its final path; no temp-file + `os.replace`, no completion marker. A crash mid-save leaves a
`checkpoint-N/` with a valid `accountant.json` but missing/truncated `dp_state.pt` — which `get_last_checkpoint` still selects,
routing into the C1 noise-reuse path (missing file) or an unpickling error (truncated).
**Fix:** write to a temp dir and atomically rename, or write a `COMPLETED` sentinel last and have resume reject checkpoints lacking it.

### M3 — `bf16_full_eval` / `fp16_full_eval` is a silent no-op during in-training eval
`_precision.py:63` casts `self._model`, but in-training eval runs through `ctx.trainable_params`/`frozen_params` (detached copies from
`make_functional`), so the cast doesn't reach the forward. Final (post-training, `nn.Module`) eval works. Inconsistent precision
mid-train vs final, no warning.
**Fix:** cast the ctx param dicts when `ctx is not None`, or skip-and-warn.

### M4 — Multi-dataset (dict) eval unsupported, but the plan claims it's implemented
`evaluate()`/`get_eval_dataloader` treat `eval_dataset` as a single dataset; a `{"val":…, "test":…}` dict (common HF pattern, listed
as "implemented" in the plan doc Phase 3) flows into `DataLoader(dict,…)` and fails confusingly.
**Fix:** loop per sub-dataset with prefixed metrics, or raise a typed "not supported" error.

### M5 — Per-example eval reuses *train*-discovered `batch_keys` against eval batches
`_dp_trainer.py:1969` builds `batch_args` from `ctx.batch_keys` (discovered from the train collator). If the eval collator emits a
different key set, `inputs.get(k)` yields `None` and vmap fails.
**Fix:** discover keys from the eval batch, or validate and error clearly.

### M6 — `noise_multiplier` drift warning false-fires on every calibrated resume
In calibrated mode the recalibrated multiplier legitimately differs from the saved one, so `_warn_on_arg_drift`
(`_dp_trainer.py:4179`) always emits "Resume arg drift on noise_multiplier … mechanisms differ." Cries wolf; trains users to ignore
privacy drift warnings (hiding genuine `target_epsilon` drift).
**Fix:** skip the `noise_multiplier` drift check when `noise_multiplier_source == "calibrated"`.

### M7 — `cyclic_poisson` and `sequential` sampling modes are dead/unreachable
Both appear in `build_sampler` (`_dpftrl.py:245–259`) and the `get_train_dataloader` docstring's "five supported modes," but
`_ALLOWED_SAMPLERS` (`_config.py:149–157`) permits neither, so `__post_init__` rejects them; `build_amplifier_factory` has no
`cyclic_poisson` branch anyway.
**Fix:** wire them into the allow-list with accounting, or remove them and the doc reference.

### M8 — `_config.py` advertises optimizer names that are rejected
The class docstring (`_config.py:46`) lists `adamw-bc` and `adamax` as accepted `optim` values, but neither exists in `_optim.py` —
both raise `ValueError` from `__post_init__`. DP bias-corrected AdamW is only reachable via
`optim="adamw", optim_args="noise_bias_correction=True"`.
**Fix:** add an `adamw-bc` alias (and decide on `adamax`), or correct the docstring.

### M9 — Cadence skip on cluster-empty Poisson rounds
`_dp_trainer.py:1361–1368`: when synced `batch_size==0`, `on_step_end` fires but `_maybe_log_save_evaluate` is skipped, so a
log/eval/save boundary landing exactly on an empty round is silently dropped (the optimizer still applies a DP-correct pure-noise
update). All ranks agree, so no divergence — a cadence surprise.
**Fix:** run the gate on empty steps or document the behavior.

---

## LOW / UX / docs

- **L1 — Dropping the `DP_INCOMPATIBLE_PARAMETERS` rejection table is a UX regression for a tool selling HF-compatibility.**
  Unsupported HF args (`gradient_accumulation_steps`, `deepspeed`, `fsdp`, `group_by_length`, …) now raise a bare
  `TypeError: unexpected keyword argument` with no rationale or alternative. Consider a thin `__init__` interceptor for a curated set
  of known HF-only kwargs that raises with the reason + the DP-correct alternative.
- **L2 — `privacy_target_epsilon=8.0` is a silent default** (`_config.py:396`); a user who never mentions privacy still gets
  DP-calibrated training to ε=8. No positivity guard on `privacy_target_epsilon`. Consider logging the resolved privacy target
  prominently and validating > 0.
- **L3 — No warning when evaluating on the private train set.** Eval consumes no budget (correct), but reported metrics carry **no**
  DP guarantee. A one-time warning when the eval dataset is the train dataset would close the footgun.
- **L4 — `batch_eval_metrics` silently unsupported** (no field). Reject explicitly or implement.
- **L5 — Doc drift:** `dptrainer.md:46–49` says calibration happens "at construction"; it actually runs in
  `train()`→`_setup_training`.
- **L6 — fp16-overflow steps consume a privacy mechanism.** The accountant is advanced (`_dp_trainer.py:1343`) before
  `training_step`, so an overflow step still composes one mechanism. DP-*safe* (conservative) but wastes ε and is
  untested/undocumented — pin it as an intentional, tested contract.
- **L7 — `validate_ddp_backend` second mismatch branch is dead** (`_distributed.py:160–169`).
- **L8 — `fp16=True` on CPU** builds an unusable scaler (CPU autocast is bf16-only) and fails later inside `torch.autocast`; a
  fail-fast in `_setup_precision` would be friendlier.
- **NITs:** `_collect_chunks` `return None if empty_ok else None` no-op (`_eval.py:361`); dead `prediction_loss_only` branch in the
  per-example path; stale `version=2` test comment vs `DP_STATE_BUNDLE_VERSION=3`; `inverse_sqrt` `timescale=0` fallthrough; stale
  precision test docstrings claiming `fp16` raises `NotImplementedError`; `eval_dtype` assumes the model has parameters.

---

## Verified-correct (so these can be trusted)

- **DP step ordering & sensitivity:** clip → DDP-sum → fp16 unscale-before-clip → finite-check → noise → pure optimizer; fp16 unscale
  runs per-example inside vmap before the clip norm, preserving the sensitivity invariant the accountant relies on.
- **Accounting:** one composition per iteration; total == `total_steps` (calibration target); on resume the prefix is genuinely
  installed into the live accountant (`_apply_runtime_state:4108`), so ε is **not** under-reported on the normal full-checkpoint
  path; missing-accountant policy is a hard error unless explicitly opted in.
- **DP-FTRL:** noise `(n_steps, min_sep, max_participations)` and the sampler's `bands`/`sampling_prob` are read off the single
  amplifier that drives accounting — verified all three amplifier types expose those attributes — so runtime mechanism and accountant
  cannot desync; `per_step` composition reconstructs the true K-step PLD.
- **DDP training step:** `sum_gradients_` preserves and cross-rank-asserts `ClippedPytree.max_norm`; shared key ⇒ identical noise
  added once to the cluster sum; `reduce_step_finite` (min-reduction) makes any rank's overflow trip all ranks *(once C2 is fixed)*;
  adaptive-clip state sync recomputes the threshold from globally-aggregated counts with a rank-independent key; sharded sample-rate
  uses the post-trim denominator consistently for sampler and accountant.
- **Config validation** shows real DP awareness: rejects `metric_for_best_model="train_*"` citing memorization leakage; rejects
  adaptive clipping under MF (sensitivity drift); blocks privacy-owned keys (`bands`, `sampling_prob`) in `sampling_kwargs`;
  positivity-guarded `clipping_norm`.
- **Eval does not consume privacy budget** (verified across all entry points); per-example losses are genuinely per-example.
- **RNG:** noise (`fold_in` over a torch generator) and the Poisson sampler (`np.random.default_rng`) use different PRNG families plus
  a per-step hash fold, so sharing the base seed does not correlate noise with subsampling.
- LR off-by-one logging is correct and end-to-end tested; DP bias-correction reads realized σ correctly; scheduler curve math matches
  HF numerically; reporting hierarchical-key rewriting (`privacy/…`, `eval/…`) works; callback registration order and
  `optimizer=None` / `on_substep_end`-not-fired are correct.

---

## Recommended priority order

1. **Add the DP-correctness test layer** (META) — it is what would have caught C2 and most of the rest.
2. **C2 (fp16 finite-check no-op)** and **C1 (save_only_model resume noise reuse)** — both silent, both confirmed.
3. **M2 (atomic checkpoint write)** — compounds C1.
4. **H1 (DDP eval deadlock), H2 (cosine_with_restarts crash), H5 (DDP sampler key)** — crashes/hangs + sampling correctness.
5. **M1 (epoch-resume ε overshoot), M3/H3 (precision & eval-shape surprises)**.
6. UX/doc cleanups (L1, M4, M6, M7, M8, the false DDP-sampler comment).

---

## Resolution status

All findings were addressed on this branch (`claude/opaque-trainer-ultra-review-g2Tlm`), each with
regression coverage where testable on CPU. DDP-only items (H1, H5, and the DDP DP-guarantee assertions)
ship with the fix plus a `cuda`-marked / unit-level guard and **still require a multi-GPU validation run**
before merge.

| Finding | Status | How |
| --- | --- | --- |
| **META** test gap | Fixed | New `tests/validation/test_dp_guarantees.py`: calibrated-ε vs reference, exact composition count, clipping-bounds, σ=nm·C, eval-spends-no-budget, σ>0 reproducibility, truncated-Poisson, rank-fold decorrelation, resume-on-budget, dict-eval, partial-checkpoint |
| **C1** save_only_model resume | Fixed | Hard-error in `_train_once` unless `privacy_resume_without_accountant`; test |
| **C2** fp16 finite-check | Fixed | `all_finite(grads.pytree)`; CPU building-block guard + `cuda` e2e test (confirmed empirically) |
| **H1** DDP eval deadlock | Fixed (loud-fail) | Reject empty shards + `eval_do_concat_batches=False` under DDP; **needs multi-GPU validation** for full uneven-shard gather |
| **H2** cosine_with_restarts crash | Fixed | Fractional cycle length in `WithRestarts` + inner-cosine float span; exact-HF non-divisible parity tests |
| **H3** eval dual-path shape | Fixed | Docstring corrected, `ignore_keys` honoured, one-time logits-only warning |
| **H4** sampler O(n) replay | Fixed (docs) | Corrected the false O(1) claim; replay is correct, an O(1) numpy-internals jump was rejected as fragile |
| **H5** DDP sampler key | Fixed (fresh-run) | `fold_in(key, rank)`; decorrelation unit test; resume-per-rank-snapshot noted, **needs multi-GPU validation** |
| **M1** epoch-resume ε overshoot | Fixed | Unconditional `global_step >= ctx.total_steps` ceiling; test |
| **M2** atomic checkpoint write | Fixed | Stage into `checkpoint-N.tmp` + `os.replace`; crash-sim test |
| **M3** full-eval no-op in train | Fixed (warn) | One-time warning that full-cast eval is post-training only |
| **M4** dict eval | Fixed | Recursive per-split prefixed eval; test |
| **M5** per-example eval keys | Fixed | Validate eval batch carries train keys; clear error |
| **M6** drift false-positive | Fixed | Skip noise_multiplier drift in calibrated mode |
| **M7** dead sampler modes | Fixed (docs) | Corrected "five modes" comment; build_sampler-only modes documented |
| **M8** optim docstring | Fixed | Added `adamw-bc` alias; corrected docstring (dropped `adamax`, fixed adafactor/lion) |
| **M9** empty-round cadence | Fixed | Run log/save/eval gate on empty rounds; loss/token accumulation guarded |
| **L1** HF-migration UX | Fixed (docs) | Migration table in `training-arguments.md` |
| **L2** budget defaults | Fixed | Positivity/range guards on ε / σ / δ |
| **L3** eval-on-train warning | Fixed | One-time warning |
| **L4** batch_eval_metrics | Documented | Dataclass already raises `TypeError`; migration table points to `eval_accumulation_steps` |
| **L5** calibration-timing doc | Fixed | `dptrainer.md` corrected |
| **L7** dead DDP-backend branch | Fixed | Removed |
| **L8** fp16+CPU | Won't fix | Finding was incorrect — `fp16_full_eval` is a CPU-valid cast; based on an outdated CPU-autocast assumption |
| Nits | Fixed | `_collect_chunks` no-op, dead `prediction_loss_only` branch, `inverse_sqrt timescale=0`, `eval_dtype` parameterless guard, stale precision/version docstrings |

Full affected-suite sweep (`opaque-transformers` + engine scheduling + dpsgd sampling, CPU, not-slow):
**668 passed, 19 skipped**; repo contract tests: **11 passed**; lint/format clean.
