# Phased DPTrainingArguments Implementation Plan

> **Historical reference (pre-cleanup).** This document describes the
> design as of the HF-parity build. It's preserved as the rationale
> trail for individual decisions, but it is **out of date** with the
> shipping field surface:
>
> - The class is renamed: `DPTrainingArguments` → `TrainingArguments`.
> - `max_grad_norm` → `clipping_norm`; `auto_find_batch_size` →
>   `auto_find_microbatch_size`; `use_liger_kernel` →
>   `use_performance_kernels`; `liger_kernel_config` →
>   `performance_kernels_config`.
> - `gradient_accumulation_steps`, `do_train`, `do_predict`,
>   `include_inputs_for_metrics`, all `hub_*` fields, `hp_name`, and
>   the entire HPO + Hub surfaces are **removed**.
> - `per_device_train_batch_size` is now the per-rank logical Poisson
>   batch (no internal grad-accumulation knob); cluster-wide logical
>   batch is `per_device_train_batch_size * world_size`
>   (the HF `train_batch_size` property).
> - `DPTrainerState` is now a standalone dataclass with an explicit
>   `version: int` field; it does **not** inherit from HF's
>   `TrainerState`.
> - The privacy accountant lives at the trainer level
>   (`self._accountant`); `save_model()` always writes
>   `accountant.json` so privacy provenance travels with the model.
>
> The user-facing migration story lives in `docs/user-guide/huggingface.md`.

## Current state

`DPTrainingArguments` declares ~80 HF-compatible fields + ~20 dp_ fields.
Most trainer-owned HF fields are now either wired into `DPTrainer` or rejected
explicitly when their semantics conflict with the functional DP-SGD path. This
plan tracks both the implemented support and the deliberate non-goals.

**Fields currently wired** (working today):
the scheduler, checkpoint/resume, evaluation, device/precision, logging,
reporting, data-pipeline, determinism, label, Hub, HPO, and public Trainer
contract surfaces described in the implemented phases below, plus all `dp_*`
fields.

---

## Phase 1: LR Scheduler — implemented

**Parameters**: `lr_scheduler_type`, `lr_scheduler_kwargs`, `warmup_ratio`, `warmup_steps`.

**Implementation**: A callable `step → learning_rate` is passed directly to torchopt's optimizer factories — torchopt's `scale_by_neg_lr` accepts `Callable[[int], float]` and embeds the step counter inside the optimizer state via `scale_by_schedule`. No `torch.optim.lr_scheduler` machinery and no LR scaling on `updates`.

**Code**:
- New `opaque.scheduling` module: the curves torchopt doesn't ship — `cosine_schedule`, `inverse_sqrt_schedule`, `constant_schedule`, `linear_schedule`, `polynomial_schedule`, `one_minus_sqrt_schedule` — plus `with_warmup(decay, num_warmup_steps, base_lr)` and `with_restarts` composition primitives that auto-shift the step counter passed to `decay`.
- HF shim `opaque.api.transformers.trainer._scheduler` (`build_lr_schedule(args, num_training_steps)`, `get_warmup_steps(...)`, `parse_optim_args(...)`): dispatches **10** of HF's `SchedulerType` strings — `linear`, `cosine`, `constant`, `constant_with_warmup`, `inverse_sqrt`, `polynomial`, `cosine_with_restarts`, `cosine_with_min_lr`, `cosine_warmup_with_min_lr`, `warmup_stable_decay` — to compositions of the primitives above; unknown kwargs raise `ValueError`.  `reduce_lr_on_plateau` is intentionally not supported: it's metric-driven and data-dependent, which doesn't fit the recipe-based static-schedule model the rest of `opaque.scheduling` is built on.
- `DPTrainer.create_scheduler(num_training_steps)` (subclass override hook); called from `_setup_training` and the resulting callable is passed into `create_optimizer` for every optimizer branch (`adam`, `sgd`, `adamw-bc`, `adamw`).
- `learning_rate` is logged in `state.log_history` at each `logging_steps` boundary, computed as `lr_schedule(global_step - 1)` — the value just applied to the optimizer update at iteration `global_step` (HF parity: torchopt's `scale_by_schedule` increments the count *after* the update, so the LR consumed by step N is `schedule(N - 1)`).

**Skip**: none for the dispatch surface (full HF coverage).  `optim_args` is implemented via `parse_optim_args`.

**Tests**:
- `packages/opaque-core/tests/scheduling/test_scheduling.py` — pointwise correctness of every primitive.
- `packages/opaque-transformers/tests/opaque_transformers/test_scheduler_dispatch.py` — HF parity for each `lr_scheduler_type` (≈200 sample points per type at `tol=1e-9`); plus dispatch error cases and warmup edge cases (`warmup_steps=0`, `warmup_steps==num_training_steps`, `warmup_steps>num_training_steps`, `polynomial(power=1.0)` ≡ `linear`).
- `packages/opaque-transformers/tests/validation/test_dp_trainer.py::TestDPTrainerLRScheduling` — end-to-end through `DPTrainer.train()`.

**Docs**: `docs/api/schedules.md`, `docs/user-guide/lr-scheduling.md`, plus an LR-scheduling section in `docs/user-guide/huggingface.md`.

---

## Phase 2: Saving & Checkpointing — implemented

All three sub-phases (2a basic save, 2b best-model tracking, 2c resume) are
wired in `DPTrainer`. See
[`docs/user-guide/checkpointing.md`](../../../../docs/user-guide/checkpointing.md)
for user-facing documentation.

Highlights of what landed:

- **Checkpoint dir layout** — `model.safetensors`, `optimizer.pt`,
  `dp_runtime_state.pt`, `accountant.json`, `trainer_state.json`,
  `training_args.bin`, `rng_state.pth` under `checkpoint-<step>/`.
- **`state_dict` / `from_state_dict` on Opaque types** — `FixedClipState`,
  `AdaptiveClipState`, `GaussianNoiseState`, `PerGroup`, plus PyTorch-style
  `state_dict` / `load_state_dict` on `PoissonSampler` /
  `TruncatedPoissonSampler`. Sampler resume is O(1) via per-iteration
  `fold_in(key, iter_count)` (no batch replay).
- **Strategies** — `save_strategy ∈ {"no", "steps", "epoch", "best"}` (HF
  parity — all four are members of `transformers.trainer_utils.SaveStrategy`,
  including `BEST`, which we previously framed as an Opaque extension);
  fractional `save_steps`, end-of-training final save, rotation that protects
  most-recent + best.
- **Best-model tracking** — HF parity: `metric_for_best_model` is auto-defaulted
  to `"loss"` under `load_best_model_at_end`; `greater_is_better` defaults
  from the metric-name suffix.
- **Resume** — `train(resume_from_checkpoint=path|True|None)`. Restores
  model, optimizer, clip / noise / sampler / accountant states, RNG snapshots,
  and `DPTrainerState`.  `ignore_data_skip` toggles sampler-state restore
  (DP-valid either way); `restore_callback_states_from_checkpoint` honors
  HF semantics.  **Data-order parity is intentionally not byte-for-byte**:
  HF's resume rebuilds the dataloader and skips ``global_step`` batches
  to replay the exact sequence; ours jumps the Poisson sampler to the
  saved ``iter_count`` via ``fold_in(key, iter_count)``, producing a
  same-distribution-but-different-sequence subsample.  Privacy budget
  is unchanged (one Poisson-amplified mechanism per iteration either
  way) — this is a deliberate variance-reduction trick that avoids
  O(steps) replay cost.  See ``DPTrainer.train`` docstring.  Sharded
  safetensors (`model.safetensors.index.json`) and pickle index loads
  are supported via HF's `load_sharded_checkpoint`.  Optimizer state
  loads on CPU first and migrates lazily on the next step (HF
  memory-profile parity).
- **Callback round-trip** — HF parity: `stateful_callbacks` field on
  `trainer_state.json` carries each callback's payload using HF's
  `ExportableState` schema (`{"args": {...}, "attributes": {...}}`) when
  the callback subclasses it (e.g. `EarlyStoppingCallback`); legacy
  `state_dict`/`load_state_dict` callbacks fall through to the older
  protocol.  `trainer_state.json` is *always* persisted, including under
  `save_only_model=True`.
- **Privacy on resume** — heterogeneous composition is preserved:
  the saved `Accountant` is loaded as the prefix and calibration of
  the remaining steps targets `dp_target_epsilon` against that prefix.
  Changing `dp_noise_multiplier` / `dp_target_epsilon` between
  checkpoint and resume warns but is allowed — the accountant composes
  whatever process the user asks for, the warning guards against
  silent drift.  `accountant.json` is written **unconditionally**
  (independent of `save_only_model`); a missing file means the user
  supplied a non-DP checkpoint or hand-edited it out, in which case
  the accountant is loaded with `acc.nonprivate()` prepended so future
  ε is ∞ — signalling unknown prior cost rather than silently
  restarting accounting.

**Files**: `trainer/__init__.py`, new `trainer/_checkpoint.py`,
`trainer/_state.py` (best fields + `to_json` / `from_json`); state-dict
methods on `opaque-core` clipping types and `opaque-dpsgd`
clipping/noise/sampling types.

---

## Phase 3: Evaluation improvements — implemented

All three sub-phases (3a core eval controls, 3b eval memory management, 3c
metrics enrichment) are wired in `DPTrainer`. The eval loop now mirrors HF's
`evaluation_loop()` + `prediction_step()` decomposition and accepts
HF-typed `compute_metrics` callbacks unchanged.

Highlights of what landed:

- **Refactor** — `DPTrainer.evaluate()` is now a thin wrapper around
  `evaluation_loop()` + `prediction_step()`. The functional path
  (`fmodel + trainable_params`, mid-training) and the `nn.Module` path
  (post-training) both produce identically-shaped `(loss, logits, labels)`
  so user-supplied `compute_metrics` callbacks run unchanged on either.
- **3a — Core eval controls** — `eval_on_start` fires once before the epoch
  loop on a fresh run (skipped on resume); `eval_delay` is honored in
  steps for `eval_strategy="steps"` and in epochs for
  `eval_strategy="epoch"`; `prediction_loss_only` short-circuits logits
  collection and never invokes `compute_metrics`. Adjacent fix:
  `eval_strategy="epoch"` no longer fires on `eval_steps` cadence — eval
  fires once per epoch boundary at the end of the inner step loop.
- **3b — Eval memory management** — `eval_accumulation_steps=N` flushes
  the prediction accumulator to CPU every N batches; `batch_eval_metrics`
  bypasses the accumulator entirely and calls `compute_metrics` once per
  batch (`compute_result=False`) plus once at the end (`compute_result=True`)
  as a stateful reducer; `eval_do_concat_batches=False` delivers
  predictions and labels as a list of per-batch tensors.
- **3c — Metrics enrichment** — `include_for_metrics ⊇ {"inputs", "loss"}`
  populates `EvalPrediction.inputs` and `EvalPrediction.losses`. `losses`
  is a 1-D tensor of per-**batch** mean loss (causal-LM models reduce
  internally; per-example loss is a future enhancement). Variable-length
  per-batch tensors are joined via right-padding (`-100` everywhere — HF
  convention) before concat. After concat, leading-dim is truncated to
  the dataset's true `num_samples` via
  `transformers.trainer_pt_utils.nested_truncate` (HF parity).
  `EvalPrediction.inputs` follows HF's path-specific shape exactly: the
  non-batched accumulator path delivers a **bare tensor** (the
  `inputs[main_input_name]` column collected via `inputs_decode`); the
  `batch_eval_metrics=True` path delivers the **full batch dict** that
  HF's per-batch reducer site receives at `trainer.py:4725`.
- **`preprocess_logits_for_metrics`** — wired identically to HF, called
  inside `prediction_step` after the forward, before tensors enter the
  accumulator.
- **HF-type re-exports** — `EvalPrediction` and `EvalLoopOutput` are
  re-exported from `transformers.trainer_utils` via
  `opaque.api.transformers.trainer._eval`, with an import-time smoke check
  asserting the four field names we depend on.
- **Validation** — `DPTrainer.__init__` calls `validate_eval_args(args,
  compute_metrics)`: raises `ValueError` if `batch_eval_metrics=True`
  without `compute_metrics`, or if `include_for_metrics` contains a key
  outside `{"inputs", "loss"}`.
- **Numpy contract** — predictions / labels / inputs / losses delivered
  to `compute_metrics` are numpy arrays (HF parity, via
  `transformers.trainer_pt_utils.nested_numpify`).  The metrics dict is
  scalarized via `denumpify_detensorize` before logging so JSON
  encoding never sees `np.float32` or 0-d Tensors.
- **`prediction_step` widening** — accepts arbitrary batch dict keys
  (`pixel_values`, `decoder_input_ids`, …) and forwards them to the
  model via `**inputs`.  Label tensors named in `self._label_names`
  (defaulted via `transformers.utils.find_labels(model.__class__)`,
  matching HF's per-architecture defaults — `["labels"]` for causal-LM,
  `["start_positions", "end_positions"]` for QA, etc.) are popped and
  passed as `labels=...`.  Signature mirrors HF exactly:
  `prediction_step(self, model, inputs, prediction_loss_only,
  ignore_keys=None)` — the `model` arg is forwarded into
  `compute_loss(model, ...)` so subclass overrides see the right
  surface.
- **Multi-task evaluation** — `evaluate(eval_dataset={"a": ds_a, "b":
  ds_b})` namespaces metric keys with `f"{prefix}_{name}_*"` and
  returns a merged dict; each sub-evaluation logs and fires callbacks
  independently (HF parity).

**Deprecated alias handling** — `include_inputs_for_metrics` is accepted
on `DPTrainingArguments` and emits a `FutureWarning` mirroring HF's
deprecation behavior; the value is folded into `include_for_metrics`
("inputs") at construction time so user code targeting the old kwarg
keeps working.

**Out of scope** (deferred to other phases): `eval_use_gather_object`,
`average_tokens_across_devices` (Phase 10 DDP); `bf16_full_eval`,
`fp16_full_eval` (Phase 4 precision); `jit_mode_eval` (Phase 11 compile);
`do_predict` (Phase 7c); a full HF-compatible `predict()` API.

**Limitations**

- The single-process eval-loss accumulator does not gather across
  ranks — DDP eval gather is deferred to Phase 10.
- `preprocess_logits_for_metrics` runs inside `prediction_step`
  before the accumulator; HF runs it after `pad_across_processes` /
  before `gather_function`.  Single-process eval is unaffected; the
  ordering will be aligned when distributed eval lands in Phase 10.

**Files**: `trainer/__init__.py` (eval refactor + `_inner_training_loop`
`eval_on_start` trigger + `_should_eval` gate + per-epoch eval call +
`find_labels`-driven label_names default + peek-ahead eval loop so
`batch_eval_metrics` reducers see `compute_result=True` inline on the
last data batch), new `trainer/_eval.py` (`_PredictionAccumulator`,
`should_run_eval_at_step`, `with_metric_prefix`, `validate_eval_args`,
`resolve_eval_num_samples`, HF-type re-exports), `trainer/_config.py`
(handles deprecated `include_inputs_for_metrics` with `FutureWarning`).

---

## Phase 4: Device & Precision

Current DPTrainer inherits device from the model and doesn't manage precision.

**Parameters**: `bf16`, `fp16`, `tf32`, `no_cuda`, `use_cpu`, `use_mps_device`, `bf16_full_eval`, `fp16_full_eval`

**What to implement**:
- Device resolution in `__init__`: respect `no_cuda`/`use_cpu`/`use_mps_device` flags; currently we just do `next(model.parameters()).device`
- `tf32`: call `torch.backends.cuda.matmul.allow_tf32 = True` / `torch.backends.cudnn.allow_tf32 = True`
- `bf16`/`fp16`: set model dtype. For DP, this affects the functional forward pass precision under vmap. Model is loaded in the requested dtype; no autocast wrapper needed (Opaque doesn't use Accelerate)
- `bf16_full_eval`/`fp16_full_eval`: cast model to requested dtype during evaluation

**Skip**: `fp16_opt_level`, `half_precision_backend` (Apex/Accelerate-specific)

**Files**: `trainer/__init__.py` (`__init__` device resolution, `_setup_training` dtype, `evaluate` eval dtype)

---

## Phase 5: Logging & Tracking — implemented

All three sub-phases are wired in `DPTrainer`.

### Phase 5a: Strategy and filtering

**Parameters**: `logging_strategy`, `logging_first_step`, `logging_nan_inf_filter`

- `logging_strategy="epoch"` ✅ — `DefaultFlowCallback.on_epoch_end` sets `should_log=True`;
  `_inner_training_loop` now flushes via `_maybe_log_save_evaluate` immediately after
  `on_epoch_end` (bug fix: the flag was previously silently dropped).
- `logging_first_step` ✅ — handled by `DefaultFlowCallback.on_step_end` setting `should_log`
  at `global_step==1`; no trainer code needed.
- `logging_nan_inf_filter` ✅ — applied at **two levels** (HF parity):
  1. Per-step loss accumulation in the training loop: a NaN/Inf step loss is replaced by
     the current running average so the smoothed curve stays finite.
  2. Any remaining NaN/Inf floats in the metrics dict logged at emit time are still
     silently dropped in `log()` via the existing filter (belt-and-suspenders).

### Phase 5b: Tracking integration

**Parameters**: `report_to`, `disable_tqdm`, `project`, `run_name`, `logging_dir`

- `report_to` ✅ — the `ValueError` block that blocked non-None values is removed;
  `build_callback_handler` now calls
  `get_reporting_integration_callbacks(args.report_to)` from
  `transformers.integrations` and registers the returned callbacks (W&B, TensorBoard,
  MLflow, Neptune, ClearML, DagsHub, DVCLive, SwanLab, …) before user callbacks.
  `WandbCallback` / `TensorBoardCallback` read `args.project`, `args.run_name`,
  `args.logging_dir` directly from `TrainingArguments` (which `DPTrainingArguments`
  inherits), so no trainer-side wiring is needed beyond callback registration.
- `disable_tqdm` ✅ — already wired via `PrinterCallback` / `ProgressCallback` selection.
- No new `_reporting.py` — `speed_metrics` is imported directly from
  `transformers.trainer_utils` (it is stable public API in HF 5.x).

### Phase 5c: Metrics enrichment

**Parameters**: `skip_memory_metrics`, `include_tokens_per_second`, `include_num_input_tokens_seen`

- `include_num_input_tokens_seen` ✅ — three modes (`"all"`, `"non_padding"`, `"no"`);
  HF normalises `True → "all"` / `False → "no"` in `TrainingArguments.__post_init__`.
  Per-step token count accumulates into `state.num_input_tokens_seen` in the training
  loop; `"non_padding"` uses `attention_mask.sum()` when available, then
  `processing_class.pad_token_id` comparison, else falls back to `numel()` with a warning.
  `num_input_tokens_seen` is injected into **every** `log()` call (not only training
  steps) via the updated `log()` method — HF parity.
- `include_tokens_per_second` ✅ — `log()` now accepts an optional `start_time` float;
  when `include_num_input_tokens_seen != "no"` and `start_time` is provided, live
  `train_tokens_per_second` is appended to each log row.  The training summary also
  passes `num_tokens=state.num_input_tokens_seen` to `speed_metrics` when the flag is set.
- `skip_memory_metrics` ✅ — `TrainerMemoryTracker(args.skip_memory_metrics)` is
  constructed in `__init__` (requires `psutil`; silently skips otherwise) and
  `.start()` is called at `__init__` time and again at the start of each training run.
  `.stop_and_update_metrics(metrics)` populates `init_mem_cpu_alloc_delta`,
  `train_mem_gpu_alloc_delta`, etc. into the training summary.

**Files**: `trainer/__init__.py` (log method updated, loop integration, training summary,
memory tracker wiring), `trainer/_callback.py` (reporting callbacks registration),
`trainer/_config.py` (report_to ValueError removed).

---

## Phase 6: Data Pipeline Contract

Goal: close the remaining HF `Trainer` dataloader contract gaps without
accidentally changing the privacy semantics of a training step.

**Hard invariant for this phase**: one optimizer step still corresponds to one
trainer-owned Poisson round. `dataloader_num_workers`,
`dataloader_prefetch_factor`, and `dataloader_persistent_workers` may
parallelize **example materialization / collation**, but must not silently turn
sampling itself into independent per-worker Poisson draws. That would change the
mechanism being executed and would require explicit `parallel_poisson`
accounting, new seeding rules, and probably a separate opt-in surface.

### Phase 6a: Column management

**Parameters**: `remove_unused_columns`

**Target HF contract**:

- If the dataset is a `datasets.Dataset`, remove columns not accepted by the
  model's forward signature before building the DataLoader.
- If the dataset is not a `datasets.Dataset`, wrap the collator so unused keys
  are filtered per example right before collation.
- Cache signature columns once, matching HF's `_set_signature_columns_if_needed`:
  model-forward parameters plus `label`, `label_ids`, and `self.label_names`.
- Log ignored columns once per split, and raise the same class of error HF does
  when pruning would remove every usable column.

**Implementation notes**:

- Keep `remove_unused_columns=False` as the escape hatch for tasks where the
  collator consumes raw features not present in `model.forward(...)`
  (`image`, `audio`, `video`, custom metadata, etc.) and synthesizes model
  inputs such as `pixel_values`.
- Reuse the same signature-column logic for train and eval so the contract stays
  aligned with the Phase 3 `prediction_step()` widening and the upcoming Phase 7
  `label_names` support.
- Avoid mutating the user dataset in-place outside the HF-style
  `datasets.Dataset.remove_columns(...)` path; for generic PyTorch datasets, the
  collator wrapper is the safe parity route.

**Validation**:

- `datasets.Dataset` path: extra columns are dropped eagerly and the info log is
  emitted once.
- Generic `torch.utils.data.Dataset` path: the wrapped collator receives only
  signature columns.
- `remove_unused_columns=False` preserves raw-feature tasks.
- Label columns survive pruning once `label_names` is set explicitly.

### Phase 6b: DataLoader tuning

**Parameters**: `dataloader_prefetch_factor`, `dataloader_persistent_workers`, `torch_empty_cache_steps`

**Target HF contract**:

- Thread `prefetch_factor` and `persistent_workers` through train/eval
  DataLoader construction when `dataloader_num_workers > 0`.
- Honor `torch_empty_cache_steps` in the training loop with HF-like cadence:
  clear device cache after the current step's tensors are no longer needed, not
  mid-step.

**Implementation notes**:

- `dataloader_prefetch_factor` is straightforward: it changes host-side queueing
  only and does not affect Poisson semantics because the batch sampler still runs
  in the main process.
- `dataloader_persistent_workers` needs a split plan:

  - **Eval loader**: safe first target. Eval datasets are stable and the loader
    can be cached/reused across repeated `evaluate()` calls.
  - **Train loader**: do not assume the current per-epoch rebuild can benefit.
    Today `get_train_dataloader()` constructs a fresh Poisson sampler keyed by
    epoch, so recreating the DataLoader each epoch would respawn workers and
    erase most of the benefit. Real train-side persistence likely requires a
    stable loader shell plus an epoch-aware sampler API (`set_epoch`-style) or a
    trainer-owned sampler object whose epoch/key can be advanced without
    rebuilding workers.
- Keep the current privacy-safe architecture by default: the sampler decides the
  batch indices in the parent process; workers only fetch dataset items for the
  indices they are given.

**Validation**:

- Loader kwargs are forwarded only when legal (`dataloader_num_workers > 0`).
- Eval with `persistent_workers=True` reuses workers across calls and does not
  regress correctness.
- Train with multiple workers preserves deterministic batches for fixed
  `args.seed` and sampler state.
- `torch_empty_cache_steps` reduces peak memory in a narrow regression test and
  does not perturb step accounting.

### Phase 6c: Parallel Poisson spike (decision gate, not baseline parity)

There is a real chance we will want the data path to become more parallel than
Phase 6b's safe host-side loading. If that work happens, it must be treated as a
mechanism change, not as a routine loader optimization.

**Decision rule**:

- If the sampler remains trainer-owned and only dataset fetch/collation happens
  in parallel workers, this stays in Phase 6.
- If workers or ranks start making independent sampling decisions, this stops
  being a plain DataLoader task and becomes `parallel_poisson` work. At that
  point it must move under the distributed/accounting track with explicit
  semantics, tests, and probably a new opt-in control surface rather than being
  hidden behind `dataloader_num_workers`.

**Why this needs care**:

- PyTorch DataLoader workers do not normally own the `batch_sampler`; with the
  current `batch_sampler=PoissonSampler(...)` design, enabling workers does **not**
  make Poisson sampling parallel.
- To make sampling itself parallel, we would need a different execution model
  (for example rank-local samplers over shards, or a worker-owned iterable data
  path), plus explicit seed partitioning and accountant updates.
- The repo already has the accounting primitive for the distributed case
  (`parallel_poisson`), so the most likely clean outcome is: Phase 6 keeps the
  current single-sampler semantics, and true parallel Poisson remains owned by
  Phase 10.

**Files**: `trainer/__init__.py` (signature-column plumbing, collator wrapper,
DataLoader kwargs, optional eval-loader caching, loop cache-empty hook),
potentially `trainer/_config.py` (additional arg validation if train-side
`persistent_workers` needs temporary gating)

---

## Phase 7: Determinism, Labels, Misc

### Phase 7a: Seed & determinism — implemented

**Parameters**: `data_seed`, `full_determinism`

**What was implemented**:
- `data_seed` seeds the Poisson / truncated-Poisson train sampler separately
  from global `seed`; when `data_seed is None`, sampler seeding falls back to
  `seed`.
- `full_determinism=True` calls HF's `enable_full_determinism(seed)` at trainer
  construction and again when an HPO trial reinitializes the model; otherwise
  `set_seed(seed)` is used for HF-compatible global RNG setup.

**Validation**:
- `tests/validation/test_dp_trainer_reproducibility.py` verifies changing
  `data_seed` changes the sampling trajectory while keeping model seed fixed.
- `tests/opaque_transformers/test_trainer_contract.py` locks in that
  `full_determinism=True` dispatches through HF's deterministic helper.

### Phase 7b: Label handling — implemented

**Parameters**: `label_names`, `label_smoothing_factor`

**What was implemented**:
- `label_names` defaults from HF's `find_labels(model.__class__)`, with PEFT
  wrappers unwrapped before inspection. Explicit `args.label_names` still wins.
- `label_smoothing_factor` is applied inside `_build_per_example_loss`, so the
  smoothed per-example loss is what `vmap`, clipping, noise, and the optimizer
  see. For causal-LM shaped logits/labels, the smoothing path applies the same
  one-token shift as the unsmoothed model loss. For classification/vector logits,
  it recomputes cross-entropy directly from logits and labels.
- If a fused CE path returns only `loss` and hides logits, DPTrainer currently
  warns once and falls back to the unsmoothed fused loss. This is a temporary
  safety rail; once the fused kernel exposes smoothing directly, that path should
  become a fused-kernel parity test instead of a lasting behavior contract.

**Validation**:
- `tests/opaque_transformers/test_trainer_contract.py` verifies label smoothing
  recomputes loss from logits for the single-example vector-logit case.

### Phase 7c: Misc features — implemented

**Parameters**: `auto_find_batch_size`, `neftune_noise_alpha`, `do_train`, `do_eval`, `do_predict`, `past_index`, `debug`

**What was implemented**:

- `auto_find_batch_size` is wired as an OOM retry loop that halves physical
  microbatch size while preserving the logical Poisson round.
- `debug="underflow_overflow"` creates HF's `DebugUnderflowOverflow` helper.
- `predict(...)` is available as part of the public Trainer contract.
- `neftune_noise_alpha` and `past_index` are explicitly rejected in
  `DPTrainingArguments.__post_init__` until audited for the functional
  per-example-gradient path.
- `do_eval` follows HF's auto-flip semantics when an eval strategy is set;
  `do_train` / `do_predict` remain script-level controls rather than trainer
  loop gates.

**Validation**:

- `tests/validation/test_dp_trainer.py` verifies explicit `train()` and
  `predict()` calls run even when `do_train=False` / `do_predict=False`, plus
  debug hook wiring, prediction output shape, `past_index` rejection, and
  `auto_find_batch_size` retry/floor behavior.
- `tests/validation/test_dp_trainer_eval_polish.py` verifies
  `auto_find_batch_size` restores model state and RNG before retrying.
- `tests/opaque_transformers/test_config.py` verifies `do_eval` auto-flip semantics
  and the full unsupported-parameter rejection table, including
  `neftune_noise_alpha` and `past_index`.

**Files**: `trainer/__init__.py`, `trainer/_config.py`,
`tests/validation/test_dp_trainer.py`,
`tests/validation/test_dp_trainer_eval_polish.py`, and
`tests/opaque_transformers/test_config.py`.

---

## Phase 8: Hub Integration — implemented

Push models to HuggingFace Hub after training.

**Parameters**: `push_to_hub`, `hub_model_id`, `hub_strategy`, `hub_token`, `hub_private_repo`, `hub_always_push`, `hub_revision`

**What was implemented**:

- `init_hf_repo(token=None)` — calls `huggingface_hub.create_repo`, sets
  `self.hub_model_id` and `self.push_in_progress = None`.  Called from
  `__init__` when `push_to_hub=True`.  Exposed as a public method on
  `DPTrainer` (mirrors `Trainer.init_hf_repo`).
- `_push_from_checkpoint(checkpoint_folder)` — async upload triggered at the
  end of `_save_checkpoint` when `push_to_hub=True`.  Respects all four
  `hub_strategy` values: `end` (skip), `every_save` (upload output_dir only),
  `checkpoint` (also upload as `last-checkpoint/`), `all_checkpoints` (also
  upload as `checkpoint-N/`).  `hub_always_push=False` skips a new push if one
  is already in flight.
- `_finish_current_push()` — blocks until `push_in_progress` completes;
  called at training end before the final `push_to_hub`.
- `push_to_hub(commit_message, blocking, token, revision, **kwargs)` — full
  push: `save_model(_internal_call=True)` → `create_model_card` →
  `upload_folder`.  Called at training end and (with `_internal_call=False`
  guard) when the user calls `save_model()` directly.  Exposed as a public
  method (mirrors `Trainer.push_to_hub`).
- `create_model_card(language, license, tags, …)` — delegates to
  `TrainingSummary.from_trainer(self, …)` for the base HF card (task tags,
  dataset metadata, eval metrics from `log_history`, hyperparameters, license
  inference), then appends an Opaque DP section bounded by
  `<!-- opaque-dp:begin/end -->` markers.  The section lists ε, δ, noise
  multiplier, clipping norm; values are read from `state.log_history` first,
  then from the live training context, then from `args`.  Tags
  `"differential-privacy"` and `"opaque"` are merged into the card metadata.
  The section is idempotent: repeated `push_to_hub` calls replace rather than
  append the block.  Exposed as a public method (mirrors
  `Trainer.create_model_card`).
- Hub progress bars are suppressed (`hf_hub_utils.disable_progress_bars`)
  during the training loop when `push_to_hub=True` (HF parity).
- Public properties `train_dataset` and `eval_dataset` added to `DPTrainer`
  so `TrainingSummary.from_trainer(self)` can read them (HF parity).

**Reuse**: `PushInProgress` (`transformers.utils.hub`), `upload_folder` /
`create_repo` / `ModelCard` (`huggingface_hub`), `TrainingSummary` /
`extract_hyperparameters_from_trainer` / `parse_log_history`
(`transformers.modelcard`), `HubStrategy` (`transformers.trainer_utils`).

**Files**: `trainer/_hub.py` (new), `trainer/__init__.py` (hub instance vars +
`init_hf_repo` call in `__init__`, `train_dataset` / `eval_dataset`
properties, `push_to_hub` / `create_model_card` / `init_hf_repo` public
methods, `_train_dispatch` split for progress-bar wrapper, `_push_from_checkpoint`
call in `_save_checkpoint`, `_finish_current_push` + `push_to_hub` at training
end, `push_to_hub` in `save_model`).

---

## Phase 9: Hyperparameter Search and Trainer Contract Gaps

HF hyperparameter search is not just a `TrainingArguments` field group. It is a
Trainer lifecycle surface built around `model_init`, `train(trial=...)`, output
directory isolation, callbacks/reporting, metric selection, and backend-specific
trial objects.

**Current status**: local Phase 9 support is wired for direct
`train(trial=...)` runs, `model_init(trial)` reinitialization, trial-scoped
output directories, public trainer helper methods,
`hyperparameter_search(..., backend="optuna")` via a DPTrainer-owned local
Optuna runner, and `hyperparameter_search(..., backend="wandb")` via a local
W&B sweep agent. Phase 12 layers `hyperparameter_search(..., backend="ray")`
on top via the Ray Tune adapter. Controller-provided dict trials remain supported as direct
`train(trial={...})` invocations.

**API surface**:

- `train(resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None)`:
  accept HF's `trial` argument and use it to populate trial-specific optimizer /
  scheduler / logging context. Dict trials are accepted for lightweight local /
  external-controller sweeps.
- `hyperparameter_search(hp_space=None, compute_objective=None, n_trials=20,
  direction="minimize", backend=None, hp_name=None, **kwargs)`: implement the
  public HF method and return a `BestRun`-compatible result. Implemented for
  local Optuna and W&B sweeps and (Phase 12) Ray Tune via
  `tune.with_parameters` actors.
- `model_init`: make each trial instantiate a fresh model through
  `model_init(trial)` or `model_init()` depending on the callable signature.
  A model passed directly to the constructor is valid for normal training but
  should be rejected for HPO unless we define a safe clone/reinit path.
- `optimizer_cls_and_kwargs` / `optimizers`: keep constructor signature parity,
  but external optimizer state must either be trial-local and DP-compatible or
  explicitly rejected. Reusing a user optimizer across trials is incorrect.

**Backend scope**:

- Implemented targets: Optuna-style local search (`optuna.Trial`), W&B sweeps
  through `wandb.agent`, the lightweight "dict trial" path used by external
  controllers, and (Phase 12) Ray Tune via the
  `tune.with_parameters(_objective, local_trainer=trainer)` actor pattern.
- Multi-rank Ray trials remain gated on Phase 10 (DDP); single-rank-per-trial
  sweeps (the common HPO case) work today.
- W&B sweeps are supported as reporting/config plumbing; the actual training
  loop still runs as independent DPTrainer invocations.

**Trial isolation contract**:

- Each trial writes under an isolated output directory, matching HF's
  `run-<trial_id>` / `checkpoint-<step>` style, so checkpoints, model cards,
  trainer state, RNG state, and accountant state cannot collide across trials.
- `run_name`, reporting callbacks, and progress bars must be trial-scoped.
  Callback state from one trial must not leak into the next.
- `load_best_model_at_end`, `metric_for_best_model`, and
  `greater_is_better` determine each trial's objective source unless the user
  supplies `compute_objective`.
- `save_total_limit` and checkpoint rotation apply per trial directory, not
  globally across the sweep.

**DP semantics**:

- Treat every trial as an independent DP training run over the same dataset.
  The accountant written for a trial reports that trial's ε/δ only.
- Do **not** silently compose ε across a sweep in `hyperparameter_search`.
  Whether an adaptive sweep over private data requires experiment-level privacy
  accounting is a higher-level policy decision and depends on what metrics are
  exposed to the search controller. If we add sweep-level accounting later, it
  should be explicit API, not hidden inside Trainer parity.
- Trial pruning / early stopping is safe mechanically, but the reported metric
  stream is still a private-data-dependent signal. Keep pruning support scoped
  to HF-compatible callbacks first; document the privacy interpretation before
  adding automatic DP composition for pruned trials.

**Other Trainer contract gaps to close in this phase or adjacent sweeps**:

- Public dataloader helpers: `get_test_dataloader`, `num_examples`,
  `num_tokens`, and `floating_point_ops`.
- Callback management helpers: `add_callback`, `remove_callback`,
  `pop_callback`, and correct `TrainerControl` propagation from every callback
  event.
- Process helpers: `is_world_process_zero`, `is_local_process_zero`,
  `_is_local_process_zero`, `_is_world_process_zero`; these become important
  once Phase 10 lands but can exist as single-process `True` helpers now.
- Public save/eval/predict parity: `save_state`, `save_metrics`,
  `log_metrics`, `predict(test_dataset, ignore_keys=None,
  metric_key_prefix="test")`, and label-less prediction behavior.
- Constructor aliases and properties: `processing_class` vs deprecated
  `tokenizer`, public `data_collator`, `compute_metrics`,
  `preprocess_logits_for_metrics`, `label_names`, `callback_handler`,
  `control`, `state`, `model`, `args`.

The public helpers listed above are now present in the single-process trainer
surface. Remaining adjacent work is mostly backend breadth (Ray execution) and
future Phase 10 distributed semantics.

**Phase 9.5 hardening**:

- Unsupported runtime and optimizer boundaries are locked in by tests against
  `DP_INCOMPATIBLE_PARAMETERS` and `_DP_OPTIMIZER_UNSUPPORTED`.
- Ray HPO rejection now explains the external-execution concern rather than
  presenting Ray as a missing local trial object.
- Reporting callbacks are tested as metrics/logging integrations that precede
  user callbacks and intentionally see `optimizer=None` / `lr_scheduler=None`
  because DPTrainer owns functional optimizer state.
- Direct dict trials are tested to rebuild callback handler state per trial.

**Validation**:

- Contract tests compare DPTrainer signatures against the supported subset of
  HF `Trainer.__init__`, `train`, `evaluate`, `predict`, and
  `hyperparameter_search`.
- HPO smoke tests with a tiny model/dataset verify trials produce isolated
  checkpoints/accountants and return `BestRun`-like results for the direct dict
  path, W&B sweeps, and Optuna when Optuna is installed.
- `model_init(trial)` receives the backend trial object and creates independent
  model parameters per trial.
- `compute_objective` overrides metric selection; default objective follows HF
  behavior for eval metrics.

**Files**: `trainer/__init__.py` (public method + train lifecycle), new
`trainer/_hpo.py` (backend dispatch, trial naming, objective resolution),
`trainer/_checkpoint.py` (trial output dirs), `trainer/_callback.py` (callback
state isolation), tests in `tests/opaque_transformers/test_trainer_contract.py` and a
new `test_hpo.py`.

---

## Phase 5d: Reporting Integration Compatibility — implemented

Treat HF reporting integrations as a compatibility surface, not just a side
effect of accepting `report_to`. Phase 5b registers the callbacks; this phase
validates which callback behaviors are safe and supported for a functional DP
trainer.

**Parameters / surfaces**: `report_to`, `logging_dir`, `run_name`, `project`,
`trackio_space_id`, reporting callbacks from `transformers.integrations`.

**What to implement**:
- Metrics/logging callbacks are supported when they consume only `args`,
  `state`, `control`, and metric payloads.
- Deterministic fake-callback coverage verifies lifecycle ordering, log payloads,
  save events, and HPO trial scoping without networked integrations.
- Optional TensorBoard smoke coverage runs when TensorBoard is installed.
- Documented and tested: callbacks inspecting `optimizer` or `lr_scheduler` see
  `None`, because DPTrainer owns functional optimizer and schedule state rather
  than `torch.optim.Optimizer` / `torch.optim.lr_scheduler` objects.
- HPO trial scoping is validated for callback state, trial names, trial params,
  and log payloads.
- Artifact uploads follow HF's callback-owned behavior: DPTrainer writes the
  checkpoint first, then fires `on_save`, so integration callbacks can upload
  checkpoint artifacts when their own flags are enabled. DPTrainer does not add
  a second artifact mediation layer in this phase.
- Artifact coverage verifies that `on_save` callbacks see a fully written
  checkpoint with model weights, `accountant.json`, `trainer_state.json`, and
  `training_args.bin`; `save_only_model=True` keeps privacy metadata while
  omitting resumability-only files.

**Files**: `trainer/_callback.py`, `trainer/_hpo.py`, tests in
`tests/opaque_transformers/test_reporting_integrations.py`,
`tests/opaque_transformers/test_trainer_contract.py`, and `test_hpo.py`, user-facing
docs in `docs/user-guide/huggingface.md`.

---

## Phase 10: Distributed Training (DDP) — implemented

Single-node multi-GPU DDP is wired end-to-end. Validated on 4× H100 via
`packages/opaque-transformers/tests/distributed/test_ddp_trainer.py` (5
scenarios, all passing): runtime foundation, per-rank shard partition,
global-mode independent streams, eval gather, RNG-per-rank checkpointing.
Uses Opaque's own distributed primitives (`opaque.distributed.sync`,
`sum_gradients_`, `local_shard`, `gather_pytree`, `reduce_scalar`) and the
`opaque.accounting.parallel_poisson` accounting mechanism. **Accelerate, FSDP,
DeepSpeed, SageMaker MP, and TPU/XLA stay rejected** — see
`DP_INCOMPATIBLE_PARAMETERS` ([_config.py:65–130](../../src/opaque/transformers/trainer/_config.py)).

User-facing doc: [docs/user-guide/distributed-trainer.md](../../../../docs/user-guide/distributed-trainer.md).

### Architectural ground rules (apply to every sub-phase)

1. **No Accelerate.** HF's [`Trainer._setup_devices`
   (training_args.py:1798–1860)](/workspaces/transformers/src/transformers/training_args.py)
   delegates to `Accelerate.PartialState`. DPTrainer's
   `_setup_devices` already bypasses that
   ([_config.py:859–880](../../src/opaque/transformers/trainer/_config.py)),
   and the `parallel_mode` / `process_index` / `local_process_index`
   properties HF exposes through `distributed_state` are not available in
   our path. Phase 10 must source rank/world from `torch.distributed`
   (initialised externally via `torchrun` / `mp.spawn`) or from the
   `LOCAL_RANK` / `RANK` / `WORLD_SIZE` env vars.
2. **NCCL only, single node first.** Mirror the constraint already documented
   in [docs/user-guide/distributed.md:299–305](../../../../docs/user-guide/distributed.md).
3. **Process-group ownership.** DPTrainer assumes the launcher
   (`torchrun` or test-side `mp.spawn`) initialised the group; the trainer
   never calls `init_process_group` itself, mirroring HF
   ([trainer.py via PartialState]) and the existing
   [examples/train_causal_lm.py:51–52](../../../../examples/train_causal_lm.py).
4. **Privacy invariant per step still holds.** One optimizer step = one
   logical Poisson round across the cluster. Sharded mode preserves the
   global rate by construction; parallel-Poisson mode is accounted for by
   `acc.parallel_poisson(...)`. Phase 6 host-side parallelism (worker
   prefetch, collation) is unchanged — workers still do not own the sampler.
5. **Functional optimizer stays synchronised by construction.** After
   `sum_gradients_`, every rank holds the same clipped-grad sum; identical
   noise via shared key + identical pure-function torchopt update keeps
   parameter trees bit-identical (see
   [docs/user-guide/distributed.md:237–250](../../../../docs/user-guide/distributed.md)).
6. **Environment gate.** All implementation and CI validation for Phase 10
   require a multi-GPU machine. The current dev box has 4× GPUs, which is
   the canonical Phase 10 test target.

### Phase 10 prerequisite — distributed coverage of new core primitives

This prerequisite is now closed. Optimizer sync registrations ship under
`opaque/optimizers/distributed.py`, and
`opaque.distributed._state._ensure_builtin_sync_types_loaded` imports them on
first dispatch miss together with clipping/profiling registrations.

The clipping precedent is
[opaque/clipping/_distributed.py](../../../../packages/opaque-core/src/opaque/clipping/_distributed.py)
which self-registers `FixedClipState`, `ClippedFunAux`, `ClippedGradAux` via
`register_sync_type` at import time, and the noise precedent is
[opaque/dpftrl/noise/_engine.py:284](../../../../packages/opaque-dpftrl/src/opaque/dpftrl/noise/_engine.py)
(MFNoiseState) and
[opaque/dpsgd/noise/gaussian.py](../../../../packages/opaque-dpsgd/src/opaque/dpsgd/noise/gaussian.py)
(GaussianNoiseState).

**Why we still need handlers when state stays in sync by construction.**
Functional optimizers and schedules are *pure*: given identical input gradient
+ identical state, every rank produces identical state. After
`sum_gradients_(grads)` everyone shares the gradient, and `init_fn(params)`
gives every rank the same initial state (deterministic from the same params).
So in steady state, optimizer state cannot drift. The reason to register sync
handlers anyway:

1. **Defensive drift detection.** AllReduce reduction order is not bit-stable
   on NCCL across hardware generations; a divergence in `opt_state` between
   two ranks usually means an upstream bug (different fp16 overflow handling,
   different vmap implementation, asymmetric NaN). Catching it via a periodic
   `assert_pytree_equal(opt_state)` is much cheaper than diagnosing diverged
   parameters 10K steps later. This is exactly what `GaussianNoiseState` does
   today (asserts `seed` and `step` match).
2. **Dispatcher completeness.** `sync(*everything)` is the trainer's
   one-stop-call between clipping and noise; if any state type passed in
   raises, the trainer has to special-case the call. Better to register
   identity handlers and let `sync` accept the entire state pytree.
3. **Future-proofing for non-pure transforms.** If someone adds a stochastic
   optimizer (e.g. SGD with random restarts) the registry forces them to
   think about cross-rank semantics rather than silently break.

**Work items (own commits, blockers for Phase 10c, non-blocking for 10a/b)**:

- New `packages/opaque-core/src/opaque/optimizers/distributed.py`. Mirror the
  clipping pattern: register an `assert_pytree_equal`-style handler for each
  optimizer-state dataclass (`AdamState`, `AdamWState`, `LionState`,
  `AdEMAMixState`, `AdafactorState`, `RMSPropState`, `AdagradState`,
  `ScheduleFreeState`). The handler validates tensor leaves are equal across
  ranks (cheap fingerprint via `assert_pytree_equal`) and asserts the
  scalar `step` matches via `assert_scalar_equal`. Skip non-tensor /
  non-scalar fields (`treespec`, `beta`, etc.).
- For composed optimizers (`make_optimizer_chain`, `_chain.py`), state is a
  tuple of inner states; register a tuple-walking handler that recurses on
  each leaf state.
- New `packages/opaque-core/src/opaque/scheduling/distributed.py`. LR
  schedules are stateless callables (`step → lr`); they don't need a
  registered handler. The schedule's *step counter* lives inside the
  optimizer state's `scale_by_schedule` slot, which is covered by the
  optimizer registration above. Document this in the new file (a one-line
  module docstring is enough — no code).
- Add an export in `opaque-core` so importing `opaque.optimizers` triggers
  registration as a side-effect (current pattern: clipping registers when
  `opaque.clipping.distributed` is imported, which is itself triggered by
  `_ensure_builtin_sync_types_loaded()` at first dispatch miss in
  [state.py:230–242](../../../../packages/opaque-core/src/opaque/distributed/state.py)).
  Extend `_ensure_builtin_sync_types_loaded()` to also import
  `opaque.optimizers.distributed`.
- Tests in `packages/opaque-core/tests/distributed/test_optimizer_sync.py`
  using `mp.spawn` (mirror `test_core_utilities.py`). Each optimizer's state
  must round-trip `sync(state)` cleanly under both equal and divergent
  inputs (intentionally drift one rank's state and confirm the handler
  raises).

This block is in the **Phase 10 prerequisite** spot rather than 10c because
DPTrainer doesn't strictly need it at the API level — the trainer calls
`sync(clip_state)` and AllReduce's the gradient itself, never `sync(opt_state)`
in the hot path. But the *trainer-level audit story* breaks without it: a user
adding `sync(*all_state)` for debugging hits a `TypeError` on the new
optimizers. Land it before declaring Phase 10c done.

### Rank-data policy — the central decision

> **Update**: the `ddp_shard='global'` / parallel-Poisson opt-in below was
> removed from the trainer (see `refactor!(transformers): drop ddp_shard='global'`).
> DP-SGD under DDP is now sharded-only: `local_shard` of the dataset, shared
> per-epoch key on every rank, regular `acc.poisson(...)` at the global rate.
> The rest of this section is kept as historical design rationale.

DPTrainer needs to commit to a default rank-data policy and offer the other
as an opt-in. Both are valid DP-SGD; they differ in sample-rate accounting.

| Mode (proposed `dp_shard` value) | Dataset visibility per rank | Sampler key | Accounting | Default? |
|---|---|---|---|---|
| `"per_rank"` (sharded) | `local_shard(D, rank, world)` | shared per-epoch key (no rank fold) | regular `acc.poisson(...)` over the **global** rate `q = expected_batch_size / |D|` (each rank's local rate equals the global rate because shard size ≈ `|D|/world`) | **yes** |
| `"global"` (parallel-Poisson) | full `D` on every rank | `fold_in(epoch_key, rank)` | `acc.parallel_poisson(acc.gaussian(nm), q_local, num_workers=world)` where `q_local = expected_batch_size / (world * |D|)` | opt-in |
| `"none"` (validation only, world=1) | full `D` | shared key | regular Poisson | implicit when world_size==1 |

The sharded default mirrors HF's `DistributedSampler` pattern (see
[trainer.py:1037–1064](/workspaces/transformers/src/transformers/trainer.py)
`_get_eval_sampler`) and matches the worked example in
[examples/train_causal_lm.py:121–139](../../../../examples/train_causal_lm.py).
Parallel-Poisson is the right choice when (a) the user can't shard cleanly
(streaming / iterable datasets), or (b) they want sampling diversity to
exceed a single-shard's index space; the cost is the extra ε term encoded
in `parallel_poisson_gaussian_pld`
([parallel_poisson.py:30–82](../../../../packages/opaque-accounting/src/opaque/accounting/amplification/parallel_poisson.py)).

The `dp_shard` arg lives next to the existing `dp_*` fields in
`DPTrainingArguments`; default `"per_rank"`. Reject any other value when
`world_size == 1` to keep single-process behaviour unchanged.

### Phase 10a: Runtime foundation (single-rank semantics correct everywhere)

**Goal.** Make `DPTrainer` rank-aware end-to-end, gate every I/O site by rank,
and lock in the rejection of unsupported wrappers — but do **not** yet change
the DP step itself. After 10a, a 1-rank "distributed" run (`torchrun --nproc-per-node=1`)
must produce byte-identical artefacts to today's single-process run.

**HF parameters surfaced**: `local_rank` (already accepted),
`ddp_backend` (default `"nccl"`), `ddp_timeout` (default 1800),
`log_on_each_node`, `save_on_each_node`, `log_level_replica`.

**New file**: `trainer/_distributed.py` — owns
- `class DDPState` — frozen dataclass `(is_distributed, rank, local_rank, world_size, backend, device)` populated once in `_setup_training`.
- `resolve_ddp_state(args) -> DDPState` — env-var aware (`LOCAL_RANK` / `RANK` / `WORLD_SIZE`); when `torch.distributed.is_initialized()`, reads via `opaque.distributed.{get_rank, get_world_size}`.
- `should_log(args, ddp) -> bool` and `should_save(args, ddp) -> bool` — mirror HF's `should_log` / `should_save` properties at [training_args.py:1992–2015](/workspaces/transformers/src/transformers/training_args.py).
- `barrier()` — thin wrapper over `opaque.distributed.barrier` that no-ops when not distributed.

**Touchpoints in `__init__.py`** (anchors are current line numbers; will shift):
- [`__init__.py:569–579`](../../src/opaque/transformers/trainer/__init__.py): replace the four hard-coded `True` returns with reads off `self._ddp` (a `DDPState` populated in `__init__`).
- [`__init__.py:3091–3094`](../../src/opaque/transformers/trainer/__init__.py): pass `rank=self._ddp.rank` to `seed_worker` instead of `0`.
- `_save_checkpoint` (around line 3855), `_save_model_artifacts` (around 3937), `save_state` (around 2521), `save_metrics` (around 2501): wrap the body in `if not should_save(self.args, self._ddp): return`. Already hold a `barrier()` after the rank-0 write so other ranks don't race with rotation / hub upload.
- `log` (line 3236) and `_maybe_log_save_evaluate` (line 3298): only the **emit** path (TensorBoard / W&B / file) gates on `should_log`; the metric *computation* still runs everywhere because subsequent ranks need the same `state.log_history` for callback-state-roundtrip checkpointing.
- `_hub.py:88, 121`: gate `init_hf_repo`, `push_to_hub`, `_push_from_checkpoint`, `_finish_current_push`, `create_model_card` so that only world-rank-0 talks to Hub. Other ranks still call the public method (HF parity); the rank-0 guard is internal.

**Touchpoints in `_config.py`**:
- [_config.py:837–853](../../src/opaque/transformers/trainer/_config.py): expand the single-process distributed-defaults block to read `WORLD_SIZE` / `RANK` / `LOCAL_RANK`, set `self._n_gpu = 1` per rank when distributed, and validate `ddp_backend` (only `"nccl"` accepted; raise on `"gloo"` / `"mpi"` / `"xccl"` etc. with a "DPTrainer only validates NCCL" message).
- Keep `fsdp`, `fsdp_config`, `fsdp_min_num_params`, `fsdp_transformer_layer_cls_to_wrap`, `accelerator_config`, `parallelism_config`, `deepspeed`, `tpu_num_cores`, `mp_parameters` rejected exactly as today; add new contract test that exercises the rejection table when `WORLD_SIZE > 1` (the env shouldn't change the verdict).
- Reject `ddp_static_graph` for now (not validated against vmap'd functional path); reject `gradient_as_bucket_view` for the same reason.

**Validation surface for 10a**
- Multi-process unit test (`tests/distributed/test_runtime_foundation.py`, `mp.spawn`, world=2 and 4): assert each rank reports correct `is_world_process_zero`, `is_local_process_zero`, and that `_save_checkpoint` writes exactly once on disk (not 4 copies).
- Contract test on a 1-rank `torchrun` run: artefact directory is byte-identical to the equivalent single-process run (modulo timestamps in `trainer_state.json`). This is the "10a does not change behaviour for single rank" guarantee.
- Reject-table test: `WORLD_SIZE=4 LOCAL_RANK=0 ... DPTrainer(args=DPTrainingArguments(deepspeed=...))` still raises `ValueError` from `__post_init__`.

### Phase 10b: Distributed sampling, accounting, and checkpoint/resume

**Goal.** Wire the rank-data policy chosen above so each rank receives the
right examples, the accountant knows what mechanism is running, and resume
restores per-rank state without replaying batches.

**New / changed args on `DPTrainingArguments`**:
- `dp_shard: Literal["per_rank", "global"] = "per_rank"` — declares the rank-data policy. Only meaningful when `world_size > 1`; on `world_size == 1` it must be `"per_rank"` (validated in `__post_init__`).
- Existing `dp_sampler`, `dp_max_batch_size`, `expected_batch_size` are unchanged in semantics; in `"per_rank"` mode the *local* sample rate equals the *global* sample rate (because each rank also has a `1/world` slice of the dataset), so we don't need a per-rank rate field.

**Sampler construction touchpoints** (`__init__.py:2992–3006` and `_dataloader.py`):
- In `get_train_dataloader`, before constructing the sampler, slice the dataset:
  ```python
  if self._ddp.world_size > 1 and a.dp_shard == "per_rank":
      from opaque.distributed import local_shard
      dataset = local_shard(dataset, rank=self._ddp.rank, world_size=self._ddp.world_size)
  ```
  with `dataset_size` and `sample_rate` recomputed off the shard. The
  *expected_batch_size* on the args is still the global batch size; the
  per-rank Poisson rate `q_local = (expected_batch_size / world) / |shard|` is
  numerically the same as the global rate by the shard-size identity, but it
  is computed locally so a non-uniform shard (last-rank remainder) is
  handled correctly.
- For `"global"` mode, do not slice; instead fold rank into the sampler key. Extend `_OpaqueEpochBaseBatchSampler._make_sampler` ([_dataloader.py:69–90](../../src/opaque/transformers/trainer/_dataloader.py)) to accept and apply a `rank` argument: `key=fold_in(fold_in(key(seed), epoch), rank)`. Keep this opt-in so `"per_rank"` mode preserves bit-identical seeding to the current single-process trainer (`fold_in(key(seed), epoch)` only).

**Accountant touchpoints** (`__init__.py:3454–3490`):
- Today the accountant builds `acc.poisson(_u(nm), sample_rate=sample_rate)` ([_init_.py:3478, 3485](../../src/opaque/transformers/trainer/__init__.py)). Branch on `dp_shard`:
  - `"per_rank"` and `world_size > 1`: still `acc.poisson(...)`. Sample rate is the global one (already correct).
  - `"global"` and `world_size > 1`: `acc.parallel_poisson(_u(nm), sample_rate=q_local, num_workers=world_size)`.
  - `world_size == 1`: unchanged.

**Checkpoint / resume**:
- Per-rank RNG snapshot already supported via [_checkpoint.py:69–81](../../src/opaque/transformers/trainer/_checkpoint.py) (`rng_state_path(ckpt_dir, rank=rank, world_size=world_size)`). Pass real `rank, world_size` from `_save_checkpoint` and `_load_rng_state` instead of the hard-coded `0, 1`.
- Sampler state-dict semantics depend on the policy. In `"per_rank"` mode the sampler is identical across ranks (epoch-keyed only); save once on rank 0 and broadcast on resume. In `"global"` mode every rank has its own `iter_count` (because key is rank-folded); save per-rank like RNG (`sampler_state_{rank}.pt`).
- Optimizer / DP runtime / accountant state are bit-identical across ranks (consequence of the architecture above), so save them only on rank-0; on resume, broadcast via `torch.distributed.broadcast_object_list` from rank 0.
- Add a `barrier()` between rank-0 finishing checkpoint write and any rank starting rotation / hub push, otherwise non-zero ranks proceed past the save before files exist.

**Touchpoints**:
- `_checkpoint.py` save/load helpers: thread `(rank, world_size)`; broadcast the sampler / optimiser state-dicts for `"per_rank"` resume.
- `__init__.py` `_save_checkpoint` (~line 3855), `_load_optimizer_and_scheduler`, `_load_rng_state` (resume path, around the existing call sites that the Phase 2 doc already describes).
- `_state.py` no changes.

**Validation surface for 10b**
- Per-rank determinism test (`mp.spawn`, world=4): the union of indices yielded by all ranks across one epoch in `"per_rank"` mode forms a partition of `range(|D|)` (within Poisson tolerance for the last rank's remainder).
- Cross-rank divergence test for `"global"` mode: same epoch index, two ranks emit different index sets with non-trivial overlap (Poisson duplication is the whole point).
- Accountant parity test: a 1-rank world produces the same `epsilon_at(delta)` as today's single-process trainer, modulo composition order. A 4-rank `"global"` run gives ε strictly larger than a 4-rank `"per_rank"` run on the same global noise multiplier (the parallel-Poisson penalty).
- Resume parity test (4-rank): `train(resume_from_checkpoint=...)` after a checkpoint at step N reproduces the next-step gradient norms and parameters that an uninterrupted run produces (already covered single-process at [tests/validation/test_dp_trainer.py]; extend to a `tests/distributed/` variant).

### Phase 10c: Distributed DP step

**Goal.** Make every optimizer step a correct single global DP-SGD update.

**Step-level surgery in `training_step` ([__init__.py:1714–1868](../../src/opaque/transformers/trainer/__init__.py))**

The single-process step today is:

```
1. (grads, aux), clip_state = grad_fn(params, *batch_args, state=clip_state)
2. fp16 overflow detection (skip step if non-finite)
3. noise_std = _noise_stddev(clip_state, noise_multiplier)
4. noisy_grads, noise_state = noise_fn(grads, noise_state, stddev=noise_std)
5. updates, opt_state = opt.update(noisy_grads, opt_state, params=trainable_params)
6. trainable_params = torchopt.apply_updates(trainable_params, updates)
```

The DDP step inserts collectives **between 1 and 3** (after clipping, before
noise). Specifically, between [line 1757 and line 1759](../../src/opaque/transformers/trainer/__init__.py):

```python
if self._ddp.is_distributed:
    from opaque.distributed import sum_gradients_, sync, gather_pytree
    # (a) sync clip state and per-example aux for adaptive clip — local-only otherwise
    ctx.clip_state, aux = sync(ctx.clip_state, aux)
    # (b) sum clipped grads across ranks
    sum_gradients_(grads)
    # (c) gather aux fields callbacks need (grad_norms etc.); only cheap fields
    aux = _gather_aux_for_metrics(aux)  # helper in _distributed.py
    # (d) fp16 finite-check must run on the post-allreduce gradient
```

Everything else is unchanged: noise is identical on every rank because the
shared key + same `noise_state` + same numerics yields identical samples (see
the existing distributed example at [examples/train_causal_lm.py:1409–1450](../../../../examples/train_causal_lm.py)).
Optimizer state stays in sync by virtue of pure-functional torchopt.

**Subtleties**
- The fp16 overflow detect at [line 1766–1776](../../src/opaque/transformers/trainer/__init__.py)
  must run on the post-AllReduce gradient. Otherwise rank A may see finite
  grads, rank B sees inf, and they diverge. Also use
  `reduce_scalar(int(grads_finite), op="min")` so any rank's overflow trips
  every rank.
- `aux` includes per-example tensors (`grad_norms`, `loss_values`,
  `clipped_grad_norms`, `group_norms`). Callbacks consume them at
  [line 1836–1866](../../src/opaque/transformers/trainer/__init__.py). For
  reporting parity with single-process we need the cluster-wide view, so
  gather these along dim 0 via `gather_pytree(aux)` (or `sync_object` for
  scalar aggregates like `clipping_rate`).
- `clip_state` is the only state that *must* round-trip via `sync()` before
  noise. For `FixedClipState` it is an assertion (`clipping_norm` matches
  across ranks); for `AdaptiveClipState` it aggregates the count and recomputes
  `clipping_norm`. The Opaque dispatcher already handles both ([state.py:245–271](../../../../packages/opaque-core/src/opaque/distributed/state.py)).
- The `on_pre_optimizer_step` callback hook at [line 1800–1807](../../src/opaque/transformers/trainer/__init__.py)
  receives the noisy grads. Once those are AllReduce'd and noised once, every
  rank fires the hook with identical arguments — no ordering issue.

**`tr_loss` aggregation** (`_maybe_log_save_evaluate` at [line 3318–3322](../../src/opaque/transformers/trainer/__init__.py)):

```python
window = max(1, global_step - self._globalstep_last_logged)
tr_loss_scalar = self._tr_loss.item()
```

Currently rank-local. To match HF's "loss is averaged across the cluster"
contract (HF does this implicitly through `accelerator.gather`), wrap with
`reduce_scalar(tr_loss_scalar, op="mean")` *just before* dividing by `window`.
The existing comment at [__init__.py:421–425](../../src/opaque/transformers/trainer/__init__.py)
already flags this: "when DDP is added, `_nested_gather(tr_loss)` will work".

**Distributed evaluation** — the bigger surgery, in
[evaluation_loop (line 2124–2300+)](../../src/opaque/transformers/trainer/__init__.py)
and [_eval.py](../../src/opaque/transformers/trainer/_eval.py):

1. **Eval sharding**: split the eval dataset across ranks via a
   `DistributedSampler`-equivalent (we don't use `torch.utils.data.DistributedSampler`
   directly because it requires `set_epoch`; a small helper that returns
   indices `range(rank, n, world)` is sufficient for stable, contiguous
   sharding). Drop-last is irrelevant since eval batches don't need to be
   uniform.
2. **Per-batch loss** is already a scalar; sum locally, AllReduce SUM via
   `reduce_scalar(total_loss, op="sum")` and `reduce_scalar(loss_samples, op="sum")` at
   [the finalize block (line 2313–2316)](../../src/opaque/transformers/trainer/__init__.py),
   then divide.
3. **`_PredictionAccumulator.finalize` ([_eval.py:_PredictionAccumulator])** must
   gather predictions, labels, inputs, losses across ranks before truncate /
   `nested_numpify`. Use `gather_pytree(...)` for tensor leaves; for
   `eval_use_gather_object=True`, use `dist.all_gather_object` (object payload
   path). Gather happens **after** padding-to-max-length across ranks (HF parity:
   [trainer.py:2721–2729 `pad_across_processes`](/workspaces/transformers/src/transformers/trainer.py)).
4. **`batch_eval_metrics=True` reducer path** at [line 2259–2293](../../src/opaque/transformers/trainer/__init__.py):
   user reducer sees per-batch tensors. We must gather **per batch** before the
   reducer sees them, otherwise the reducer's accumulator runs on rank-local
   data and `compute_result=True` produces a rank-local metric. Insert the
   gather at the top of the inner if-branch around line 2270.
5. **`include_for_metrics={"inputs", "loss"}`** payloads need the same
   gather treatment.
6. **`preprocess_logits_for_metrics`** runs in `prediction_step`. HF runs it
   *after* `pad_across_processes` and *before* gather; we currently run it
   inside `prediction_step` *before* the accumulator. Phase 3 documented this
   discrepancy (see Phase 3 "Limitations"); Phase 10c is where we align.
   Concretely: lift the call out of `prediction_step` and apply it to the
   already-gathered logits in `evaluation_loop` once the rank-collective
   payload is assembled.

**HF parameters surfaced**:
- `ddp_find_unused_parameters` — accepted, defaults to `None`. Because we
  don't wrap the model in `torch.nn.parallel.DistributedDataParallel`
  (functional path), this flag is consumed *only* when the user constructs a
  DDP-wrapped model and passes it in; we'll surface it in the constructor's
  validation but otherwise it's a no-op for the functional path. Document
  this clearly.
- `ddp_bucket_cap_mb`, `ddp_broadcast_buffers` — same treatment: accepted but
  only consumed when `model` is wrapped externally. The functional path uses
  raw AllReduce on the gradient pytree without DDP buckets, so these are
  inert. Don't warn (HF parity); document.
- `eval_use_gather_object` — *promoted out of `DP_INCOMPATIBLE_PARAMETERS`*
  ([_config.py:122–125](../../src/opaque/transformers/trainer/_config.py));
  consumed by the new gather path.
- `average_tokens_across_devices` — *promoted out of
  `DP_INCOMPATIBLE_PARAMETERS`* ([_config.py:126–129](../../src/opaque/transformers/trainer/_config.py));
  when true, AllReduce SUM the per-step token count and use the cluster-wide
  total in `compute_loss` normalisation and in the
  `include_num_input_tokens_seen` running counter
  ([__init__.py around the existing token counter](../../src/opaque/transformers/trainer/__init__.py)).
- `log_on_each_node`, `save_on_each_node`, `log_level_replica` — surfaced in
  10a, fully consumed here for any rank that emits.

### Phase 10d: Tests, examples, docs

**Tests** under `packages/opaque-transformers/tests/distributed/` (mirroring
[packages/opaque-dpsgd/tests/distributed/](../../../opaque-dpsgd/tests/distributed/)
which uses `mp.spawn` so neither `torchrun` nor a fixed launcher is required):

- `test_runtime_foundation.py` — 10a: rank/world helpers, save gating,
  rejection table under `WORLD_SIZE>1`, 1-rank parity.
- `test_distributed_sampler.py` — 10b: per-rank shard partition, global mode
  duplication, sampler resume parity.
- `test_distributed_step.py` — 10c: numerical equivalence between (a) a
  4-rank sharded run with `expected_batch_size=B` and (b) a 1-rank run with
  `expected_batch_size=B` and identical data/seed (the *whole point* of
  central DP-SGD). Tolerance: bit-identical noise (shared key) → bit-identical
  parameters; allow `atol=1e-6` for AllReduce-induced reduction order.
- `test_distributed_eval.py` — 10c: gather correctness for
  `compute_metrics`, `batch_eval_metrics`, `eval_use_gather_object` true and
  false, `include_for_metrics={"inputs","loss"}`.
- `test_accountant_modes.py` — 10b: ε from `parallel_poisson` strictly larger
  than ε from `poisson` at the same noise multiplier; 1-rank world reproduces
  today's value.
- All marked `pytestmark = pytest.mark.cuda` and skipped when
  `torch.cuda.device_count() < 2`, mirroring the existing core distributed tests.

**Example**: extend `examples/train_causal_lm.py` so that the DPTrainer
launch path documents the same `dist.init_process_group` / `local_shard` /
`fold_in(key, rank)` surgery the standalone DP-SGD example already shows
(or, cleaner, add a sibling `examples/dp_trainer_ddp.py`).

**Docs**:
- `docs/user-guide/distributed-trainer.md` — new page, mirrors
  [docs/user-guide/distributed.md](../../../../docs/user-guide/distributed.md)
  but in DPTrainer terms: "set `dp_shard='per_rank'`, launch with
  `torchrun --nproc-per-node=4 your_script.py`, and DPTrainer does the rest".
  Cover the `per_rank` vs `global` decision matrix.
- `docs/user-guide/huggingface.md` — add a "Distributed training" section
  pointing at the new page.
- This planning doc — flip the Phase 10 row in the summary table from
  **planned** to **implemented** as each sub-phase lands.

### Migration / rollout order

10a → 10b → 10c → 10d sequentially. 10a is non-behavioural for `world_size==1`
(safe to merge first), 10b is the privacy-relevant change (must land with
its accountant tests), 10c is the largest delta but only activates when
`world_size > 1`, 10d backfills the surface for users.

Phase 10c **must** happen on the multi-GPU machine; 10a and 10b can be drafted
single-process and validated on multi-GPU before merge, but a CPU-only smoke
test is *not* sufficient evidence — `mp.spawn` over CUDA is the gate.

### Out of scope for Phase 10 (deferred to Phase 12+)

- Multi-node DDP (rendezvous via TCP store / etcd). Single-node 4-GPU only.
- TPU/XLA, FSDP, DeepSpeed, SageMaker MP — already rejected; remain rejected.
- `torch.compile` under DDP — Phase 11a; rerun the precision/compile suite
  after 10 lands.
- Mixed-precision autocast under DDP — Phase 11b.
- Ray-driven multi-process HPO — Phase 12.
- Activation checkpointing under DDP with vmap — already supported by Opaque
  ([test_ddp_integration.py:53–65](../../../opaque-dpsgd/tests/distributed/test_ddp_integration.py)),
  no DPTrainer-side work needed beyond a smoke test.

**Files**: `trainer/__init__.py` (process helpers, save/log gates, sampler
plumbing, `training_step` collectives, eval gather, `tr_loss` reduce),
`trainer/_config.py` (`ddp_backend` validation, `dp_shard` field, promotion
of `eval_use_gather_object` / `average_tokens_across_devices` out of the
incompatibility table), `trainer/_dataloader.py` (`rank` arg on epoch
samplers), `trainer/_checkpoint.py` (already parameterised — thread real
rank/world), `trainer/_hub.py` (rank-0 gates), new
`trainer/_distributed.py` (`DDPState`, `should_log` / `should_save`,
`gather_aux_for_metrics`, eval-shard helper), new
`tests/distributed/` package, new `docs/user-guide/distributed-trainer.md`.

---

## Phase 11: Compile, Kernels, and Precision

Compilation and precision support must be proven around the functional DP step:
`make_functional`, `vmap`, per-example loss construction, clipping, noise,
accounting, and torchopt updates.

### Phase 11a: Compile and kernel validation

**Parameters**: `torch_compile`, `torch_compile_backend`,
`torch_compile_mode`, `use_liger_kernel`, `liger_kernel_config`,
`jit_mode_eval`.

**What to implement**:
- Keep compile and Liger flags rejected until correctness is proven against the
  non-compiled functional path.
- Validate `torch.compile(model, backend=..., mode=...)` before
  `make_functional`, including per-example gradient parity, clipped-gradient
  parity, optimizer update shape/state parity, and final metric parity.
- Define precedence between Liger patches and `opaque.performance.huggingface`
  patches before enabling `use_liger_kernel`.
- Validate `jit_mode_eval` separately in eval-only paths where no training
  `vmap(grad(...))` is involved.

**Files**: `trainer/__init__.py`, potential `_compile.py`, tests in
`tests/opaque_transformers/test_compile.py`.

### Phase 11b: Mixed precision autocast and fp16

**Parameters**: `fp16`, `fp16_full_eval`, `bf16`, `bf16_full_eval`, `tf32`,
`half_precision_backend`, `fp16_opt_level`.

**What to implement**:
- Keep Apex/Accelerate-specific knobs rejected, but support safe native
  precision paths where the DP math is validated.
- Split full-cast bf16 support from fp16 autocast support.
- Design autocast and `GradScaler` interaction around the functional optimizer
  step; clipping, noise calibration, and accountant state must operate on the
  intended numeric values.
- Re-run the precision suite under Phase 10 once distributed training exists.

**Files**: `trainer/__init__.py`, potential `_precision.py`, tests in
`tests/opaque_transformers/test_precision_training.py`.

---

## Phase 12: External HPO Execution Backends — implemented

Ray Tune is an external execution controller, not a local trial backend like
Optuna or a W&B sweep agent. The Phase 12 adapter mirrors HF's
`transformers.integrations.run_hp_search_ray` bullet-for-bullet so users get
HF-parity behavior with DP-correct checkpoint contents.

**Parameters / surfaces**: `backend="ray"`, `RAY_SCOPE` env var, all `tune.run`
kwargs (`resources_per_trial`, `progress_reporter`, `scheduler`, …).

**Implementation**:

- `_run_ray_search` in [`_hpo.py`](../../src/opaque/transformers/trainer/_hpo.py)
  packages the trainer for Ray's actor model: a single instance is pickled
  via `tune.with_parameters(_objective, local_trainer=trainer)` and shipped
  to each Tune actor, which then calls `trainer.train(trial=config_dict)`.
  The trainer reuses its existing dict-trial path (model_init reinvocation,
  trial-scoped output dir, callback handler rebuild).
- HF parity defaults applied identically:
  - `resources_per_trial` defaults to `{"cpu": 1, "gpu": 1}` when
    `trainer.args.n_gpu > 0` (same gate as HF's `run_hp_search_ray`, driven
    by `DPTrainingArguments._setup_devices` rather than a raw CUDA probe).
  - After defaults merge, `trainer.args._n_gpu` is set from
    `resources_per_trial["gpu"]` so per-trial allocation matches HF.
  - `progress_reporter` defaults to `CLIReporter(metric_columns=["objective"])`.
  - ASHA / Hyperband / Median / PBT schedulers raise the parity error if
    `do_eval=False` or `eval_strategy=NO`.
  - `dynamic_modules_import_trainable` wrapper preloads `datasets` dynamic
    modules inside each actor (HF's fix for issue #11565).
- `_scrub_for_pickling` swaps the memory tracker to skip-only (with the
  same warning HF emits when forcing skip for serialisation), drops
  `trainer.model` (rebuilt per actor via `call_model_init(trial)`), clears
  cached dataloader handles, and asserts `args.output_dir` is absolute
  (Tune chdirs each trial).  Only TensorBoard is popped from callbacks;
  other integrations may still break pickling — mirror HF or disable
  `report_to` for Ray sweeps.
- TensorBoard callback is popped before launch and re-attached after
  `tune.run` returns.
- Trainer hooks added for `HPSearchBackend.RAY` parity:
  - `_hp_search_setup` — `params = dict(trial); params.pop("wandb", None)`.
  - `_get_output_dir` — uses `ray.train.get_context().get_trial_id()`.
  - `_report_to_hp_search` — calls `ray.train.report(metrics, checkpoint=...)`
    with the optional checkpoint built via `_tune_save_checkpoint`.
  - `_tune_save_checkpoint(checkpoint_dir)` — DP analogue of HF's; writes a
    self-contained `checkpoint-<global_step>/` inside Ray's temp dir
    containing the model, `accountant.json`, `trainer_state.json`,
    `dp_runtime_state.pt`, `dp_optimizer.pt`, `rng_state.pth`, and
    `training_args.bin`. `save_only_model` is intentionally ignored —
    Ray expects checkpoints to be complete enough to resume after eviction.
- `BestRun` assembled from
  `analysis.get_best_trial(metric="objective", mode=direction[:3], scope=os.getenv("RAY_SCOPE", "last"))`
  with the `analysis` object exposed via `BestRun.run_summary` (HF parity).
- HF does **not** expose `ray_scope` as a `TrainingArguments` field; scope
  selection remains environment-driven (`RAY_SCOPE`).

- `hyperparameter_search(backend=None)` calls `default_dp_hp_backend()`,
  which walks **Optuna → Ray → W&B** (SigOpt skipped) — the same relative
  order as HuggingFace's `default_hp_search_backend` for the backends
  DPTrainer implements.

- Ray trial resume picks the **highest-step** `checkpoint-*` directory
  under the Ray unpack path (deterministic; avoids `next(glob)`).

**Optional dependency**: `pip install opaque-transformers[ray-hpo]` pulls
in `ray[tune]>=2.7,<3`. The lazy-import `ImportError` redirects users at
that extra. Symmetric extras `[optuna-hpo]`, `[wandb-hpo]`, and the
`[hpo]` umbrella mirror the pattern.

**Out of scope** (deferred):
- SigOpt — no DP-specific blocker, but HF's SigOpt path also assumes
  Accelerate state, and there is no demand surface yet.
- Sweep-level DP composition across trials — explicit non-goal; each
  trial's accountant is independent (matches Optuna / W&B today).
- Ray Train (separate from Ray Tune) — unrelated; Phase 12 is HPO only.

**Validation**:
- `tests/opaque_transformers/test_hpo.py` covers: SigOpt rejection,
  `_scrub_for_pickling` round-trips through `pickle.dumps`, absolute-path
  scrub assertion,
  end-to-end dispatch through a fake Ray stack with `BestRun` shape,
  scheduler-requires-eval parity, `_get_output_dir(trial)` reading the
  Ray trial id, `_tune_save_checkpoint` snapshot completeness, the
  guard rejecting calls outside the training loop, **default backend**
  resolution (`default_dp_hp_backend` + `backend=None` routing),
  **`_pick_latest_ray_resume_checkpoint`**, **`_sync_ray_trial_gpu_to_args`**,
  `RAY_SCOPE` selection behavior,
  / `args.n_gpu`-driven default `resources_per_trial`, and the memory-tracker
  warning on Ray scrub.

**Files**: [`trainer/_hpo.py`](../../src/opaque/transformers/trainer/_hpo.py)
(adapter + helpers), [`trainer/__init__.py`](../../src/opaque/transformers/trainer/__init__.py)
(four trainer-side hooks + `_tune_save_checkpoint`),
[`trainer/_config.py`](../../src/opaque/transformers/trainer/_config.py),
[`pyproject.toml`](../../pyproject.toml) (per-backend HPO extras),
[`tests/opaque_transformers/test_hpo.py`](../../tests/opaque_transformers/test_hpo.py).

---

## Phase 13: Optional Functional Optimizer Expansion

DPTrainer can only support optimizers whose state and update rules fit the
functional DP pipeline. Ordinary HF optimizer names are not enough: every
candidate needs a torchopt-compatible transform or an equivalent functional
implementation.

**Parameters / surfaces**: `optim`, `optim_args`, `optim_target_modules`,
`optimizer_cls_and_kwargs`, `optimizers`.

**What to implement**:
- Evaluate candidate optimizers one by one, starting from those that can be
  represented as functional torchopt transforms with explicit state.
- Keep bitsandbytes 8-bit/paged optimizers, Apex-fused optimizers,
  backend-specific XLA/NPU variants, and arbitrary `torch.optim.Optimizer`
  objects rejected unless they are reimplemented or wrapped with proven
  functional DP semantics.
- Decide whether `optim_target_modules` belongs here or in a separate
  parameter-group phase; per-layer optimizer targeting interacts with pytree
  state and clipping groups.
- Expand optimizer construction tests so every supported name verifies state
  initialization and at least one DP update.

**Files**: `trainer/__init__.py`, `_config.py`, optional `_optim.py`, tests in
`test_config.py` and optimizer-focused validation tests.

---

## Phase 14: Quantization and Model-Wrapper Policy

HF Trainer accepts or coordinates many model wrappers and placement policies
that can silently break functional per-example gradients. This phase decides
which are non-goals, which are explicit rejections, and which deserve future
support.

**Surfaces**: quantized / low-bit training, PEFT edge cases, `device_map`,
tensor parallelism, context/sequence parallelism, `parallelism_config`,
SageMaker model parallel, XLA/NPU-specific model paths.

**What to implement**:
- Add early validation that rejects unsafe wrappers before training starts,
  especially wrappers that hide parameters, shard parameters, change device
  placement behind the trainer, or replace optimizer semantics.
- Decide whether low-bit / quantized finetuning is a future target. If yes,
  prove per-example gradient correctness and functional optimizer compatibility
  for each wrapper family before enabling it.
- Keep PEFT cases that behave like ordinary PyTorch modules separate from
  quantized or device-mapped PEFT paths.
- Document the support matrix so users know which HF runtime conveniences are
  intentionally outside DPTrainer's execution model.

**Files**: `_config.py`, `trainer/__init__.py`, possible model-inspection
helper, tests covering representative wrappers when optional dependencies are
available.

---

## Summary: phase → parameters → files

| Phase | Parameters | Count | Key file changes |
|---|---|---|---|
| **1: LR Scheduler** | lr_scheduler_type, lr_scheduler_kwargs, warmup_ratio, warmup_steps | 4 | `__init__.py`: create_scheduler, loop integration |
| **2a: Basic saving** | output_dir, overwrite_output_dir, save_strategy, save_steps, save_safetensors, save_only_model, save_total_limit | 7 | `__init__.py` + new `_checkpoint.py` |
| **2b: Best model** | load_best_model_at_end, metric_for_best_model, greater_is_better | 3 | `__init__.py`, `_checkpoint.py` |
| **2c: Resume** | resume_from_checkpoint, restore_callback_states, ignore_data_skip | 3 | `__init__.py`, `_checkpoint.py` |
| **3a: Eval controls** | eval_on_start, eval_delay, prediction_loss_only | 3 | `__init__.py` |
| **3b: Eval memory** | eval_accumulation_steps, eval_do_concat_batches, batch_eval_metrics | 3 | `__init__.py` |
| **3c: Eval metrics** | include_inputs_for_metrics, include_for_metrics | 2 | `__init__.py` |
| **4: Precision** | bf16, fp16, tf32, no_cuda, use_cpu, use_mps_device, bf16/fp16_full_eval | 8 | `__init__.py` |
| **5a: Log strategy** | logging_strategy, logging_first_step, logging_nan_inf_filter | 3 | `__init__.py` (epoch-end flush bug fix, step-level NaN/Inf guard) |
| **5b: Tracking** | report_to, disable_tqdm, project, run_name, logging_dir | 5 | `_callback.py` (get_reporting_integration_callbacks), `_config.py` (remove ValueError) |
| **5c: Log metrics** | skip_memory_metrics, include_tokens_per_second, include_num_input_tokens_seen | 3 | `__init__.py` (TrainerMemoryTracker, token counting, log() start_time) |
| **5d: Reporting compatibility** | report_to integration behavior, artifacts, callback optimizer/scheduler visibility, trackio_space_id | Integration surface | `_callback.py`, `_hpo.py`, docs — **implemented** |
| **6a: Columns** | remove_unused_columns | 1 | `__init__.py` (signature columns, eager prune/collator wrapper) — **implemented** |
| **6b: DataLoader** | dataloader_prefetch_factor, dataloader_persistent_workers, torch_empty_cache_steps | 3 | `__init__.py` (loader kwargs, eval-loader reuse, loop cache-empty hook) — **implemented** |
| **7a: Determinism** | data_seed, full_determinism | 2 | `__init__.py` — **implemented** |
| **7b: Labels** | label_names, label_smoothing_factor | 2 | `__init__.py` — **implemented** |
| **7c: Misc** | auto_find_batch_size, neftune_noise_alpha, do_train/eval/predict, past_index, debug | 7 | `__init__.py`, `_config.py` — **implemented** |
| **8: Hub** | push_to_hub, hub_model_id, hub_strategy, hub_token, hub_private_repo, hub_always_push, hub_revision | 7 | `__init__.py` + new `_hub.py` — **implemented** |
| **9: HPO / Trainer contract** | train(trial), hyperparameter_search, model_init lifecycle, public Trainer helper methods | API surface | `__init__.py` + new `_hpo.py`, contract tests |
| **9.5: Runtime/HPO hardening** | unsupported runtime tables, optimizer rejection surface, Ray rejection, callback isolation | Boundary surface | `_config.py`, `_hpo.py`, tests — **implemented** |
| **10 (prereq): opaque-core sync registrations** | optimizer/schedule state types missing from `register_sync_type` registry | Cross-package surface | new `opaque/optimizers/distributed.py` (+ extend `_ensure_builtin_sync_types_loaded`), tests in `opaque-core/tests/distributed/` — **implemented** |
| **10a: DDP foundation** | local_rank, ddp_backend, ddp_timeout, log_on_each_node, save_on_each_node, log_level_replica, process gates | Runtime surface | new `_distributed.py`, `__init__.py` (process helpers + save/log/hub gates + per-rank RNG paths), `_config.py` (NCCL-only validation, `LOCAL_RANK`-aware device pick) — **implemented** |
| **10b: Distributed sampling/accounting** | `local_shard` of the dataset, shared per-epoch sampler key on every rank, regular `acc.poisson` at the global rate, per-rank RNG snapshots | Mechanism surface | `__init__.py` (sampler construction, accountant), `_dataloader.py`, `_config.py` — **implemented** (the `ddp_shard='global'` / `parallel_poisson` opt-in originally landed here was later removed) |
| **10c: Distributed step/eval** | step-level `sum_gradients_(grads)` + `sync(clip_state, aux)` + post-AllReduce fp16 finite-check, eval-shard + `gather_pytree` inside `finalize`, cluster-wide `total_loss` / `total_samples` reduction, promote `eval_use_gather_object` + `average_tokens_across_devices` out of rejection table | 6 | `__init__.py` (training_step, evaluation_loop, get_eval_dataloader, token counter), `_eval.py` (gather inside finalize), `_config.py` (table edits) — **implemented** |
| **10d: Tests, examples, docs** | `tests/distributed/` (5 scenarios via subprocess runner), `docs/user-guide/distributed-trainer.md` | Coverage surface | new test runner + 5 pytest scenarios + new user-guide page — **implemented** (CI smoke-test on 4× H100) |
| **11a: Compile/kernels** | torch_compile, torch_compile_backend, torch_compile_mode, use_liger_kernel, liger_kernel_config, jit_mode_eval | 6 | `__init__.py`, optional `_compile.py` — **planned** |
| **11b: Mixed precision** | fp16, fp16_full_eval, bf16/bf16_full_eval validation, tf32, half_precision_backend, fp16_opt_level | Precision surface | `__init__.py`, optional `_precision.py` — **planned** |
| **12: External HPO execution** | backend="ray", `RAY_SCOPE` env-based best-trial scope, external-controller trial execution | Backend surface | `_hpo.py`, `__init__.py`, `_config.py` — **implemented** |
| **13: Optimizer expansion** | optim, optim_args, optim_target_modules, optimizer_cls_and_kwargs, optimizers | Optimizer surface | `__init__.py`, `_config.py`, optional `_optim.py` — **planned** |
| **14: Quantization/wrappers** | quantization, PEFT edge cases, device_map, tensor/context/sequence parallelism, parallelism_config | Model wrapper surface | `_config.py`, `__init__.py`, model-inspection helpers — **planned** |
| **Total implemented** | | **~82** | |

---

## Parameters NOT implemented (with reasons)

| Parameter | HF Default | Reason not implemented |
|---|---|---|
| `per_gpu_train_batch_size` | None | **Deprecated** since Transformers 4.0; use `per_device_train_batch_size` |
| `per_gpu_eval_batch_size` | None | **Deprecated** since Transformers 4.0; use `per_device_eval_batch_size` |
| `adafactor` | False | **Deprecated**; use `optim="adafactor"` instead |
| `fp16_opt_level` | "O1" | **Apex-specific**; DPTrainer uses model dtype directly, no mixed-precision autocast wrapper |
| `half_precision_backend` | "auto" | **Accelerate-specific**; DPTrainer doesn't use Accelerate for mixed precision |
| `fp16_backend` | "auto" | **Deprecated** alias for `half_precision_backend` |
| `torchdynamo` | None | **Deprecated**; use `torch_compile` instead |
| `use_legacy_prediction_loop` | False | **Deprecated**; only the modern `evaluation_loop` exists |
| `group_by_length` | False | **DP-incompatible**: non-uniform batching breaks Poisson sampling; DP amplification by subsampling requires each example to be independently included with equal probability p = batch_size/n |
| `length_column_name` | "length" | **Depends on** `group_by_length` which is excluded |
| `dataloader_drop_last` | False | **Not applicable**: Poisson sampling produces variable-size batches by design; dropping the last batch is meaningless when every batch is independently sampled |
| `fsdp` | None | **Not supported**: DPTrainer uses Opaque's own DDP primitives, not PyTorch FSDP; per-example gradient computation via vmap is incompatible with FSDP parameter sharding |
| `fsdp_min_num_params` | 0 | **Depends on** `fsdp` which is not supported |
| `fsdp_config` | None | **Depends on** `fsdp` which is not supported |
| `fsdp_transformer_layer_cls_to_wrap` | None | **Depends on** `fsdp` which is not supported |
| `accelerator_config` | None | **Not supported**: DPTrainer replaces Accelerate's backward/optimizer step with functional DP-SGD; Accelerate's gradient accumulation, mixed precision, and distributed abstractions conflict with Opaque's per-example gradient mechanics |
| `parallelism_config` | None | **Not supported**: same as accelerator_config |
| `deepspeed` | None | **Not supported**: DeepSpeed ZeRO's parameter/gradient sharding is incompatible with vmap-based per-example gradient computation |
| `tpu_num_cores` | None | **Not supported**: TPU/XLA is not supported by Opaque's CUDA/CPU vmap backend |
| `tpu_metrics_debug` | False | **Depends on** TPU support which is excluded |
| `optim_target_modules` | None | **Deferred to Phase 13**: per-layer optimizer configuration; complex interaction with functional optimizer state |
| `trackio_space_id` | "trackio" | **Deferred to Phase 5d**: reporting integrations need callback-by-callback compatibility validation |
| `mp_parameters` | "" | **Not supported**: SageMaker model parallel — not a supported execution environment |
| `ray_scope` | "last" | **Not a DPTrainingArguments field (HF parity)**: best-trial scope is read from `RAY_SCOPE` env var by `_run_ray_search`. |
| `push_to_hub_model_id` | None | **Deprecated** alias; use `hub_model_id` |
| `push_to_hub_organization` | None | **Deprecated** alias |
| `push_to_hub_token` | None | **Deprecated** alias; use `hub_token` |

---

## Recommended execution order from current state

Phase 9.5 is the immediate stabilization layer: keep runtime/backend rejection
tests green, keep local Optuna/W&B HPO isolated, and keep the phase tracker in
sync with branch reality.

On a CPU/MPS/single-GPU development machine, do Phase 5d next. Reporting
compatibility is the best local feature-parity slice because it validates real
HF callback behavior without changing the DP mechanism or requiring multi-rank
execution.

On a multi-GPU machine, start Phase 10a before broader runtime work.
Distributed process helpers, rank-aware logging/saving, and explicit wrapper
rejection give us a stable place to hang DDP semantics without changing the DP
mechanism yet.

Then proceed to Phase 10b and Phase 10c: sampling/accounting policy first,
gradient/noise/eval synchronization second. This order keeps privacy semantics
ahead of performance plumbing.

Phase 5d can run before or in parallel with Phase 10 because reporting
compatibility is mostly callback-contract validation. Phase 11a/11b should wait
until the functional single-process path is stable under the new boundaries;
rerun the precision/compile checks under DDP after Phase 10 lands.

Phase 12, Phase 13, and Phase 14 are independent feature-parity tracks. Ray
execution, optimizer expansion, and quantization/model-wrapper support each need
their own explicit DP design rather than being enabled as incidental HF parity.
