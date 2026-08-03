# TrainingArguments

`opaque.transformers.TrainingArguments` mirrors the subset of
HuggingFace `TrainingArguments` DPTrainer honours, plus DPTrainer's
own privacy / clipping / sampling / patching fields.  Unsupported HF
knobs are intentionally omitted from the surface.

This page covers the fields you'll touch most often, with the
non-obvious semantics called out.  For the complete field inventory
with every type / default / coercion rule, see
[API reference — transformers](../../reference/transformers.md#trainingarguments).

## Batch-size contract

DPTrainer interprets batch-size args differently from stock HF.  Read
this once before tuning:

- `per_device_train_batch_size` is the **per-rank logical Poisson
  batch** — the expected sample size drawn on each rank for one DP-SGD
  step.  (This matches the HF interpretation when gradient accumulation
  is one step — DPTrainer is permanently in that regime; see below.)
- Cluster-wide logical batch is `per_device_train_batch_size *
  world_size` (exposed as the HF property `train_batch_size`).  The
  sample rate `q = train_batch_size / N_total` drives privacy
  accounting.
- Internal microbatch chunking is only activated by
  `auto_find_microbatch_size=True` on OOM retry — it splits the
  per-rank logical batch into smaller vmap calls without changing the
  logical batch or the sample rate (privacy-neutral).

To grow the effective batch, raise `per_device_train_batch_size` (the
expected Poisson round size); the physical vmap chunk (`microbatch_size`,
auto-shrunk under `auto_find_microbatch_size`) is decoupled from it and
privacy-neutral.

## Privacy targets

The user-facing DP budget knobs:

```python
args = TrainingArguments(
    privacy_target_epsilon=8.0,
    privacy_target_delta=1e-5,          # default: 1 / (10 * dataset_size)
    clipping_norm=1.0,                  # scalar global clip, or per-group dict
    privacy_noise_multiplier=None,      # None ⇒ calibrate from epsilon
)
```

`privacy_target_epsilon` is the search target; calibration finds the
smallest noise multiplier that achieves it under the configured
sampler + composition.  Set `privacy_noise_multiplier` directly to
skip calibration.

**At least one of `privacy_noise_multiplier` / `privacy_target_epsilon`
must be set.**  Neither has a silent default — construction raises if
both are `None`.  The two valid shapes are:

- `privacy_target_epsilon=ε` (NM left as `None`) — calibrate the noise
  multiplier from the budget at `train()` start.
- `privacy_noise_multiplier=σ` (target_eps left as `None`) — fix the
  noise multiplier; accounted ε is reported but not constrained.

Setting `privacy_noise_multiplier=0.0` together with a
`privacy_target_epsilon` raises: the non-private path can't honour a
finite ε target, and silently dropping one of them would hide a
configuration mistake.

### Stop-at-ε

When `privacy_target_epsilon` is set alongside a non-zero
`privacy_noise_multiplier`, training halts at the first logging
boundary where the accumulated ε from the privacy accountant meets or
exceeds the target.  The halt records `state.privacy_target_epsilon_reached
= True` and surfaces as a normal early-stop control flow (callbacks see
the final eval / save / log pass).  The check runs every
`logging_steps` (so setting `logging_steps=0` disables it
silently — explicit logging is the contract for stop-at-ε visibility).
A resume against a checkpoint where the budget is already spent
short-circuits before the first training step.

`clipping_norm` accepts a positive scalar (global clipping), a dict
keyed by regex on parameter names with a `"fallback"` entry
(per-group clipping), or a JSON / `key=value,...` string with the
same shape.  It also accepts `math.inf` to **disable clipping** (the
single canonical no-clip value) — see the non-private baseline below.

## Non-private baseline

Set `privacy_noise_multiplier=0.0` to run the *same* trainer, mechanism,
sampler, and accounting surface with **no privacy** (ε = ∞) — a useful
baseline for measuring the utility cost of the DP machinery:

```python
import math

args = TrainingArguments(
    privacy_noise_multiplier=0.0,   # no noise (σ = 0)
    clipping_norm=math.inf,         # optional: disable clipping too
)
```

- **No noise** — the realized noise standard deviation is
  `noise_multiplier × clip = 0`, so gradients pass through unchanged.
- **ε = ∞** — the accountant composes a non-private step, so
  `metrics["privacy_epsilon"]` is reported as `inf` (a faithful "no
  guarantee" output, not an error).  Calibration is skipped and the
  resolved multiplier is recorded as `0.0` (source `"fixed"`).
- **Clipping is independent.** It stays on by default (the configured
  `clipping_norm` / `clipping_mode`); keep it for a "clipping-only"
  ablation, or disable it with `clipping_norm=math.inf` for plain
  non-private SGD.

This works for both `"gaussian"` (DP-SGD) and the `mf_*` (DP-FTRL)
mechanisms.  Disabling clipping (`clipping_norm=math.inf`) is rejected
unless `privacy_noise_multiplier=0.0`, since infinite sensitivity with
noise would yield infinite noise and `NaN` gradients.

## Sampling and noise

| Field | Use |
|---|---|
| `sampling_mode` | `"auto"` (default) pairs the sampler with `privacy_noise_mechanism`; explicit values `{"poisson", "random_allocation", "k_out_of_t", "b_min_sep", "balls_in_bins", "cyclic_poisson", "sequential"}` are validated against the mechanism's allow-list. |
| `sampling_kwargs` | Forwarded to the sampler. `truncated_batch_size=N` caps Poisson draws at `N` and is unavailable for random allocation. |
| `clipping_mode` | `"fixed"` (default), `"adaptive"`, or `"auto"`.  `adaptive` is rejected under any `mf_*` mechanism (MF noise requires constant per-step sensitivity). |
| `clipping_kwargs` | Adaptive / AUTO-S kwargs (`target_clipping_rate`, `norm_max`, `gamma`). |
| `privacy_noise_mechanism` | `"gaussian"` (default, DP-SGD), or one of the DP-FTRL matrix-factorization mechanisms: `"mf_band"`, `"mf_blt"`, `"mf_bisr"`, `"mf_bsr"`, `"mf_lambda_cgd"`, `"mf_identity"`. |
| `privacy_noise_mechanism_kwargs` | Mechanism extras.  For `"gaussian"`: e.g. `bound=...` for the bounded Gaussian variant.  For `mf_*`: per-strategy kwargs (auto-filled from Mellum-shaped defaults — see below). |
| `noise_calibration_kwargs` | Calibration search bounds; defaults `{"min": 0.01, "max": 10.0, "tolerance": 1e-3}`. |

All dict-shaped fields accept a `Mapping`, a JSON object string, or
the HF-style comma string `"a=1,b=2"`.

### DP-FTRL mechanisms

Picking a `mf_*` mechanism auto-resolves the sampler and auto-fills
the strategy kwargs:

| `privacy_noise_mechanism` | Auto-resolved sampler | Default kwargs |
|---|---|---|
| `mf_band` | `b_min_sep` (or explicit `poisson`) | `{"bands": 16}` |
| `mf_blt` | `balls_in_bins` | `{"max_buffers": 16}` |
| `mf_bisr` | `balls_in_bins` | `{"bandwidth": 4}` |
| `mf_bsr` | `balls_in_bins` | `{"bandwidth": 8, "alpha": 1.0, "beta": 0.9}` |
| `mf_lambda_cgd` | `balls_in_bins` | `{"lambda_": 0.5}` |
| `mf_identity` | `poisson` | `{}` |

`"auto"` remains Poisson for Gaussian and identity MF. Gaussian explicitly
accepts `sampling_mode="random_allocation"` and `sampling_mode="k_out_of_t"`.
For k-out-of-t, set `sampling_kwargs={"total_participations": k}` so each
example participates in exactly `k` optimizer steps (uniform over the run).
Identity MF explicitly accepts `sampling_mode="balls_in_bins"`. Horizon modes
are adapted to the trainer's step-wise accountant through
`opaque.accounting.per_step`.

So the minimal DP-FTRL configuration is one field:

```python
args = TrainingArguments(privacy_noise_mechanism="mf_band")
# sampling_mode resolves to "b_min_sep"
# privacy_noise_mechanism_kwargs resolves to {"bands": 16}
```

Defaults are tuned for a Mellum/Kstack-shaped causal-LM target;
they're a sensible starting point, not universally optimal.  Override
any kwarg explicitly via `privacy_noise_mechanism_kwargs={...}`; user
keys win on collision with the defaults table.

Mechanism constraints (validated at construction):

- BandMF requires `bands <= total_steps`; shrink `bands` for short
  runs.
- BallsInBins requires `total_steps % num_bins == 0` where
  `num_bins = expected_steps_per_epoch` (so configure `max_steps` /
  dataset / batch size to satisfy this).
- BSR requires `alpha > beta` (paper constraint).

## Compute / precision

| Field | Default | Notes |
|---|---|---|
| `bf16` | `False` | bf16 autocast on the per-example loss closure. |
| `bf16_full_eval` | `False` | Cast the model to bf16 for the eval scope only. |
| `gradient_checkpointing` | `False` | Pair with `gradient_checkpointing_kwargs={"use_reentrant": False}` — reentrant checkpointing doesn't compose with vmap. |
| `torch_compile` | `False` | Compiles the per-example loss closure (not the model).  Tries `fullgraph=True` first; falls back with a warning. |

## Patches and kernels

Three flags drive the model patching at trainer construction time:

| Field | Default | Effect |
|---|---|---|
| `use_compat_patches` | `True` | vmap-safety patches (eager attention, batchify, vmap-safe masking / collator / checkpoint hooks). |
| `use_performance_kernels` | `False` | CUDA + Triton kernel group (RoPE, RMSNorm, SwiGLU/GeGLU, cross-entropy).  Auto-`False` on hosts without CUDA + Triton. |
| `performance_kernels_config` | `None` | Flat dict forwarded as kwargs to `apply_model_patches` / `apply_runtime_patches` — per-key override. |

See [Model patches](model-patches.md) for the full configuration matrix.

## Saving and resume

Standard HF save fields work as expected:

| Field | Default | Effect |
|---|---|---|
| `save_strategy` | `"steps"` | One of `{"no", "steps", "epoch", "best"}`.  `"best"` auto-injects `BestModelSaveCallback`. |
| `save_steps` / `save_total_limit` / `save_safetensors` | — | Standard HF semantics. |
| `save_on_each_node` | `False` | Every node's rank-0 writes a copy (for node-local storage). |
| `save_only_model` | `False` | Skip the DP runtime bundle (optimizer / sampler / RNG); ships weights + `accountant.json` only. |
| `load_best_model_at_end` | `False` | Restore the best-eval checkpoint after `train()`.  Raises if no improving step was recorded. |

Resume claims:

- `train(resume_from_checkpoint=<path>)` restores model weights,
  optimizer / clip / noise state, sampler cursor, RNG snapshots, and
  the privacy accountant in one call.
- The saved accountant is the **prefix**: heterogeneous composition
  with the remaining steps is DP-valid.  Calibration runs over the
  remaining steps to hit the original `privacy_target_epsilon`
  against that prefix.
- `resume_from_checkpoint` requires a **complete DP checkpoint**
  (`dp_state.pt` + `dp_optimizer.pt` + `accountant.json`).  A
  weights-only export (`save_only_model=True`, an HF checkpoint, a
  pretrained model) is rejected — to start a fresh DP run from such
  weights, load them at construction (`model=...`); the run begins with
  a zero accountant, which is correct only when the prior training had
  no DP cost (e.g. public-data warmup).
- `ignore_data_skip=True` skips sampler-state restore (useful when
  the dataset shape changed between runs); the resumed run starts
  each epoch from a fresh subsample sequence.
- `restore_callback_states_from_checkpoint=True` reads saved callback
  state and copies attributes back onto the live callback instances
  (e.g. `EarlyStoppingCallback`'s patience counter).

## Evaluation

| Field | Default | Effect |
|---|---|---|
| `eval_strategy` | `"no"` | One of `{"no", "steps", "epoch"}`. |
| `eval_steps` | `None` | Falls back to `logging_steps` when `eval_strategy="steps"` and unset. |
| `prediction_loss_only` | `False` | Skip logits / labels materialisation; only loss reaches `compute_metrics`. |
| `include_for_metrics` | `[]` | Subset of `{"inputs", "loss"}`.  `"loss"` switches to the vmap'd per-example eval (real per-example losses). |
| `metric_for_best_model` | `None` | Required when `save_strategy="best"` or `load_best_model_at_end=True`.  Auto-prefixed with `eval_` if missing. |

## Distributed

- `local_rank`, `ddp_backend`, `ddp_timeout` follow stock HF.
- `average_tokens_across_devices=True` (default) — averages per-rank
  token counts into a cluster-wide total for `num_input_tokens_seen` /
  `train_tokens_per_second`.
- DPTrainer is DDP-only; `DataParallel` is not supported.

For per-rank sharding, accountant cluster-wide composition, and
rank-gated checkpointing, see
[Distributed DPTrainer](../distributed-trainer.md).

## Converting from HF / TRL configs

Rather than hand-port an upstream config, convert it with the classmethod on
the matching opaque config:

```python
from opaque.transformers import TrainingArguments

args = TrainingArguments.from_hf(hf_args, privacy_target_epsilon=8.0)
```

- `TrainingArguments.from_hf(hf_args, …)` — from `transformers.TrainingArguments`
- `SFTConfig.from_trl(trl_cfg, …)` — from `trl.SFTConfig`
- `DPOConfig.from_trl(trl_cfg, …)` — from `trl.DPOConfig`

(`SFTConfig` / `DPOConfig` live in `opaque.transformers.trl`.)

What the converter does:

- **renames** legacy names (`per_gpu_train_batch_size`, `lr_scheduler_type`, …);
- **collapses** the HF effective batch (`per_device_train_batch_size ×
  gradient_accumulation_steps`) into the logical Poisson batch, with
  `per_device_train_batch_size` becoming the vmap `microbatch_size` and
  `auto_find_batch_size → auto_find_microbatch_size`;
- **loosely maps** `max_grad_norm → clipping_norm` (no warning — pass an
  explicit `clipping_norm=` to override);
- **remaps** optimizers (`adamw_torch`/`adamw_hf → adamw`,
  `adamw_torch_fused → adamw` + `optim_args={"fused": True}`,
  `adafactor=True → optim="adafactor"`) and `use_liger_kernel →
  use_performance_kernels`;
- **drops** irrelevant fields (with a `RuntimeWarning` when non-default), and
  **raises** with a per-field rationale on unsupported ones (`fp16`, `fsdp`,
  paged optimizers, …).

A DP knob is required as an override (`privacy_noise_multiplier=` or
`privacy_target_epsilon=`) — upstream configs carry no privacy budget. Any
other keyword overrides the converted field **by name** after translation
(e.g. `use_performance_kernels=True`); performance kernels default OFF on
conversion to match HF/TRL since their default can't be distinguished from an
explicit value.

## Migrating from HF: unsupported arguments

`TrainingArguments` is a standalone dataclass, not a subclass of
`transformers.TrainingArguments`, so passing an unsupported HF knob to the
constructor raises `TypeError` (the converters above translate or drop
these for you). For reference, the notable ones:

| HF argument | Why it's unsupported | DPTrainer alternative |
| --- | --- | --- |
| `group_by_length`, `length_column_name` | Length-bucketed batching breaks the equal per-example inclusion probability Poisson amplification relies on | Leave examples unsorted; Poisson sampling handles variable lengths |
| `dataloader_drop_last` | The Poisson / random samplers produce variable-size batches, so dropping a "last batch" is meaningless; the sequential batch sampler already enforces drop-last internally where it matters for correctness | n/a (handled by the sampler) |
| `deepspeed`, `fsdp`, `fsdp_config`, `accelerator_config`, `parallelism_config` | Parameter/gradient sharding is incompatible with vmap per-example gradients | Use Opaque's built-in DDP (`torchrun` + sharded data) |
| `tpu_num_cores`, `mp_parameters` | TPU/XLA and SageMaker MP are not supported execution backends | CUDA / CPU only |
| `fp16`, `fp16_full_eval`, `fp16_opt_level`, `half_precision_backend`, `fp16_backend` | fp16 dynamic loss scaling adds a per-example unscale-before-clip step for no benefit on bf16-capable hardware | `bf16=True` (native bf16 autocast; no loss scaler) |
| `optim="adamw_8bit"` / paged / Apex-fused | No functional torchopt equivalent | A supported `optim` name (see the optimizer table) |
| `batch_eval_metrics` | Streaming metric reduction not implemented | Use `eval_accumulation_steps` to bound eval memory |

`per_gpu_*` and the deprecated `push_to_hub_*` aliases are *renamed* to their
modern equivalents; `adafactor=True` is *mapped* to `optim="adafactor"`.
`torchdynamo` and other superseded flags are dropped — use their modern
replacements.

## See also

- [DPTrainer](dptrainer.md) — what to call once the args are configured.
- [Model patches](model-patches.md) — kernel and patch configuration.
- [API reference — transformers](../../reference/transformers.md) —
  every field, type, default, and validation rule.
