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
  step.  Matches HF semantics at `gradient_accumulation_steps=1`.
- Cluster-wide logical batch is `per_device_train_batch_size *
  world_size` (exposed as the HF property `train_batch_size`).  The
  sample rate `q = train_batch_size / N_total` drives privacy
  accounting.
- Internal microbatch chunking is only activated by
  `auto_find_microbatch_size=True` on OOM retry — it splits the
  per-rank logical batch into smaller vmap calls without changing the
  logical batch or the sample rate (privacy-neutral).

DPTrainer has **no `gradient_accumulation_steps`** — every step is
one atomic clip-noise-step over one Poisson sample.

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

`clipping_norm` accepts a positive scalar (global clipping), a dict
keyed by regex on parameter names with a `"fallback"` entry
(per-group clipping), or a JSON / `key=value,...` string with the
same shape.

## Sampling and noise

| Field | Use |
|---|---|
| `sampling_mode` | Only `"poisson"` is supported today. |
| `sampling_kwargs` | Forwarded to the sampler.  `truncated_batch_size=N` caps Poisson draws at `N` (weaker privacy at the same `q` unless recalibrated). |
| `clipping_mode` | `"fixed"` (default), `"adaptive"`, or `"auto"`. |
| `clipping_kwargs` | Adaptive / AUTO-S kwargs (`target_clipping_rate`, `norm_max`, `gamma`). |
| `privacy_noise_mechanism` | `"gaussian"` is the only mechanism today. |
| `privacy_noise_mechanism_kwargs` | Mechanism extras — e.g. `bound=...` for the bounded Gaussian variant. |
| `noise_calibration_kwargs` | Calibration search bounds; defaults `{"min": 0.01, "max": 10.0, "tolerance": 1e-3}`. |

All dict-shaped fields accept a `Mapping`, a JSON object string, or
the HF-style comma string `"a=1,b=2"`.

## Compute / precision

| Field | Default | Notes |
|---|---|---|
| `bf16` / `fp16` | `False` | Autocast on the per-example loss closure.  `fp16` enables dynamic loss scaling. |
| `bf16_full_eval` / `fp16_full_eval` | `False` | Cast the model for the eval scope only. |
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
- `privacy_resume_without_accountant=True` opts in to the
  warmup-on-public-data, then-DP workflow — resume from a checkpoint
  with no `accountant.json` and the trainer treats prior training as
  zero DP cost.  Without this flag, missing `accountant.json` raises
  `FileNotFoundError`.
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

## See also

- [DPTrainer](dptrainer.md) — what to call once the args are configured.
- [Model patches](model-patches.md) — kernel and patch configuration.
- [API reference — transformers](../../reference/transformers.md) —
  every field, type, default, and validation rule.
