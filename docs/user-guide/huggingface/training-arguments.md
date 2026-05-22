# TrainingArguments

`opaque.transformers.TrainingArguments` is a standalone dataclass that
mirrors the subset of HuggingFace `TrainingArguments` DPTrainer
honours, plus DPTrainer-specific fields.  Unsupported HF knobs are
intentionally omitted from the surface (see
[Troubleshooting — unsupported features](troubleshooting.md#unsupported-hf-trainer-features)).

This page is the field reference, grouped by concern.  Fields tagged
**DPT** are DPTrainer-specific (no equivalent on stock HF
`TrainingArguments`).

## Batch-size contract

DPTrainer interprets batch-size args differently from stock HF.  Read
once before tuning:

- `per_device_train_batch_size` is the **per-rank logical Poisson batch
  size** — the expected size of the sample drawn on each rank for one
  DP-SGD step.  Matches HF semantics at `gradient_accumulation_steps=1`.
- Cluster-wide logical batch is `per_device_train_batch_size *
  world_size` (exposed as the HF property `train_batch_size`).  The
  sample rate `q = train_batch_size / N_total` drives privacy
  accounting.
- Internal microbatch chunking — only activated by
  `auto_find_microbatch_size` on OOM retry — splits the per-rank
  logical batch into smaller vmap calls.  This never changes the
  logical batch and is privacy-neutral.

DPTrainer has **no `gradient_accumulation_steps`**: every DP-SGD step
is one atomic clip-noise-step over one Poisson sample.

## Privacy (DPT)

The user-facing DP knobs.  Every field here is DPTrainer-specific.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `privacy_target_epsilon` | `float` | `8.0` | User target ε.  Calibration searches for the smallest noise multiplier that achieves this. |
| `privacy_target_delta` | `float \| None` | `None` | User target δ.  When unset, computed automatically as `1 / (10 * dataset_size)`. |
| `clipping_mode` | `str` | `"fixed"` | One of `{"fixed", "adaptive", "auto"}`. |
| `clipping_norm` | `float \| dict[str, Any] \| str` | `1.0` | Per-example DP clip bound.  Scalar for global clipping; JSON dict with `"fallback"` key for per-group clipping (keys are regex patterns over parameter names). |
| `clipping_kwargs` | `dict[str, Any]` | `{}` | Extra kwargs for adaptive / auto clipping (`target_clipping_rate`, `norm_max`, `gamma`). |
| `sampling_mode` | `str` | `"poisson"` | Only `"poisson"` is currently supported. |
| `sampling_kwargs` | `dict[str, Any]` | `{}` | Poisson sampler kwargs.  `truncated_batch_size` caps batches (weaker privacy than plain Poisson at the same `q` unless recalibrated). |
| `privacy_noise_mechanism` | `str` | `"gaussian"` | Only `"gaussian"` currently supported. |
| `privacy_noise_multiplier` | `float \| None` | `None` | Fixed σ.  When unset, calibration searches for the value that achieves `privacy_target_epsilon`. |
| `privacy_noise_radius` | `float` | `3.0` | Calibration search bound. |
| `privacy_noise_mechanism_kwargs` | `dict[str, Any]` | `{}` | Extra kwargs forwarded into `opaque.dpsgd.noise.gaussian_noise` (e.g. `bound` for the bounded variant). |
| `noise_calibration_kwargs` | `dict[str, Any]` | `{}` | Calibration search bounds; defaults: `{"min": 0.01, "max": 10.0, "tolerance": 1e-3}`. |
| `privacy_resume_without_accountant` | `bool` | `False` | Opt-in to resume from a checkpoint missing `accountant.json`.  Use for the *warmup-on-public-data, then DP-fine-tune* workflow where the resumed checkpoint has zero prior DP cost.  Default `False` raises on missing accountant. |

The dict-shaped fields (`clipping_kwargs`, `sampling_kwargs`,
`noise_calibration_kwargs`, `privacy_noise_mechanism_kwargs`,
`optim_args`, `lr_scheduler_kwargs`,
`performance_kernels_config`, `gradient_checkpointing_kwargs`)
accept any of:

- `Mapping` (including OmegaConf `DictConfig`)
- JSON object string: `'{"a": 1, "b": 2}'`
- HF-style comma string: `"a=1,b=2"`
- `None`

The trainer normalises to `dict[str, Any] | None` at construction
time.

## Compute / precision

| Field | Type | Default | Purpose |
|---|---|---|---|
| `use_cpu` | `bool` | `False` | Pin to CPU even if CUDA is available. |
| `use_mps_device` | `bool` | `False` | Use MPS (Apple Silicon).  CPU / MPS skip the Triton kernel group automatically. |
| `bf16` | `bool` | `False` | Enable bf16 autocast on the loss closure. |
| `fp16` | `bool` | `False` | Enable fp16 autocast + dynamic loss scaling.  Mutually exclusive with `bf16`. |
| `bf16_full_eval` | `bool` | `False` | Cast the model to bf16 for the eval scope only. |
| `fp16_full_eval` | `bool` | `False` | Cast the model to fp16 for the eval scope only. |
| `tf32` | `bool \| None` | `None` | Toggle TF32 on Ampere+ GPUs. |
| `gradient_checkpointing` | `bool` | `False` | Enable activation recomputation.  Pair with `gradient_checkpointing_kwargs={"use_reentrant": False}` for vmap-safety. |
| `gradient_checkpointing_kwargs` | `dict \| str \| None` | `None` | Forwarded to `model.gradient_checkpointing_enable(...)`. |
| `torch_compile` | `bool` | `False` | Wrap the per-example loss closure with `torch.compile`.  Tries `fullgraph=True` first; falls back to `fullgraph=False` with a warning on first-call failure. |
| `torch_compile_backend` | `str \| None` | `None` | Defaults to `"inductor"` when compile is on. |
| `torch_compile_mode` | `str \| None` | `None` | One of `{"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}`. |

### Patch / kernel surface (DPT)

These three drive `apply_runtime_patches` + `apply_model_patches` at
trainer construction time — see [Model patches](model-patches.md):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `use_compat_patches` | `bool` | `True` | Apply vmap-safety patches (eager-attention, batchify, vmap-safe masking / collator / checkpoint hooks).  Set `False` for custom models that don't need them. |
| `use_performance_kernels` | `bool` | `False` | Enable the CUDA + Triton kernel group (`rope`, `rms_norm`, `activation`, `cross_entropy`).  Auto-`False` on hosts without CUDA + Triton. |
| `performance_kernels_config` | `dict \| str \| None` | `None` | Flat dict of opaque-patches kwargs forwarded as-is.  Keys: `rope`, `rms_norm`, `activation`, `cross_entropy`, `fused_linear_cross_entropy`, `kv_cache`, `eager_attention`, `batchify`. |

`kv_cache` is in the `performance` bucket (not `kernels`) and stays on
by default regardless of `use_performance_kernels`.  Disable it
explicitly via `performance_kernels_config={"kv_cache": False}` for
models whose forward depends on the HF `DynamicCache`.

## Batch sizes

| Field | Type | Default | Purpose |
|---|---|---|---|
| `per_device_train_batch_size` | `int` | `8` | Per-rank logical Poisson batch size (see [Batch-size contract](#batch-size-contract)). |
| `per_device_eval_batch_size` | `int` | `8` | Eval batch size (fixed, not Poisson). |
| `eval_accumulation_steps` | `int \| None` | `None` | Move accumulated eval tensors to CPU every N batches (memory control). |
| `eval_delay` | `float` | `0.0` | Skip eval for the first N steps (or epochs, depending on strategy). |
| `auto_find_microbatch_size` | `bool` | `False` | DPT.  On OOM during training, halve the microbatch size and retry.  Logical batch stays the same; privacy-neutral. |

## Optimizer / LR

| Field | Type | Default | Purpose |
|---|---|---|---|
| `learning_rate` | `float` | `5e-5` | |
| `weight_decay` | `float` | `0.0` | |
| `adam_beta1` | `float` | `0.9` | |
| `adam_beta2` | `float` | `0.999` | |
| `adam_epsilon` | `float` | `1e-8` | |
| `optim` | `str` | `"adamw"` | Torchopt name.  Supported: `{"adam", "adamw", "sgd", "rmsprop", "adagrad", "adafactor", "ademamix", "lion", "radam", "adadelta", "schedule_free"}`.  HF aliases (`adamw_torch`, `adamw_torch_fused`, `lion_32bit`, …) are mapped to the matching opaque factory.  HF names without a DP-aware mapping (8-bit, paged, GaLore, fused-CUDA, NPU, XLA) are rejected with a redirect message. |
| `optim_args` | `dict \| str \| None` | `None` | Optimizer kwargs.  `dict[str, Any]` (incl. OmegaConf), JSON, or HF-style comma string. |
| `lr_scheduler_type` | `SchedulerType \| str` | `"linear"` | HF enum value. |
| `lr_scheduler_kwargs` | `dict \| str \| None` | `{}` | Schedule kwargs. |
| `warmup_ratio` | `float` | `0.0` | Warmup as fraction of `max_steps`. |
| `warmup_steps` | `int` | `0` | Warmup steps; mutually exclusive with non-zero `warmup_ratio`. |

## Training duration

| Field | Type | Default | Purpose |
|---|---|---|---|
| `num_train_epochs` | `float` | `3.0` | Epoch count.  `state.epoch` is fractional during the inner loop. |
| `max_steps` | `int` | `-1` | When `>= 0`, overrides `num_train_epochs` (the loop stops at this step). |

## Logging

| Field | Type | Default | Purpose |
|---|---|---|---|
| `log_level` | `str` | `"passive"` | Trainer log level on rank 0. |
| `log_level_replica` | `str` | `"warning"` | Trainer log level on non-zero ranks. |
| `log_on_each_node` | `bool` | `True` | If `False`, only world-rank 0 logs. |
| `logging_dir` | `str \| None` | `None` | Defaults to `{output_dir}/runs`. |
| `logging_strategy` | `str` | `"steps"` | One of `{"no", "steps", "epoch"}`. |
| `logging_first_step` | `bool` | `False` | Emit a log row at step 0. |
| `logging_steps` | `float` | `500` | Fractional values `< 1` interpreted as `fraction * max_steps`. |
| `disable_tqdm` | `bool \| None` | `None` | Auto-inferred from log level. |
| `report_to` | `str \| list[str] \| None` | `None` | Integrations: `"wandb"`, `"tensorboard"`, `"mlflow"`, …  `"all"` expands; `None`/`"none"`/`[]` disables. |
| `run_name` | `str \| None` | `None` | Used by W&B / TB / MLflow as the run name. |
| `project` | `str \| None` | `None` | Used by W&B as the project. |

## Saving

| Field | Type | Default | Purpose |
|---|---|---|---|
| `save_strategy` | `str` | `"steps"` | One of `{"no", "steps", "epoch", "best"}`.  `"best"` auto-injects `BestModelSaveCallback`. |
| `save_steps` | `float` | `500` | Fractional values `< 1` interpreted as `fraction * max_steps`. |
| `save_total_limit` | `int \| None` | `None` | Older checkpoints rotated out via `rotate_checkpoints`. |
| `save_safetensors` | `bool` | `True` | Use safetensors (default) vs `.bin` (pickle).  Load supports both. |
| `save_on_each_node` | `bool` | `False` | If `True`, every node's rank-0 process writes a checkpoint copy. |
| `save_only_model` | `bool` | `False` | Skip the DP runtime bundle (`dp_state.pt`, `dp_optimizer.pt`, `accountant.json`).  Useful for shipping a final model; resume from this checkpoint requires `privacy_resume_without_accountant=True`. |
| `restore_callback_states_from_checkpoint` | `bool` | `False` | Restore per-callback state on resume (`EarlyStoppingCallback`'s patience counter, …). |

## Evaluation

| Field | Type | Default | Purpose |
|---|---|---|---|
| `eval_strategy` | `str` | `"no"` | One of `{"no", "steps", "epoch"}`. |
| `eval_steps` | `float \| None` | `None` | Coerces to int if `> 1`; `< 1` interpreted as fraction.  Falls back to `logging_steps` when `eval_strategy="steps"` and `eval_steps` unset. |
| `eval_on_start` | `bool` | `False` | Run one eval pass before any training steps. |
| `eval_do_concat_batches` | `bool` | `True` | Concat per-batch predictions / labels at finalize.  Set `False` to expose per-batch lists. |
| `prediction_loss_only` | `bool` | `False` | Skip logits / labels materialisation — only loss reaches `compute_metrics`. |
| `include_for_metrics` | `list[str]` | `[]` | Opt into the optional `EvalPrediction` fields.  Values: `"inputs"`, `"loss"`.  `"loss"` switches eval to the vmap'd per-example closure (real per-example losses, not the batch mean repeated). |
| `metric_for_best_model` | `str \| None` | `None` | Eval metric name.  Auto-prefixed with `eval_` if missing.  Required when `save_strategy="best"` or `load_best_model_at_end=True`. |
| `greater_is_better` | `bool \| None` | `None` | Direction.  Inferred from the metric name (`loss` suffix → `False`). |
| `load_best_model_at_end` | `bool` | `False` | After training, restore the checkpoint that scored best on `metric_for_best_model`.  Raises if no checkpoint was recorded — see [Checkpoint and resume — load-best failure modes](checkpointing.md#load_best_model_at_end-failure-modes). |
| `ignore_data_skip` | `bool` | `False` | Skip sampler-state restore on resume.  Use when dataset size has changed between runs. |

## DataLoader

| Field | Type | Default | Purpose |
|---|---|---|---|
| `dataloader_num_workers` | `int` | `0` | Worker processes for the train / eval loaders. |
| `dataloader_persistent_workers` | `bool` | `False` | Keep workers alive across the (synthetic) epoch boundary. |
| `dataloader_pin_memory` | `bool` | `True` | Auto-disabled on CPU / MPS. |
| `dataloader_prefetch_factor` | `int \| None` | `None` | Requires `dataloader_num_workers > 0`. |
| `dataloader_drop_last` | `bool` | `False` | Drop incomplete final batch. |
| `remove_unused_columns` | `bool` | `True` | Strip dataset columns not in the model `forward` signature. |
| `torch_empty_cache_steps` | `int \| None` | `None` | Call `torch.cuda.empty_cache()` every N steps. |

## Distributed

| Field | Type | Default | Purpose |
|---|---|---|---|
| `local_rank` | `int` | `-1` | Deprecated.  Read from `LOCAL_RANK` env var. |
| `ddp_backend` | `str \| None` | `None` | One of `{"nccl", "gloo", "mpi", "xccl", "hccl", "cncl", "mccl"}`.  Auto-resolved when unset. |
| `ddp_timeout` | `int` | `1800` | Seconds passed to `init_process_group(timeout=...)`. |
| `average_tokens_across_devices` | `bool` | `True` | DPT.  Under DDP, average per-rank token counts into a cluster-wide total for `num_input_tokens_seen` and `train_tokens_per_second`. |

For DPTrainer-specific distributed behaviour (per-rank sharding,
cluster-wide accountant composition, rank-gated checkpointing), see
[Distributed Training](../distributed-trainer.md).

## Labels

| Field | Type | Default | Purpose |
|---|---|---|---|
| `label_names` | `list[str] \| None` | `None` | Auto-discovered from `find_labels(model.__class__)`; defaults to `["labels"]`. |
| `label_smoothing_factor` | `float` | `0.0` | When `> 0`, the trainer applies `F.cross_entropy(..., label_smoothing=...)` on the exposed logits inside `compute_per_example_loss`.  Honored by the opaque CE kernels too — passed through as a loss kwarg. |

## Reproducibility

| Field | Type | Default | Purpose |
|---|---|---|---|
| `seed` | `int` | `42` | Master seed.  Python / NumPy / torch globals reseeded; DP RNG chain folds from this. |
| `data_seed` | `int \| None` | `None` | Sampler seed.  Falls back to `seed` when unset. |
| `full_determinism` | `bool` | `False` | Enable deterministic algorithms (slower). |

## Token counting

| Field | Type | Default | Purpose |
|---|---|---|---|
| `include_tokens_per_second` | `bool` | `False` | Emit `train_tokens_per_second` in end-of-train metrics. |
| `include_num_input_tokens_seen` | `bool \| str` | `False` | One of `{"no", "all", "non_padding"}` (bool → string).  `"non_padding"` uses `attention_mask` or `pad_token_id`; costs one `.sum().item()` host sync per step. |

## Misc

| Field | Type | Default | Purpose |
|---|---|---|---|
| `output_dir` | `str \| None` | `None` | Defaults to `"trainer_output"` when unset. |
| `overwrite_output_dir` | `bool` | `False` | If `False`, the trainer warns when `output_dir` already contains checkpoints. |
| `resume_from_checkpoint` | `str \| None` | `None` | Path to a checkpoint directory.  `True` (passed to `train()`) auto-finds the latest. |
| `skip_memory_metrics` | `bool` | `True` | Skip the HF-borrowed `TrainerMemoryTracker` snapshots. |
| `cpu_offload_activations` | `bool` | `False` | Offload activations to CPU between forward and backward. |
| `debug` | `list \| str \| None` | `""` | HF debug flags.  Supports `"underflow_overflow"` (installs `DebugUnderflowOverflow`). |

## Validation

`__post_init__` runs cross-field validation idempotently:

- Strategy strings validated against allowed sets (`{"no", "steps", "epoch"}` for log/eval, `{"no", "steps", "epoch", "best"}` for save).
- `bf16` + `fp16` mutually exclusive; same for `warmup_ratio` + `warmup_steps`.
- `save_strategy="best"` requires `eval_strategy != "no"` (so a best-metric can be picked).
- `load_best_model_at_end=True` requires both `save_strategy != "no"` and `eval_strategy != "no"`.
- `save_steps > 0` enforced when `save_strategy != "no"`.
- `torch_compile_mode` checked against the allowed set.
- `report_to="all"` expands via HF's `get_available_reporting_integrations()`.

Errors raise `ValueError` at construction (loud failure, not silent
coercion).
