# Troubleshooting

Common failure modes you'll hit and what to do about them, plus the
list of HF Trainer features DPTrainer intentionally does *not*
implement.

## Failure modes

### `RuntimeError: load_best_model_at_end=True but no best checkpoint was recorded`

Eval never improved on `metric_for_best_model`.  The flag is meant to
prevent silently substituting the last-trained weights when no improving
checkpoint exists, so it raises rather than soft-failing.

Resolve:

- Verify `metric_for_best_model` matches a key returned by your
  `compute_metrics` (the trainer auto-prefixes with `eval_` when
  missing — but the underlying key must exist).
- Verify `greater_is_better` matches the metric direction.
- Increase eval frequency (`eval_strategy="steps"` with a lower
  `eval_steps`) so improvements aren't lost between eval boundaries.

### `RuntimeError: best checkpoint recorded ... but no weights file was found`

The recorded best-checkpoint folder was rotated out or partially
deleted.  `save_total_limit` rotates older checkpoints; if the best
step is older than the rotation window and `BestModelSaveCallback`
hasn't yet protected it, the folder can disappear.

Resolve: raise `save_total_limit`, or remove it (it protects
`state.best_model_checkpoint` automatically when both are set).

### `RuntimeError: DPTrainer._restore_params: trainable_params keys do not match`

The saved `trainable_params` snapshot doesn't match the live model's
`requires_grad=True` set.  Two common causes:

- **Typo'd parameter name in a subclass override** — the snapshot
  contains the typo'd key; the live model doesn't.  Without strict
  validation, `load_state_dict(..., strict=False)` would silently leave
  that parameter at its initial value.
- **Mid-run `requires_grad` churn** — a callback froze / unfroze
  layers between snapshot and resume.  The snapshot reflects the
  pre-callback set; the live model reflects the post-callback set.

Resolve: drop the typo or stop the callback churn.  If the trainable
set legitimately changed between runs (e.g. you added a new LoRA
module), build a fresh checkpoint rather than resuming from the stale
one.

### `FileNotFoundError: Cannot resume from ... accountant.json is missing`

Resuming without the saved accountant would silently discard the spent
privacy budget.  The trainer raises rather than silently allowing this.

Resolve:

- Restore `accountant.json` from the source of truth.
- Or, **only** when the resumed checkpoint genuinely has zero prior DP
  cost (warmup on public data, then DP-fine-tune on private), set
  `privacy_resume_without_accountant=True`.

### `ValueError: unsupported dp_state bundle version N`

The checkpoint was written by an older trainer version whose
`RuntimeCheckpoint` schema differs from the current one.  No implicit
migration — restore from a known-good checkpoint of the current
version, or roll back the trainer to the version that wrote the
checkpoint.

### `ValueError: ... dataset_size mismatch ...` from sampler resume

`PoissonSampler.from_state_dict` validates the saved `dataset_size`
against `len(data_source)`.  Resume on a different-sized dataset will
trip this.

Resolve: pass `ignore_data_skip=True` to skip sampler-state restore.
The new run starts each epoch from `consumed=0` — still DP-valid
(Poisson sequences are iid by construction).

### Resume arg-drift warning

```
Resume arg drift on noise_multiplier: saved=1.1, current=0.95 ...
```

Heterogeneous composition is DP-valid; the warning fires when a
compare-on-resume field (`sample_rate`, `target_delta`,
`noise_multiplier`, `expected_steps_per_epoch`, `expected_batch_size`,
`total_steps`) differs between the saved checkpoint and the live
`args`.  Verify the change was intentional — most often it's a
`per_device_train_batch_size` change carrying through to
`sample_rate`, which is fine; an unintentional knob bump shows up
here.

### fp16 overflow inflates `train_fp16_overflow_steps`

`fp16=True` enables dynamic loss scaling.  When a step's per-example
gradients contain inf / NaN, the loss scaler downscales and skips the
optimiser step.  The skip is surfaced via `state.fp16_overflow_steps`
(present in end-of-train metrics as `train_fp16_overflow_steps`).

A non-zero counter is benign in low single digits — that's the loss
scaler doing its job.  Persistent overflow (every step) means the
scale is too high for the current grad norms; switch to `bf16=True`
when the GPU supports it (bf16 has fp32-range exponent and doesn't
need dynamic scaling).

### `RuntimeError: DPTrainer.compute_per_example_loss: model forward returned no 'loss' field`

The model's forward returned an output without a `"loss"` field, and
no `compute_loss_func` was supplied to bridge.  Three options:

- Pass `compute_loss_func=...` to the trainer constructor for a
  callable bypass (see
  [Subclassing — callable bypass](subclassing.md#callable-bypass-compute_loss_func)).
- Override `compute_per_example_loss` in a subclass.
- Pass `labels` in the input so HF's per-model `LOSS_MAPPING` dispatch
  triggers the model's built-in loss computation.

### vmap-related errors under custom models

```
NotImplementedError: ... not yet implemented the batching rule ...
```

A PyTorch op the model uses has no `vmap` batching rule.  Common
culprits:

- **Flash Attention 2** — uses `torch.nonzero` (dynamic shapes break
  vmap).  Set `attn_implementation="sdpa"` on `from_pretrained`.
- **Flex attention** — `HigherOrderOperator` has no vmap support
  (upstream PyTorch limitation).  Use SDPA.
- **Custom ops with no batching rule** — wrap with `with_batch_dim`
  (see [Model patches — Other models](model-patches.md#other-models)).

The SDPA backward warning
(`_scaled_dot_product_*_attention_backward ... not yet implemented`)
is benign: SDPA backward falls back to per-sample processing under
vmap.  Upstream PyTorch has a pending fix for proper batching rules.

### "No detectable family" info log

```
INFO opaque: no detectable family for model_type=<your_model_type>; …
```

The model isn't on the
[compatibility matrix](model-patches.md#model-compatibility).  Either:

- Use a supported model family (see the matrix).
- Disable the auto-detection log by setting `use_compat_patches=False`
  (only safe if your model is vmap-compatible without opaque's
  compat shims).
- Wrap the model forward with `with_batch_dim` and pass
  `use_compat_patches=False`.

### Sampler resume seems to skip data

This is expected: `PoissonSampler.consumed` advances on every batch
yielded.  A resumed run picks up at the saved cursor, so the early
batches you saw before resume aren't re-yielded.  If you want a fresh
sequence (e.g. you re-shuffled the dataset), pass
`ignore_data_skip=True`.

### `gradient_checkpointing` raises under vmap

Pair `gradient_checkpointing=True` with
`gradient_checkpointing_kwargs={"use_reentrant": False}`.  Reentrant
checkpointing reads / writes `requires_grad` flags in ways that don't
compose with vmap.  Opaque's runtime patches (`vmap_checkpointing`)
handle the non-reentrant path.

### LoRA fusion not applied

Auto-fusion (`opaque_lora_qkv` / `opaque_lora_mlp`) requires *all*
projections in a group (Q+K+V or gate+up+down) to have LoRA adapters
with `bias="none"`.  Check `LoraConfig.target_modules` covers all
projections in the target attention or MLP block, and that the
adapters have no bias.

For per-model LoRA-fusion eligibility, see
[Model patches — fused LoRA operations](model-patches.md#fused-lora-operations).

## Unsupported HF Trainer features

DPTrainer is API-shaped after HF `Trainer` but intentionally omits a
few features that don't compose with DP-SGD.  If you're migrating from
stock `Trainer`, these are the gaps to know.

### Hyperparameter search

`Trainer.hyperparameter_search()` doesn't exist on DPTrainer.  Running
a sweep over private data is itself an adaptive use of the dataset's
privacy budget; the silent composition that
`Trainer.hyperparameter_search` performs would obscure that.

Drive sweeps from your orchestration layer (Ray, Optuna, W&B) and
account for sweep-level composition explicitly: one fresh DPTrainer per
trial, read each trial's ε / δ from its `accountant.json`, compose them
at the sweep level.

### Hub publishing

`push_to_hub`, `hub_strategy`, `hub_model_id`, etc. are not on
`TrainingArguments`.  Keeping the upload outside the trainer ensures
the privacy accountant always travels with the artefact the user
publishes — `accountant.json` lives next to the weights in
`save_model(output_dir)`, and a `huggingface_hub.upload_folder` call
from your orchestration layer ships both.

### Multi-task evaluation via dict eval datasets

`evaluate(dict[str, Dataset])` and `predict(dict[str, Dataset])`
recursive dispatch is **not** supported.  Loop yourself:

```python
results = {name: trainer.evaluate(eval_dataset=ds) for name, ds in eval_sets.items()}
```

### `Trainer.compute_loss`

Replaced by `compute_per_example_loss`.  HF's `compute_loss` operates
on a full batch; the DP path needs per-example forwards to get
per-example gradients before clipping.  See
[Subclassing DPTrainer](subclassing.md).

### `compute_loss_func` signature

HF: `(outputs, labels, num_items_in_batch) -> scalar`, called once per
batch.  DPTrainer: `(outputs, labels) -> scalar`, called per example
under vmap.  There's no `num_items_in_batch` — each call sees one
example.  Batch-mean math happens at the clip / noise layer
(`normalize_by`).

### `DataParallel`

DDP only.  `DataParallel` isn't supported because the noise needs to
be added once after gathering per-rank clipped gradients; the
`DataParallel` forward-then-replicate shape doesn't compose with
that.  See [Distributed Training](../distributed-trainer.md).

### `logging_nan_inf_filter`

Removed.  fp16 overflow surfaces via `state.fp16_overflow_steps`
(present in end-of-train metrics) rather than by filtering loss values
from the log.  NaN reaching the post-step loss reflects a genuine
forward / loss-math divergence; propagating it is the right behaviour.

### `gradient_accumulation_steps`

Not supported.  Every DP-SGD step is one atomic clip-noise-step over
one Poisson sample.  HF accumulation would change the effective
batch — and therefore the sample rate — without telling the
accountant.  Use `auto_find_microbatch_size=True` for OOM-driven
microbatch chunking; this is privacy-neutral because it doesn't change
the logical batch.

### Per-batch `return_loss` kwarg

Dropped.  Was a HF-side shim for CLIP-style contrastive models whose
forward needs an explicit `return_loss=True` to compute the
contrastive loss.  CLIP-shape contrastive losses depend on
cross-batch interactions that can't compose with per-example DP-SGD.

## Stale-API symptoms

If you're upgrading a script written against an older trainer version:

- `prediction_step` returning `(loss, logits, labels)` and the eval
  loop concatenating them is unchanged.
- `EvalLoopOutput` and `PredictionOutput` are gone — both `evaluate`
  and `predict` return `EvaluationResult` (which carries the same
  fields under different names; see
  [DPTrainer — EvaluationResult](dptrainer.md#evaluationresult)).
- `OpaqueEpochPoissonBatchSampler` is gone — a single
  `PoissonSampler(n_steps=total_steps)` drives the whole run, with
  registry-based resume.

## See also

- [Checkpoint and resume](checkpointing.md) — resume contract,
  failure modes around `load_best_model_at_end` and accountant
  policy.
- [Subclassing DPTrainer](subclassing.md) — `compute_per_example_loss`
  and the vmap-safety constraints.
- [Model patches](model-patches.md) — `with_batch_dim` and the
  compatibility matrix.
- [Limitations](../../limitations.md) — broader scope (non-text
  modalities, sequence-length constraints, etc.).
