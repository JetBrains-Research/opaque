# DPTrainer API

`DPTrainer` mirrors HuggingFace's `Trainer` interface: construct with
`(model, args, datasets, …)`, call `train()` / `evaluate()` /
`predict()`, query the result.  Under the surface the loop runs DP-SGD
with `vmap(grad(...))` over per-example losses, but the public API is
designed so a user familiar with `Trainer` doesn't have to think about
that.

This page covers what's exposed.  For configuration, see
[TrainingArguments](training-arguments.md).  For domain-specific
loss overrides (SFT / DPO / KTO), see
[Subclassing](subclassing.md).

## Constructor

```python
from opaque.transformers import DPTrainer, TrainingArguments

trainer = DPTrainer(
    model=model,
    args=TrainingArguments(...),
    data_collator=None,                 # default: padding collator if processing_class given
    train_dataset=train_ds,
    eval_dataset=eval_ds,               # required if eval_strategy != "no"
    processing_class=tokenizer,
    compute_loss_func=None,             # see subclassing.md
    compute_metrics=None,                # callable(EvalPrediction) -> dict
    callbacks=None,                      # list of TrainerCallback
    optimizers=(None, None),             # *not supported* — DPTrainer owns the optimizer
    optimizer_cls_and_kwargs=None,       # functional torchopt factory override
    preprocess_logits_for_metrics=None,  # vmap-batched
)
```

Constructor argument semantics:

| Argument | Type | Purpose |
|---|---|---|
| `model` | `PreTrainedModel` | Required.  Moved to `args.device` immediately; PEFT wrappers are detected and cached on `self._is_peft`. |
| `args` | `TrainingArguments` | Required (defaults are filled in if omitted but you want explicit privacy targets). |
| `data_collator` | `Callable \| None` | Defaults to `DataCollatorWithPadding(tokenizer)` when `processing_class` is a tokenizer, else `default_data_collator`. |
| `train_dataset` | `Dataset \| None` | May be `None` only when `eval_strategy="no"` and `train()` is not called. |
| `eval_dataset` | `Dataset \| None` | Required when `eval_strategy != "no"`. |
| `processing_class` | `PreTrainedTokenizerBase \| SequenceFeatureExtractor \| None` | Used for the default collator selection and token-count metrics. |
| `compute_loss_func` | `Callable[[outputs, labels], Tensor] \| None` | Per-example loss override; called *under vmap* — sees one example's `outputs` and `labels`, returns a scalar.  See [Subclassing — compute_loss_func](subclassing.md#callable-bypass-compute_loss_func). |
| `compute_metrics` | `Callable[[EvalPrediction], dict] \| None` | User-supplied evaluation metrics callback.  Called once per eval pass with concatenated predictions / label_ids / inputs / losses. |
| `callbacks` | `list[TrainerCallback] \| None` | User callbacks; `DefaultFlowCallback` is auto-prepended. |
| `optimizers` | `tuple[Any \| None, Any \| None]` | **Not supported.** Passing non-`None` raises `RuntimeError`: DPTrainer owns the functional torchopt optimizer; the standard `(torch.optim, lr_scheduler)` pair has no place in the DP-SGD path. |
| `optimizer_cls_and_kwargs` | `tuple[Callable, dict] \| None` | DPTrainer-specific.  Override the optimizer with a functional torchopt-style factory.  Validated against the functional contract at construction time. |
| `preprocess_logits_for_metrics` | `Callable \| None` | Vmap-batched.  Lets `compute_metrics` consume a reduced representation of logits (e.g. argmax) instead of the raw tensor. |

The constructor also seeds Python / NumPy / torch global RNGs from
`args.seed` so non-DP randomness (head init, dataset shuffling without
explicit seed, user `compute_metrics` calling `torch.randn`) is
reproducible run-to-run.

## Training

### `train(resume_from_checkpoint=None, ignore_keys_for_eval=None) -> TrainOutput`

Runs the full DP-SGD loop.  Returns a named tuple
`TrainOutput(global_step, training_loss, metrics)`.

```python
out = trainer.train()
print(out.global_step, out.training_loss)
print(out.metrics["privacy_epsilon"])
```

`resume_from_checkpoint` accepts:

- `None` or `False` — fresh run.
- A path string — resume from that directory.
- `True` — auto-find the latest `checkpoint-*/` under `output_dir`.
  This is a convenience over stock HF; if no checkpoint exists, the
  trainer logs a warning and starts fresh.  See
  [Checkpoint and resume](checkpointing.md) for the full contract.

`ignore_keys_for_eval` is forwarded to every training-loop eval pass
(`evaluate()` calls fired by `eval_strategy="steps"` /
`eval_strategy="epoch"`).

End-of-train metrics always include:

- `train_loss` (running mean across the run)
- `train_steps`
- `train_runtime`, `train_samples_per_second`, `train_steps_per_second`
- `privacy_epsilon`, `privacy_delta`, `privacy_noise_multiplier`
- `train_fp16_overflow_steps` (only when the fp16 loss scaler was
  active for the run)
- `num_input_tokens_seen` (only when `include_num_input_tokens_seen != "no"`)

### `evaluate(eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> dict[str, float]`

Runs one eval pass and returns the metrics dict.  Side effects:

- Installs the cached-accountant barrier on the active training
  context (so subsequent ε queries reuse the privacy-loss
  distribution up to this point — see the
  [`opaque.accounting.cached`](../accounting.md) docs).
- Appends the metrics row to `state.log_history` via `log()`.
- Fires `on_evaluate` callback.
- Updates `state.best_metric` / `state.best_global_step` when
  configured.
- Feeds the metric into a metric-driven LR schedule (`ReduceLROnPlateau`).

Direct user calls behave identically to the eval calls the training
loop makes.

`evaluate` does **not** accept a `dict[str, Dataset]`; multi-task
eval was dropped — caller loops themselves if they want per-task
metrics.

### `predict(test_dataset, ignore_keys=None, metric_key_prefix="test") -> EvaluationResult`

Like `evaluate` but returns the full result (predictions + labels +
metrics + num_samples) instead of just the metrics dict.  Fires
`on_predict` callback at the end.

## `EvaluationResult`

```python
from opaque.transformers import EvaluationResult

@dataclass
class EvaluationResult:
    predictions: Any | None
    label_ids: Any | None
    metrics: dict[str, float]
    num_samples: int
```

This is the unified return type for `evaluation_loop`, `evaluate`,
and `predict` — replaces HF's split `EvalLoopOutput` /
`PredictionOutput` pair.  `predictions` and `label_ids` are `None`
when `prediction_loss_only=True` (the loop never materialises
prediction tensors).  Otherwise they're numpy arrays after the
opaque-distributed gather + truncation to the dataset's true sample
count.

## Optional `EvalPrediction` fields

By default `compute_metrics` receives only `predictions` and
`label_ids`.  Opt into the optional fields via
`args.include_for_metrics`:

- `"inputs"` — populates `EvalPrediction.inputs` with the model's
  primary input column (sniffed from `model.main_input_name`;
  `"input_ids"` for text, `"pixel_values"` for vision).
- `"loss"` — populates `EvalPrediction.losses` with **real
  per-example losses** computed via the vmap'd eval closure
  (`prediction_step` switches to a per-example forward when this is
  requested).

```python
args = TrainingArguments(..., include_for_metrics=["inputs", "loss"])
```

## Logging and persistence

| Method | Purpose |
|---|---|
| `log(logs, start_time=None)` | Append a row to `state.log_history`, fire `on_log` callback. |
| `log_metrics(split, metrics)` | Pretty-print metrics for a split (`"train"`, `"eval"`, `"test"`). |
| `save_metrics(split, metrics, combined=True)` | Write `metrics.json` / `all_results.json` under `output_dir`. |
| `save_state()` | Write `trainer_state.json`. |
| `save_model(output_dir=None, _internal_call=False)` | Write model weights + tokenizer + `training_args.bin` + the DP runtime state bundle (see [Checkpoint and resume](checkpointing.md)). |

## Distributed and process-zero checks

```python
trainer.is_world_process_zero()  # True on the global rank-0 process
trainer.is_local_process_zero()  # True on each node's rank-0 process
```

`DPTrainerState` carries the same flags (`state.is_world_process_zero`,
`state.is_local_process_zero`) so callbacks can rank-gate side
effects (e.g. W&B / TB only writing from rank 0).

## Callback wiring

`DPTrainer` auto-registers:

- `DefaultFlowCallback` (HF standard) — drives the
  `should_log` / `should_evaluate` / `should_save` flow flags on
  `TrainerControl` from the `*_strategy` args.
- `BestModelSaveCallback` — auto-injected when
  `save_strategy="best"`.  Tracks the best eval metric (compared via
  `is_metric_improved` against `state.best_metric`) and writes the
  checkpoint only on improvement.
- A `ProgressCallback` / `PrinterCallback` pair (per the
  `disable_tqdm` arg).
- Reporting callbacks (W&B, TensorBoard, MLflow, …) selected by
  `args.report_to`.  These are wrapped with
  `wrap_reporting_callback_class` so privacy-metric keys
  (`privacy_epsilon`, `privacy_clip_rate`, …) are rewritten to
  hierarchical paths (`privacy/epsilon`, `privacy/clip_rate`) for
  backends that support nested-metric trees.

User callbacks can be added at any time:

```python
trainer.add_callback(MyCallback())
trainer.add_callback(MyCallback)   # class, instantiated inside the handler
trainer.remove_callback(MyCallback)
popped = trainer.pop_callback(MyCallback)
```

`add_callback` also appends to `self._base_callbacks` so the callback
survives a `_reset_state_for_batch_size_retry` (the trainer rebuilds
the handler from `_base_callbacks` after an OOM retry).

Note: `on_substep_end` is **not fired**.  Each Poisson round under
DP-SGD is one atomic clip-noise-step; there's no substep concept.

## Overridable hooks for subclasses

The methods most likely to be overridden in domain-specific subclasses
(SFT / DPO / KTO):

| Method | Override scope |
|---|---|
| [`compute_per_example_loss(fmodel, params, inputs, *, return_logits=False)`](subclassing.md) | **The primary DP-correct extension point.**  See [Subclassing](subclassing.md). |
| `prediction_step(model, inputs, prediction_loss_only, ignore_keys=None)` | Override only if the default ModelOutput-shaped eval is wrong for the model. |
| `evaluation_loop(dataloader, *, description, prediction_loss_only, ignore_keys, metric_key_prefix)` | Override only if the full eval loop needs custom orchestration. |
| `create_optimizer()` | Override only to swap the functional torchopt optimizer; prefer `optimizer_cls_and_kwargs` constructor arg. |
| `create_scheduler(num_training_steps)` | Override only for LR schedules outside the built-in factory. |
| `get_train_dataloader()` | Override only to swap the sampler family (e.g. dp-ftrl variants).  DP-correctness requires the sampler produces each example with independent probability `ctx.sample_rate`. |
| `get_eval_dataloader(eval_dataset=None)` | Standard `DataLoader` — usually no override needed. |

## End-to-end example

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from opaque.transformers import DPTrainer, TrainingArguments

# Model + tokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:1%]")
def tokenize(batch):
    enc = tok(batch["text"], truncation=True, padding="max_length", max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
ds = ds.map(tokenize, batched=True, remove_columns=["text"])
ds.set_format("torch")

# Args — note the privacy fields are DPTrainer-specific
args = TrainingArguments(
    output_dir="gpt2-dp",
    per_device_train_batch_size=8,
    num_train_epochs=1.0,
    learning_rate=5e-4,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="no",
    privacy_target_epsilon=8.0,
    privacy_target_delta=1e-5,
    clipping_norm=1.0,
)

trainer = DPTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
out = trainer.train()
print(
    f"trained {out.global_step} steps; "
    f"loss={out.training_loss:.4f}; "
    f"ε={out.metrics['privacy_epsilon']:.2f}"
)
```

## See also

- [TrainingArguments](training-arguments.md) — every config field.
- [Subclassing](subclassing.md) — the `compute_per_example_loss`
  override hook.
- [Checkpoint and resume](checkpointing.md) — what `save_model` writes
  and how `train(resume_from_checkpoint=...)` reads it.
- [Distributed Training](../distributed-trainer.md) — DDP specifics
  (per-rank sharding, accountant cluster-wide composition).
