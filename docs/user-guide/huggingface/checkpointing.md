# Checkpoint and Resume

DPTrainer persists every piece of state needed to resume a run with no
loss of privacy provenance.  A resumed run continues the same DP-SGD
process — the saved accountant becomes the *prefix*, noise calibration
targets the original ε against that prefix, and the Poisson sampler
picks up at the recorded cursor.

This page covers the on-disk layout, the typed runtime bundle, the
resume contract, and the failure modes you'll hit when something is
inconsistent.

## Checkpoint directory layout

`save_strategy="steps"` / `"epoch"` / `"best"` write
`{output_dir}/checkpoint-{step}/` containing:

| File | Written by | Purpose |
|---|---|---|
| `model.safetensors` (or `pytorch_model.bin`) | `model.save_pretrained` | Model weights.  Safetensors when `save_safetensors=True` (default). |
| `config.json`, `generation_config.json`, … | `model.save_pretrained` | HF model config. |
| Tokenizer files | `processing_class.save_pretrained` | When a tokenizer / feature extractor is supplied. |
| `training_args.bin` | `_save_training_args` | Pickled `TrainingArguments`. |
| `trainer_state.json` | `_save_trainer_state` | `DPTrainerState` (global_step, epoch, log_history, best_metric, callback states, …). |
| `accountant.json` | `_save_accountant` | Privacy provenance — composed mechanisms.  Reads with `opaque.serialization.from_state_dict(Accountant(), …)`. |
| `dp_state.pt` | `_save_dp_runtime` | Typed `RuntimeCheckpoint` (clip + noise + sampler + scheduler state).  Skipped when `save_only_model=True`. |
| `dp_optimizer.pt` | `_save_optimizer` | Functional torchopt state.  Skipped when `save_only_model=True`. |
| `rng_state.pth` (single rank) / `rng_state_{rank}.pth` (DDP) | `_save_rng_state` | Python / NumPy / torch / CUDA / MPS RNG snapshot for the non-DP RNG.  Per-rank because each rank has its own state. |

The DP RNG chain (used by `gaussian_noise` and `PoissonSampler`) is
seeded from `args.seed` and folded by step count; it doesn't need a
saved snapshot.

## `RuntimeCheckpoint`

`dp_state.pt` deserialises to a typed dataclass — replaces the prior
ad-hoc `dict[str, Any]` so resume code is attribute-driven instead of
string-keyed:

```python
@dataclass
class RuntimeCheckpoint:
    version: int
    clip_state: dict[str, Any]          # opaque.serialization flat dict
    noise_state: dict[str, Any]         # opaque.serialization flat dict
    sampler_state: dict[str, Any] | None  # PoissonSampler state (consumed cursor + RNG)
    sample_rate: float                   # compare-on-resume
    target_delta: float                  # compare-on-resume
    noise_multiplier: float              # compare-on-resume
    expected_steps_per_epoch: int        # compare-on-resume
    expected_batch_size: int             # compare-on-resume
    total_steps: int                     # compare-on-resume
    lr_schedule_state: dict[str, Any] | None
```

Fields tagged `compare_on_resume=True` are checked against the live
`args` on resume — drift logs a warning (heterogeneous composition is
DP-valid, but a saved-vs-current mismatch is worth surfacing).  Adding
a new compare-on-resume field is a one-line edit on the dataclass; the
drift-check loop iterates `dataclasses.fields(...)`.

The bundle's `version` field is checked against
`DP_STATE_BUNDLE_VERSION`; resume fails with `ValueError` on an
unsupported version (no implicit migration).

## Resume contract

`train(resume_from_checkpoint=...)` accepts:

- `None` (or `False`) — fresh run.
- A path string — resume from that directory.
- `True` — auto-find the latest `checkpoint-*/` under `output_dir`
  (logs a warning and starts fresh if none exists).  Convenience over
  stock HF, which raises in that case.

Resume restores in this order:

1. **Model weights** via `_load_model_weights` (HF `from_pretrained`
   shape; sharded checkpoints handled).
2. **Runtime bundle** + **accountant** via `_read_runtime_for_resume`.
3. **Trainer state** via `_read_trainer_state` → reseats
   `self.state` from `trainer_state.json` and rebinds the callback
   handler's state pointer.
4. **Training context** built with `prefix_accountant=accountant` —
   calibration runs against the remaining steps.
5. **Clip / noise / optimizer state** merged into the live context
   via `opaque.serialization.from_state_dict`.
6. **LR schedule** state restored when the saved scheduler matches the
   current `lr_scheduler_type`.
7. **Sampler state** installed *before* the `DataLoader` binds — the
   restored `PoissonSampler` carries the `consumed` cursor so iteration
   resumes at the right step.
8. **RNG snapshot** restored from the per-rank file.
9. **Callback states** restored when
   `restore_callback_states_from_checkpoint=True`.

## Accountant prefix-and-recalibrate

The saved accountant is the *prefix*: composition continues from there,
not from zero.  Calibration then targets the original
`privacy_target_epsilon` against the remaining steps:

```
total_eps = compose(saved_accountant, remaining_run)
calibrate noise_multiplier so total_eps == privacy_target_epsilon
```

The saved noise multiplier doesn't constrain the new run: if the
remaining-step count, sample rate, or even mechanism family changed,
the calibration searches for a new σ that achieves the user's ε
target.  Heterogeneous composition is DP-valid; the
`_warn_on_arg_drift` log surfaces the change so it isn't silent.

### `privacy_resume_without_accountant`

There's exactly one legitimate scenario where the prefix is empty: a
**warmup on public data, then DP-fine-tune on private data** workflow,
where the saved checkpoint genuinely carries zero DP cost.  Opt in with
`privacy_resume_without_accountant=True`; the trainer installs an empty
`Accountant()` as the prefix.  Calibration then runs over the remaining
steps as if no prior training existed.

Without this flag, a missing `accountant.json` is a hard
`FileNotFoundError` — resuming without the saved provenance would
silently discard the spent privacy budget.

## Sampler resume

`PoissonSampler` carries a `consumed: int` cursor recording how many
batches have been drawn so far.  `state_dict(sampler)` serialises that
cursor plus the sampler's RNG state; `from_state_dict(template, sd)`
validates the template's `len(data_source)` against the saved
`dataset_size` and installs the cursor.

The trainer takes the *single sampler driving the run* and routes its
state through the `opaque.serialization` registry — no per-epoch
sampler reinstantiation, no fast-forward iteration.

When the dataset shape changes between save and resume (different
length, different ordering after a re-shuffle), `from_state_dict`
raises `ValueError`.  Override with `ignore_data_skip=True`:

```python
args = TrainingArguments(
    ...,
    ignore_data_skip=True,  # skip sampler-state restore, start from consumed=0
)
```

The new run starts each epoch from a fresh subsample sequence — still
DP-valid (a Poisson sequence is iid by construction; the saved state
just made it reproducible).

## Argument drift warnings

The compare-on-resume fields on `RuntimeCheckpoint` are checked
against the live `args` after the training context is built.  Drift
logs at `warning` level so it's visible by default:

```
Resume arg drift on sample_rate: saved=0.0625, current=0.125 — heterogeneous composition still gives a correct ε but the saved/current mechanisms differ
```

The composition is still DP-valid; the warning exists so a user who
changed `per_device_train_batch_size` (and therefore `sample_rate`)
without intending to doesn't miss it.

## Callback-state drift

When `restore_callback_states_from_checkpoint=True`, the trainer reads
`state.stateful_callbacks` (populated from `trainer_state.json`) and
copies saved attributes back onto the live callback instances.

Drift between the saved and live callback sets logs at `info` level
(not warning — adding or removing a callback between runs is a normal
workflow):

```
Resume callback drift: 1 saved callback(s) not present in live trainer (state will not be restored): ['EarlyStoppingCallback']
Resume callback drift: 1 live callback(s) not present in saved state (will start with fresh state): ['CustomScheduleCallback']
```

The HF `ExportableState` shape (`{"args": {...}, "attributes":
{...}}`) is used to restore attributes only — callback identity is
preserved (the same instance the user passed in stays attached).

## Saving without DP runtime — `save_only_model`

`save_only_model=True` skips `dp_state.pt`, `dp_optimizer.pt`, and the
per-rank RNG snapshot.  Useful for shipping a final model where the
user only wants the weights + tokenizer + config + `accountant.json`.

Resuming from a `save_only_model=True` checkpoint requires
`privacy_resume_without_accountant=True` *if and only if* the
checkpoint has no `accountant.json` either; the accountant is still
written by `save_only_model=True` if one exists in the live context.

## `save_model()` versus checkpoint dirs

`save_model(output_dir)` (called manually or via `save_state()`) writes
into `output_dir` directly, not into a `checkpoint-N/` subdir.  It
writes weights + tokenizer + `training_args.bin` + `accountant.json`.
It does **not** write the DP runtime bundle, the optimizer state, the
trainer state, or the RNG snapshot.

This is the "shipping model" path: the accountant travels with the
weights, but the run can't be resumed from this directory alone.  Use
`_save_checkpoint` (driven by `save_strategy`) when resume is
required.

## `load_best_model_at_end` failure modes

`load_best_model_at_end=True` restores the best-eval checkpoint into
the live model after `train()` finishes.  Two failure modes raise
`RuntimeError`:

1. **No best checkpoint recorded.**  Happens when eval never improved
   on `metric_for_best_model` — either the metric configuration is
   wrong, or the eval cadence is too coarse to ever produce a saved
   improving step.  Soft-failing here would leave the user with the
   last-trained weights silently masquerading as "best", which is
   exactly what the flag is supposed to prevent.
2. **Best checkpoint exists but has no weights file.**  The
   `checkpoint-N/` folder was rotated out or partially deleted between
   save and load.  Resolve by ensuring `save_total_limit` is high
   enough to keep the best checkpoint, or by setting
   `metric_for_best_model` to a metric that produces improving steps
   before rotation kicks in.

The `BestModelSaveCallback` (auto-injected when `save_strategy="best"`)
sets `state.best_global_step` whenever eval improves; the next save
boundary looks up the folder named `checkpoint-{best_global_step}` and
records it as `state.best_model_checkpoint` — so even when the
improving step falls into a non-saved bucket, a later save can still
register the right folder, provided `save_total_limit` hasn't already
rotated it out.

## Multi-rank checkpoints

Under DDP, rank-0 writes the shared artefacts (model weights, trainer
state, training args, accountant, optimizer, DP runtime).  Every rank
writes its own RNG snapshot to `rng_state_{rank}.pth` (per-rank because
each rank's non-DP RNG drifts independently — collator stochasticity,
model-init randomness, eval shuffling).  A barrier at the start (after
directory creation) and at the end (after rotation) keeps all ranks in
lockstep.

`save_on_each_node=True` switches to **every node's rank-0 process
writes a copy** (useful when the output directory lives on
node-local storage).

## Code path summary

```
train(resume_from_checkpoint=...)
  ├─ _resolve_resume_path  →  None | "/path/to/checkpoint-N"
  └─ if resume_path:
      ├─ _load_model_weights
      ├─ _read_runtime_for_resume  →  (RuntimeCheckpoint?, Accountant?)
      ├─ _read_trainer_state       →  DPTrainerState
      └─ _setup_training(prefix_accountant=...)
           └─ _calibrate_noise(prefix_accountant=...)
      then
      ├─ _apply_runtime_state(ctx, runtime, accountant, ckpt_dir)
      │   ├─ ctx.clip_state ← from_state_dict
      │   ├─ ctx.noise_state ← from_state_dict
      │   ├─ ctx.opt_state ← from_state_dict
      │   └─ ctx.lr_schedule.load_state_dict
      ├─ _warn_on_arg_drift(runtime)
      ├─ _load_rng_state(ckpt_dir)
      ├─ _load_callback_states()
      └─ _inner_training_loop(saved_sampler_state=...)
           └─ DataLoader binds restored PoissonSampler
```

## See also

- [DPTrainer API](dptrainer.md) — `save_model()`, `save_state()`,
  `train(resume_from_checkpoint=...)`.
- [TrainingArguments](training-arguments.md) — `save_strategy`,
  `save_total_limit`, `save_only_model`,
  `privacy_resume_without_accountant`, `ignore_data_skip`,
  `restore_callback_states_from_checkpoint`.
- [Troubleshooting](troubleshooting.md) — common resume failures.
- [Privacy accounting](../accounting.md) — `Accountant` composition
  semantics.
