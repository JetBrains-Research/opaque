# HuggingFace Integration

Opaque ships a HuggingFace-shaped trainer (`opaque.transformers.DPTrainer`)
plus a per-model patch surface (`opaque.patches`) so existing
`transformers` models can be trained under DP-SGD with the familiar
`Trainer.train()` / `evaluate()` / `predict()` interface.

This section is the routing guide.  Pick the page that matches what
you're trying to do.

## What's covered

- **[DPTrainer](dptrainer.md)** — `train()`, `evaluate()`, `predict()`,
  `EvaluationResult`, the callback wiring, and the log / save surface.
  Start here if you have a model and a dataset and want to train.
- **[TrainingArguments](training-arguments.md)** — every field on the
  dataclass, grouped by concern (privacy, compute, sampling, output,
  schedule, distributed, …).  DPTrainer-specific knobs are flagged
  inline so you know which fields don't exist on stock HF Trainer.
- **[Model patching](model-patches.md)** — what `apply_runtime_patches`
  and `apply_model_patches` do, why vmap needs them, the model
  compatibility matrix, the Triton kernel surface, and the umbrella
  flags that control them.
- **[PEFT and LoRA](peft.md)** — `make_functional` primer, full LoRA
  example end-to-end, and how other PEFT methods slot in.
- **[Subclassing DPTrainer](subclassing.md)** — `compute_per_example_loss`
  is the single override hook for SFT, DPO, KTO, and other
  domain-specific trainers.  The trainer composes vmap + grad + clip
  around it; subclasses just compute one example's loss.
- **[Checkpoint and resume](checkpointing.md)** — checkpoint layout,
  the typed `RuntimeCheckpoint` bundle, sampler-state restore,
  accountant prefix-and-recalibrate, and the failure modes around
  `load_best_model_at_end`.
- **[Troubleshooting](troubleshooting.md)** — known failure modes,
  unsupported HF Trainer features (HPO, hub publishing, multi-task
  eval), and stale-API symptoms.

## Which page do I start on?

```
Have a model + dataset, want to train?
│
├─ Yes ── start with dptrainer.md
│         then training-arguments.md for tuning
│
└─ No ─── Building a custom loss / SFT / DPO / KTO?
          ├─ Yes ── subclassing.md
          │
          └─ No ─── Something else
                    ├─ Model not on compatibility matrix → model-patches.md
                    ├─ LoRA / adapters → peft.md
                    ├─ Resuming a run → checkpointing.md
                    └─ Hit an error → troubleshooting.md
```

## Quick start

A minimal DPTrainer run looks like a HuggingFace `Trainer` run with a
few DP-specific fields on `TrainingArguments`:

```python
from transformers import AutoModelForCausalLM
from opaque.transformers import DPTrainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")
args = TrainingArguments(
    output_dir="run-0",
    per_device_train_batch_size=8,
    num_train_epochs=3.0,
    # DPTrainer-specific privacy knobs:
    privacy_target_epsilon=8.0,
    privacy_target_delta=1e-5,
    clipping_norm=1.0,
)

trainer = DPTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)
trainer.train()
metrics = trainer.evaluate()
```

The trainer auto-patches the model on construction (vmap-safety,
optional Triton kernels) and calibrates the noise multiplier from
`privacy_target_epsilon`.  Every detail lives in the topic pages
linked above.

## Scope

The HF integration prioritises **decoder-only text models**
(`*ForCausalLM` and shared text modules) tested against
`transformers==4.57.1`.  Vision-language stacks (e.g.
`*ForConditionalGeneration`) are not in the default patch set; see
[Model patches — model compatibility](model-patches.md#model-compatibility)
for the curated matrix.
