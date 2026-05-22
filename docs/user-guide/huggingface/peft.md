# PEFT and LoRA

LoRA is a practical necessity for DP training of large models.
Per-example gradient computation via `vmap` requires memory
proportional to `batch_size * trainable_parameters`.  With full
fine-tuning of a 7B model, this is prohibitive.  LoRA reduces
trainable parameters to ~0.1% of the model, making per-example
gradients feasible.

DPTrainer auto-detects PEFT wrappers and handles them transparently:
the functional model conversion partitions parameters by
`requires_grad`, and the trainer's `_restore_params` strict-validates
the trainable-key set on resume.

This page covers the primitives (the `make_functional` API), the
end-to-end LoRA recipe, and how other PEFT methods integrate.

## The `make_functional` primer

PyTorch models store parameters internally.  To use them with
`vmap(grad(...))` (or `clipped_grad` directly), convert to functional
form:

```python
from opaque.functional import make_functional

model = AutoModelForCausalLM.from_pretrained("gpt2")
fmodel, params = make_functional(model)

def loss_fn(params, input_ids, labels):
    out = fmodel(params, input_ids=input_ids, labels=labels)
    return out.loss
```

`make_functional(model)` returns:

- `fmodel` — a callable that takes parameters as the first argument
  followed by the model's normal arguments.
- `params` — the model's parameters as a flat dict keyed by parameter
  name.

### Separating trainable and frozen parameters

For PEFT (LoRA, adapters, BitFit, …) you want to clip and noise *only*
the trainable subset.  `partition_trainable=True` splits parameters by
their `requires_grad` attribute:

```python
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def loss_fn(trainable_params, input_ids, labels):
    all_params = {**frozen, **trainable_params}
    out = fmodel(all_params, input_ids=input_ids, labels=labels)
    return out.loss
```

Only `trainable_params` receives per-example gradients.  Frozen
parameters are treated as constants by `vmap`, which drastically
reduces memory since per-example gradients are only computed for the
trainable subset.

DPTrainer does this partitioning automatically — you don't call
`make_functional` yourself when using the trainer.  The trainer's
internal training context (`_TrainingContext`) carries
`trainable_params` and `frozen_params` separately so the vmap'd loss
closure can `{**frozen, **trainable}` for each forward pass.

## LoRA with DPTrainer

The DPTrainer recipe is straightforward — load the model, wrap with
LoRA, hand to the trainer:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from opaque.transformers import DPTrainer, TrainingArguments

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)

args = TrainingArguments(
    output_dir="llama-dp-lora",
    per_device_train_batch_size=4,
    num_train_epochs=1.0,
    learning_rate=3e-4,
    privacy_target_epsilon=8.0,
    privacy_target_delta=1e-5,
    clipping_norm=1.0,
    use_performance_kernels=True,            # CUDA + Triton
)

trainer = DPTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    processing_class=tok,
)
trainer.train()
```

The trainer detects the `PeftModel` wrapper at construction time
(`self._is_peft = True`), partitions parameters via
`make_functional(partition_trainable=True)` inside `_setup_training`,
and clips / noises only the LoRA adapters.

## LoRA without DPTrainer

Direct usage with `clipped_grad`, no trainer:

```python
from peft import LoraConfig, get_peft_model
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.functional import make_functional
from opaque.random import key

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))

# Functional form: only LoRA params are trainable.
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def loss_fn(trainable_params, input_ids, labels):
    out = fmodel(
        {**frozen, **trainable_params},
        input_ids=input_ids.unsqueeze(0),
        labels=labels.unsqueeze(0),
    )
    return out.loss

# DP components.
grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2),
    clipping_norm=1.0, normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))

# Training loop.
for input_ids, labels in dataloader:
    grads, clip_state = grad_fn(trainable, input_ids, labels, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    trainable = {k: trainable[k] - lr * noisy_grads[k] for k in trainable}
```

## Other PEFT methods

Any parameter-efficient method that uses standard `requires_grad`
flags works with `make_functional(partition_trainable=True)`:

| Method | Library | Notes |
|---|---|---|
| **LoRA** | `peft` | Recommended.  Well-tested with Opaque; auto LoRA-fusion when adapters have no bias. |
| **Adapters** (bottleneck) | `peft` / `adapter-transformers` | Works.  More trainable params than LoRA at same capacity. |
| **BitFit** (bias-only) | Manual (`requires_grad=False` on non-bias) | Minimal trainable params; very memory-efficient. |
| **Prefix tuning** | `peft` | Works but virtual tokens add complexity to loss computation. |
| **IA3** | `peft` | Very few trainable params.  Good for memory-constrained settings. |

The key requirement is that trainable parameters are identified by
`requires_grad=True`.  If a PEFT method uses custom forward hooks
instead of standard parameters, it may not work with `vmap`.

## LoRA hyperparameters

| Parameter | Typical values | Effect on DP training |
|---|---|---|
| `r` (rank) | 4, 8, 16 | Higher → more trainable params → more memory for vmap. |
| `lora_alpha` | 16, 32 | Scaling factor; does not affect memory. |
| `target_modules` | `["q_proj", "v_proj"]` | More modules → more trainable params. |
| `bias` | `"none"` | Required for fused LoRA kernels (`Opaque_LoRA_QKV` / `Opaque_LoRA_MLP`). |

Start with `r=8`, `target_modules=["q_proj", "v_proj"]`, `bias="none"`.
Increase rank or add modules only if accuracy is insufficient.

## Resume validation

When resuming a PEFT-wrapped run, `DPTrainer._restore_params` validates
that the keys in the saved `trainable_params` snapshot match the
current model's `requires_grad=True` set.  Mismatch raises
`RuntimeError`.

This catches two failure modes:

- **Typo'd parameter names** in subclass overrides — would otherwise
  leave the parameter at its initial value.
- **Mid-run `requires_grad` churn** — a callback freezing /
  unfreezing layers between snapshot and resume; the snapshot no
  longer matches the model.

If the trainable set legitimately changed between runs (e.g. user
added a new LoRA module), build a fresh checkpoint rather than
resuming from the stale one.

## Memory profile

Approximate memory for LoRA fine-tuning at common scales (one rank,
fp16, sequence length 1024):

| Model | LoRA params | DP-SGD peak | Full fine-tune peak |
|---|---|---|---|
| Llama-3.1-8B (`r=8`, q+v) | ~7M | ~22 GB | OOM on 80 GB |
| Qwen2-7B (`r=8`, q+v) | ~6M | ~20 GB | OOM on 80 GB |
| Gemma-2B (`r=8`, q+v) | ~3M | ~10 GB | ~50 GB |

LoRA's per-example gradient memory scales with trainable params, not
total params.  Full fine-tuning of even small models becomes
prohibitive once you account for `batch_size × trainable_params`
gradient tensors.

## See also

- [Model patches](model-patches.md) — fused LoRA kernels, model
  compatibility matrix.
- [Memory Optimizations](../memory-optimizations.md) — gradient
  checkpointing, microbatching.
- [Per-example gradient clipping](../clipping.md) —
  `clipped_grad`, `auto_clipped_grad`, `adaptive_clipped_grad`.
- [Utilities reference](../../reference/utilities.md) —
  `make_functional` signatures.
