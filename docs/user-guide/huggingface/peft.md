# PEFT and LoRA

This page covers Opaque's support for parameter-efficient fine-tuning
(PEFT) — primarily LoRA — driven directly through the
`make_functional` / `clipped_grad` API.  The recipe doesn't require
DPTrainer; it's the same primitives the trainer uses internally, and
the page closes with a note on what changes when the trainer drives
the run.

## Why PEFT for DP

Per-example gradient computation via `vmap(grad(...))` requires memory
proportional to `batch_size * trainable_parameters`.  For full
fine-tuning of a 7B model this is prohibitive; with LoRA the
trainable surface drops to ~0.1% of the model and per-example
gradient memory drops accordingly.  In practice, PEFT — LoRA
specifically — is a precondition for DP fine-tuning at LLM scale.

## `make_functional`

PyTorch modules store parameters internally.  To use them with
`vmap(grad(...))` or `clipped_grad`, convert to functional form:

```python
from opaque.functional import make_functional

model = AutoModelForCausalLM.from_pretrained("gpt2")
fmodel, params = make_functional(model)

def loss_fn(params, input_ids, labels):
    out = fmodel(params, input_ids=input_ids, labels=labels)
    return out.loss
```

Returns:

- `fmodel` — a callable that takes parameters as the first argument
  followed by the model's normal arguments.
- `params` — the model's parameters as a flat dict keyed by parameter
  name.

### Trainable / frozen partition

For PEFT (LoRA, adapters, BitFit, …) you want to clip and noise
*only* the trainable subset.  `partition_trainable=True` splits
parameters by their `requires_grad` attribute:

```python
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def loss_fn(trainable_params, input_ids, labels):
    out = fmodel(
        {**frozen, **trainable_params},
        input_ids=input_ids,
        labels=labels,
    )
    return out.loss
```

Only `trainable_params` receives per-example gradients.  Frozen
parameters are treated as constants by `vmap` — broadcast, not
replicated per example — so per-example gradient memory scales with
the trainable subset alone.

## LoRA recipe

LoRA (Hu et al. 2021) replaces a frozen linear layer `W (d_in × d_out)`
with `W + A @ B * (alpha / r)` where `A (d_in × r)` and `B (r × d_out)`
are small trainable matrices.  Only `A` and `B` are trained; the base
`W` stays frozen.  Trainable parameter count drops from `d_in × d_out`
to `r × (d_in + d_out)`.

End-to-end recipe with `clipped_grad` and `gaussian_noise`:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.functional import make_functional
from opaque.random import key

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = get_peft_model(
    model,
    LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none"),
)

# Functional form: only LoRA params are trainable.
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def loss_fn(trainable_params, input_ids, labels):
    out = fmodel(
        {**frozen, **trainable_params},
        input_ids=input_ids.unsqueeze(0),
        labels=labels.unsqueeze(0),
    )
    return out.loss

# DP components.  `normalize_by` is the cluster-wide logical batch (Poisson E[B]).
batch_size = 64
lr = 3e-4
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

The `frozen` dict is captured once at `make_functional` time and
remains a constant in the loss closure for the whole run.

### Other PEFT methods

Any parameter-efficient method that uses standard `requires_grad`
flags works with `make_functional(partition_trainable=True)`:

| Method | Library | Notes |
|---|---|---|
| **LoRA** | `peft` | Recommended.  Well-tested with Opaque; auto LoRA-fusion (`opaque_lora_qkv` / `opaque_lora_mlp`) when adapters have no bias. |
| **Adapters** (bottleneck) | `peft` / `adapter-transformers` | Works.  More trainable params than LoRA at same capacity. |
| **BitFit** (bias-only) | Manual (`requires_grad=False` on non-bias) | Minimal trainable params; very memory-efficient. |
| **Prefix tuning** | `peft` | Works but virtual tokens add complexity to loss computation. |
| **IA3** | `peft` | Very few trainable params.  Good for memory-constrained settings. |

The key requirement is that trainable parameters are identified by
`requires_grad=True`.  If a PEFT method uses custom forward hooks
instead of standard parameters, it may not work with `vmap`.

### LoRA hyperparameters

| Parameter | Typical values | Effect on DP training |
|---|---|---|
| `r` (rank) | 4, 8, 16 | Higher → more trainable params → more memory for vmap. |
| `lora_alpha` | 16, 32 | Scaling factor; does not affect memory. |
| `target_modules` | `["q_proj", "v_proj"]` | More modules → more trainable params. |
| `bias` | `"none"` | Required for fused LoRA kernels (`opaque_lora_qkv` / `opaque_lora_mlp`). |

Start with `r=8`, `target_modules=["q_proj", "v_proj"]`, `bias="none"`.
Increase rank or add modules only if accuracy is insufficient.

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

## DPTrainer integration

`DPTrainer` hands a PEFT-wrapped model the same treatment:

- Construction detects the `PeftModel` wrapper and caches the result
  on `self._is_peft`.
- `_setup_training` calls `make_functional(model,
  partition_trainable=True)` internally — you don't construct the
  functional form yourself.
- Clipping and noising target the trainable LoRA adapters only;
  frozen base parameters are broadcast.
- `apply_model_patches(peft=True)` (default) enables the fused LoRA
  kernels when the adapter layout makes them applicable.

The trainer flow is identical to the recipe above, just packaged:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from opaque.transformers import DPTrainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = get_peft_model(
    model,
    LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none"),
)

args = TrainingArguments(
    output_dir="llama-dp-lora",
    per_device_train_batch_size=4,
    privacy_target_epsilon=8.0,
    privacy_target_delta=1e-5,
    clipping_norm=1.0,
    use_performance_kernels=True,
)
trainer = DPTrainer(model=model, args=args, train_dataset=train_ds)
trainer.train()
```

## See also

- [Model patches](model-patches.md) — fused LoRA kernels and the
  per-model compatibility matrix.
- [Memory Optimizations](../memory-optimizations.md) — gradient
  checkpointing, microbatching.
- [Per-example gradient clipping](../clipping.md) —
  `clipped_grad`, `auto_clipped_grad`, `adaptive_clipped_grad`.
- [Utilities reference](../../reference/utilities.md) —
  `make_functional` signatures.
- [DPTrainer](dptrainer.md) — when the trainer drives the LoRA recipe
  for you.
