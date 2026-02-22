# LoRA Fine-tuning with Differential Privacy

[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) is a parameter-efficient fine-tuning method
well suited for differential privacy.  This guide shows how to combine LoRA with DP-SGD using Opaque.

## Why LoRA + DP

| Challenge        | Full Fine-tuning              | LoRA Fine-tuning             |
|------------------|-------------------------------|------------------------------|
| **Parameters**   | All (~7B for Llama-2-7B)      | ~0.1% (adapters only)        |
| **Memory**       | Very high (per-example grads) | Low (small adapters)         |
| **DP Overhead**  | ~10x slower                   | ~2x slower                   |

LoRA trains only low-rank adapter matrices, so per-example gradient computation
via `torch.func.vmap` is much cheaper.

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise, make_functional
from opaque.random import key

# 1. Load model + LoRA
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# 2. Functional form — only LoRA params are trainable
fmodel, lora_params, frozen = make_functional(model, partition_trainable=True)

# 3. Per-example loss
def loss_fn(lora_params, input_ids, labels):
    out = fmodel(lora_params, frozen, input_ids=input_ids.unsqueeze(0),
                 labels=labels.unsqueeze(0))
    return out.loss

# 4. Calibrate noise
batch_size, dataset_size, num_steps = 8, 10_000, 3750
sample_rate = batch_size / dataset_size

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=5.0,
)
noise_multiplier = result.param

# 5. DP components
grad_fn, clip_state = clipped_grad(
    loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2),
)
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity(), key=key(42),
)

# 6. Training loop
from opaque.accounting.accountant import Accountant

step_proc = acc.poisson(acc.gaussian(noise_multiplier), sample_rate)
acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))

for input_ids, labels in dataloader:
    grads, clip_state = grad_fn(lora_params, input_ids, labels, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    lora_params = {k: lora_params[k] - 1e-4 * noisy_grads[k] for k in lora_params}

    acct = acct | step_proc
    if acct.budget_exceeded:
        break

print(f"Final ε = {acct.epsilon_at(1e-5):.2f}")
```

## `make_functional` with `partition_trainable`

When using LoRA, only adapters are trainable.  Pass `partition_trainable=True`
to separate trainable and frozen parameters:

```python
fmodel, trainable_dict, frozen_dict = make_functional(model, partition_trainable=True)

# Call the functional model with both dicts:
output = fmodel(trainable_dict, frozen_dict, input_ids=..., labels=...)
```

`clipped_grad` then differentiates with respect to `trainable_dict` only
(`argnums=0` by default).

## LoRA Hyperparameters

### Rank

| r  | Trainable params | Quality  |
|----|------------------|----------|
| 4  | Very few         | Adequate |
| 8  | Few              | Good     |
| 16 | Moderate         | Better   |

Start with `r=8`.  Increase to 16 if accuracy is insufficient.

### Target Modules

```python
# Minimal (fastest)
target_modules = ["q_proj"]

# Standard (good balance)
target_modules = ["q_proj", "v_proj"]

# Comprehensive
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

## Memory Optimization

1. **Gradient checkpointing**: `model.gradient_checkpointing_enable()`
2. **Microbatching**: `clipped_grad(..., microbatch_size=2)`
3. **Smaller rank**: `r=4` instead of `r=8`
4. **BF16**: Reduce memory further

## Best Practices

1. **Start small** — debug with GPT-2 (124M) before scaling.
2. **Calibrate noise** — use `acc.calibrate()`, don't guess.
3. **Track budget** — use `Accountant` and stop when exhausted.
4. **Use Poisson sampling** — for privacy amplification.

## Troubleshooting

**Out of memory**: Reduce rank, enable gradient checkpointing, use microbatching.

**Poor accuracy**: Increase rank, add more target modules, relax ε.

**Training too slow**: Fewer target modules, smaller rank.

## See Also

- [Fine-tuning an LLM](../tutorials/llm_finetuning.ipynb) -- LoRA + DP tutorial
- [Gradient Clipping](clipping.md) — Per-example gradient details
- [Sampling](sampling.md) — Privacy amplification
- [PEFT Documentation](https://huggingface.co/docs/peft) — LoRA reference
