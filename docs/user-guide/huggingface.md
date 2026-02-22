# HuggingFace Compatibility

Opaque works with HuggingFace Transformers models out of the box. When you
`import opaque`, it automatically patches core Transformers functions to be
compatible with `torch.func.vmap`, which is required for per-example
gradient computation. No manual patching is needed.

## Auto-patching

On import, Opaque applies patches to the following HuggingFace Transformers
components:

- **Causal mask creation** (`transformers.masking_utils.create_causal_mask`)
  -- handles arbitrary batch dimensions under vmap.
- **Key-value head repetition** (`repeat_kv`) -- uses negative indexing to
  work with both batched (4D) and unbatched (3D) tensors.
- **Eager attention forward** -- replaces model-specific attention with
  vmap-compatible implementations that use dynamic shapes.

These patches are applied for LLaMA, Mistral, Qwen2, Phi, Phi-3, OLMo,
Gemma, and Gemma2 models. Models not in this list may still work if their
attention implementation follows the standard Transformers pattern.

To disable auto-patching (e.g., for debugging), set the environment
variable before importing:

```python
import os
os.environ["OPAQUE_NO_PATCH"] = "1"
import opaque
```

### Why patches are needed

`torch.func.vmap` removes the batch dimension: a function that normally
receives input of shape `(batch, seq_len, hidden)` instead receives
`(seq_len, hidden)` and vmap handles the batching externally.
HuggingFace models use hardcoded dimension indices (e.g., `x.shape[0]`
for batch size) and data-dependent control flow that break under vmap.

The patches replace these with:

- **Negative indexing** (`shape[-3]`, `transpose(-2, -1)`) which works
  regardless of how many leading dimensions exist.
- **Dynamic dimension detection** (`ndim == 2` for vmap vs `ndim >= 3`
  for normal) to adapt mask shapes.
- **Broadcasting** instead of explicit batch dimension manipulation.

### Attention implementation support

| Attention type | Supported | Notes |
|---------------|-----------|-------|
| `eager` | Yes | Explicitly patched and tested |
| `sdpa` | Yes | Uses patched `repeat_kv`; default in recent Transformers |
| `flash_attention_2` | No | Uses `torch.nonzero` (dynamic shapes incompatible with vmap) |
| `flex_attention` | No | Tensor metadata issues with vmap |

If your model defaults to Flash Attention 2, set `attn_implementation`
when loading:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="sdpa",  # or "eager"
)
```

## Functional model conversion

PyTorch models store parameters internally. To use them with
`clipped_grad`, convert to functional form with `make_functional`:

```python
from opaque import make_functional, clipped_grad

model = AutoModelForCausalLM.from_pretrained("gpt2")
fmodel, params = make_functional(model)

def loss_fn(params, input_ids, labels):
    out = fmodel(params, input_ids=input_ids.unsqueeze(0),
                 labels=labels.unsqueeze(0))
    return out.loss

grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2), l2_clip_norm=1.0,
)
```

`make_functional` returns:

- `fmodel` -- a callable that takes parameters as the first argument
  followed by the model's normal arguments.
- `params` -- the model's parameters as a flat tuple (or dict, depending
  on the implementation).

### Separating trainable and frozen parameters

For parameter-efficient fine-tuning (LoRA, adapters), use
`partition_trainable=True` to separate parameters by their
`requires_grad` attribute:

```python
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def loss_fn(trainable_params, input_ids, labels):
    out = fmodel(trainable_params, frozen,
                 input_ids=input_ids.unsqueeze(0),
                 labels=labels.unsqueeze(0))
    return out.loss

grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2), l2_clip_norm=1.0,
)
```

Only `trainable_params` receives per-example gradients. Frozen parameters
are treated as constants by `vmap`, which drastically reduces memory usage
since per-example gradients are only computed for the trainable subset.

## Using LoRA with DP-SGD

LoRA is a practical necessity for DP training of large models. Per-example
gradient computation via `vmap` requires memory proportional to
`batch_size * trainable_parameters`. With full fine-tuning of a 7B model,
this is prohibitive. LoRA reduces trainable parameters to ~0.1% of the
model, making per-example gradients feasible.

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from opaque import make_functional, clipped_grad, gaussian_noise
from opaque.random import key

# Load model with LoRA adapters
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="sdpa",
)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

# Functional form: only LoRA params are trainable
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def loss_fn(trainable_params, input_ids, labels):
    out = fmodel(trainable_params, frozen,
                 input_ids=input_ids.unsqueeze(0),
                 labels=labels.unsqueeze(0))
    return out.loss

# DP components
grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2), l2_clip_norm=1.0,
)
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity(), key=key(42),
)

# Training loop
for input_ids, labels in dataloader:
    grads, clip_state = grad_fn(trainable, input_ids, labels, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    trainable = {k: trainable[k] - lr * noisy_grads[k] for k in trainable}
```

### LoRA hyperparameters

| Parameter | Typical values | Effect on DP training |
|-----------|---------------|----------------------|
| `r` (rank) | 4, 8, 16 | Higher = more trainable params = more memory for vmap |
| `lora_alpha` | 16, 32 | Scaling factor; does not affect memory |
| `target_modules` | `["q_proj", "v_proj"]` | More modules = more trainable params |

Start with `r=8` and `target_modules=["q_proj", "v_proj"]`. Increase rank
or add modules only if accuracy is insufficient.

## Feature compatibility

| Feature | Status | Notes |
|---------|--------|-------|
| Mixed precision (fp16/bf16) | Supported | Pass `dtype` to `clipped_grad` for accumulation dtype |
| LoRA / PEFT adapters | Supported | Use `make_functional(partition_trainable=True)` |
| Gradient checkpointing | Not supported | Incompatible with vmap; use microbatching instead |
| `torch.compile` | Supported | Works with vmap and patches |

Gradient checkpointing (`model.gradient_checkpointing_enable()`) uses
`autograd.Function` internally, which is incompatible with `vmap`. Use
the `microbatch_size` parameter on `clipped_grad` as an alternative for
memory reduction. See [Known Limitations](../limitations.md) for details.

## Troubleshooting

**"vmap over _NoopSaveInputs" error:** Gradient checkpointing is enabled.
Disable it: do not call `model.gradient_checkpointing_enable()`.

**Flash Attention errors under vmap:** Set `attn_implementation="sdpa"` or
`attn_implementation="eager"` when loading the model.

**Model not in patched list:** If your model is not LLaMA, Mistral, Qwen2,
Phi, Phi-3, OLMo, Gemma, or Gemma2, it may still work if its attention
follows the standard pattern. Try it; if it fails under vmap, the error
message will indicate which operation needs a vmap rule.

**`make_functional` fails:** Some models use non-standard parameter
registration. Ensure the model is a standard `nn.Module` with parameters
accessible via `model.parameters()`.

## API reference

See [Utilities API Reference](../api/utilities.md) for
`make_functional` signatures and [Clipping API Reference](../api/core/clipping.md)
for `clipped_grad` with `partition_trainable` examples.
