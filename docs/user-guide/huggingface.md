# HuggingFace Compatibility

Opaque works with HuggingFace Transformers models out of the box. When you
`import opaque`, it automatically patches core Transformers functions to be
compatible with `torch.func.vmap`, which is required for per-example
gradient computation. No manual patching is needed.

## Auto-patching

On import, Opaque applies patches to the following HuggingFace Transformers
components:

- **Causal mask creation** (`create_causal_mask`, `_ignore_causal_mask_sdpa`)
  -- handles arbitrary batch dimensions under vmap, including sliding-window
  models (Gemma2, Phi-3, Mistral).
- **Key-value head repetition** (`repeat_kv`) -- uses negative indexing to
  work with both batched (4D) and unbatched (3D) tensors.
- **Eager attention forward** -- replaces model-specific attention with
  vmap-compatible implementations that use dynamic shapes.
- **Batchify wrappers** -- automatically adds/removes the batch dimension
  for model forward methods called under `vmap(grad(...))`.

These patches are applied for LLaMA, Mistral, Qwen2, Qwen3, Phi-3,
Gemma, Gemma2, Granite, Cohere, and Cohere2 models. DeepSeek models
inherit LLaMA patches automatically. GPT-2 works without patches (simple
architecture). Other models may work if their attention implementation
follows the standard Transformers pattern.

To disable auto-patching (e.g., for debugging), set the environment
variable before importing:

```python
import os
os.environ["OPAQUE_SKIP_COMPAT_PATCHES"] = "all"
import opaque
```

For finer control, see [Kernel Optimizations — Configuration](kernel-optimizations.md#configuration).

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

| Attention type | Status | Notes |
|---------------|--------|-------|
| `sdpa` | **Recommended** | Fused CUDA kernels (flash/efficient/cuDNN) with O(N) memory |
| `eager` | Supported | Materializes full attention matrix — O(N²) memory |
| `flash_attention_2` | Not compatible | Uses `torch.nonzero` for unpadding (dynamic shapes break vmap) |
| `flex_attention` | Not compatible | HigherOrderOperator has no vmap support (upstream PyTorch limitation) |

SDPA is the default attention implementation in Transformers and requires no
configuration. It provides significant memory savings over eager because fused
kernels avoid materializing the `(heads, seq, seq)` attention matrix. Measured
at Qwen2-0.5B scale with LoRA:

| seq_len | Microbatch | Eager memory | SDPA memory | Savings |
|---------|-----------|-------------|------------|---------|
| 512 | full batch | 7.38 GB | 4.22 GB | 1.75x |
| 1024 | full batch | 22.28 GB | 6.20 GB | 3.59x |
| 1024 | 2 | 11.80 GB | 4.42 GB | 2.67x |

If your model defaults to Flash Attention 2, override to SDPA when loading:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="sdpa",
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
    out = fmodel(params, input_ids=input_ids, labels=labels)
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
    all_params = {**frozen, **trainable_params}
    out = fmodel(all_params, input_ids=input_ids, labels=labels)
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
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
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

### Other PEFT methods

Any parameter-efficient method that uses standard `requires_grad` flags works
with `make_functional(partition_trainable=True)`:

| Method | Library | Notes |
|--------|---------|-------|
| **LoRA** | `peft` | Recommended. Well-tested with Opaque. |
| **Adapters** (bottleneck) | `peft` / `adapter-transformers` | Works. More trainable params than LoRA at same capacity. |
| **BitFit** (bias-only) | Manual (`requires_grad=False` on non-bias) | Minimal trainable params, very memory efficient. |
| **Prefix tuning** | `peft` | Works but virtual tokens add complexity to loss computation. |
| **IA3** | `peft` | Very few trainable params. Good for memory-constrained settings. |

The key requirement is that trainable parameters are identified by
`requires_grad=True`. If a PEFT method uses custom forward hooks instead of
standard parameters, it may not work with `vmap`.

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
| SDPA attention | Recommended | Default; up to 3.6x memory savings over eager |
| Mixed precision (fp16/bf16) | Supported | Pass `dtype` to `clipped_grad` for accumulation dtype |
| LoRA / PEFT adapters | Supported | Use `make_functional(partition_trainable=True)` |
| Kernel optimizations | Supported | Auto-applied on import; see [Kernel Optimizations](kernel-optimizations.md) |
| Microbatching | Supported | Works with both SDPA and eager attention |
| Gradient checkpointing | Not supported | Incompatible with vmap; use microbatching instead |
| `torch.compile` | Supported | Works with vmap and patches |

Gradient checkpointing (`model.gradient_checkpointing_enable()`) uses
`autograd.Function` internally, which is incompatible with `vmap`. Use
the `microbatch_size` parameter on `clipped_grad` as an alternative for
memory reduction. See [Known Limitations](../limitations.md) for details.

## Model selection

Opaque's auto-patching covers these model families:

| Model family | Tested sizes | Notes |
|-------------|-------------|-------|
| GPT-2 | 124M, 355M | Works without patches; good for prototyping |
| LLaMA / Llama 3 | 7B, 8B, 70B (LoRA) | Recommended starting point |
| DeepSeek | 7B | Uses LLaMA architecture; inherits patches |
| Mistral | 7B | Similar architecture to LLaMA |
| Qwen2 / Qwen3 | 0.5B, 7B | |
| Phi-3 | 3.8B | Combined gate_up_proj variant |
| Gemma / Gemma2 | 2B, 7B | GeGLU activation, softcap attention (Gemma2) |
| Granite | 3B, 8B | Divisive logit scaling |
| Cohere / Cohere2 | 8B | Multiplicative logit scaling |

**What makes a model vmap-compatible:** The model must not use
`torch.nonzero`, data-dependent control flow (`if tensor.item() > 0`), or
operations that depend on the batch dimension being a specific index. Most
standard Transformer architectures work. Encoder-only models (BERT, RoBERTa)
typically work without patches.

**Sequence length and memory:** Attention is $O(\text{seq}^2)$ in memory
(for eager/SDPA). Combined with per-example gradients, long sequences
significantly increase memory. Use shorter sequences (512-1024) when possible,
or reduce the microbatch size.

## Other models

Models not in the table above may work if their architecture follows
the standard Transformers pattern. If a model fails under `vmap`, the most
common cause is that its `forward` method assumes a batch dimension that
`vmap` has stripped.

### Wrapping the loss function

Use `with_batch_dim` to add a leading batch dimension to the arguments
that `vmap` unbatches:

```python
from opaque.utils.functional import with_batch_dim

def loss_fn(params, input_ids, labels):
    out = fmodel(params, input_ids=input_ids, labels=labels)
    return out.loss

loss_fn = with_batch_dim(loss_fn, batch_argnums=(1, 2))

grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2), l2_clip_norm=1.0,
)
```

`batch_argnums` specifies which positional arguments get `unsqueeze(0)`.
Under `vmap`, `input_ids` arrives as `(seq,)`; the wrapper makes it
`(1, seq)` before the model sees it.

### Wrapping the model forward

Alternatively, patch the model's forward method once. This is what Opaque
does internally for supported HuggingFace models:

```python
model.forward = with_batch_dim(
    model.forward,
    batch_kwargs={"input_ids": 2, "attention_mask": 2, "labels": 2, "inputs_embeds": 3},
    min_ndim=2,
)
```

`batch_kwargs` maps keyword argument names to their expected minimum
number of dimensions. When a tensor has fewer dimensions than the
threshold, `unsqueeze(0)` is applied on entry and `squeeze(0)` on the
output. When the tensor already has the expected shape, it is a no-op.

## Distributed HuggingFace models

DDP works with patched HuggingFace models. The patches are applied once
on import and affect all ranks identically.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

dist.init_process_group("nccl")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
# DDP wraps the model; make_functional unwraps it
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)
```

The noise key must be identical across ranks so that each rank adds the same
noise after `sum_gradients`. See [Distributed Training](distributed.md).

## Troubleshooting

**"vmap over _NoopSaveInputs" error:** Gradient checkpointing is enabled.
Disable it: do not call `model.gradient_checkpointing_enable()`.

**Flash Attention errors under vmap:** Set `attn_implementation="sdpa"` when
loading the model. SDPA is the default and recommended implementation.

**"not yet implemented the batching rule" warning:** This PyTorch warning
about `_scaled_dot_product_*_attention_backward` indicates that SDPA backward
falls back to per-sample processing under vmap. A fix has been submitted
upstream to PyTorch to add proper batching rules for SDPA backward.

**Model not in patched list:** See [Other models](#other-models)
for how to use `with_batch_dim` to add vmap support.

**`make_functional` fails:** Some models use non-standard parameter
registration. Ensure the model is a standard `nn.Module` with parameters
accessible via `model.parameters()`.

## API reference

See [Utilities API Reference](../api/utilities.md) for
`make_functional` signatures and [Clipping API Reference](../api/clipping.md)
for `clipped_grad` with `partition_trainable` examples.
