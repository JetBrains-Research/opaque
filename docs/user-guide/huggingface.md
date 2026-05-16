# HuggingFace Compatibility

Opaque works with HuggingFace Transformers models. Patches are split across
two sub-packages:

- `opaque.patches` — the operational entrypoint. `apply_runtime_patches()`
  enables runtime fixes (checkpointing, masking, collator, loss mapping), and
  `apply_model_patches(model)` wires compat wrappers and Triton kernels into a
  concrete model instance.
- `opaque.transformers` — the namespace package installed by
  `opaque-transformers`; it carries the Transformers-facing dependency bundle,
  while the patch APIs live under `opaque.patches`.

**Scope:** the curated matrix prioritises **decoder-only text** models
(``ForCausalLM`` and shared text modules). Vision-language stacks
(e.g. ``*ForConditionalGeneration``) are not part of the default patch set.
Tested against `transformers==4.57.1`.

Recommended usage matches the training examples:

```python
from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

model = AutoModelForCausalLM.from_pretrained(...)
# Optional: attach LoRA / PEFT adapters here.
apply_model_patches(model)
```

Call `apply_runtime_patches()` once near process startup, before creating
checkpointed models or Hugging Face data collators. Call
`apply_model_patches(model)` after the model is instantiated and after any
PEFT/LoRA wrapping, so the patcher sees the final module graph.

## Auto-patching

After `apply_runtime_patches()` and `apply_model_patches(model)`, Opaque
patches the following HuggingFace Transformers components:

- **Causal mask creation** (`create_causal_mask`, `_ignore_causal_mask_sdpa`)
  -- handles arbitrary batch dimensions under vmap, including sliding-window
  models (Gemma2, Phi-3, Mistral).
- **Key-value head repetition** (`repeat_kv`) -- uses negative indexing to
  work with both batched (4D) and unbatched (3D) tensors.
- **Eager attention forward** -- replaces model-specific attention with
  vmap-compatible implementations that use dynamic shapes.
- **Batchify wrappers** -- automatically adds/removes the batch dimension
  for model forward methods called under `vmap(grad(...))`.

These patches are applied for LLaMA, Mistral, Ministral, Qwen2, Qwen3,
SmolLM3, OLMo2, OLMo3, GLM4, Phi-3, Gemma, Gemma2, Gemma3 (text), Granite,
Cohere, Cohere2, and Exaone4 models. DeepSeek models inherit LLaMA patches
automatically. GPT-2 works without patches (simple architecture). Other
text models may work if their attention implementation follows the
standard Transformers pattern.

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

## Model compatibility

Opaque's auto-patching covers these model families. The table shows both
vmap compatibility and which fused Triton kernels are applied per model:

| Model | Tested sizes | SwiGLU/GeGLU | RMSNorm | RoPE | CE | Fused Linear CE | LoRA Fusion |
|-------|-------------|--------------|---------|------|----|-----------------|-------------|
| LLaMA / Llama 3 | 7B, 8B, 70B (LoRA) | SwiGLU | Yes | Yes | Yes | Yes | QKV + MLP |
| Mistral | 7B | SwiGLU | Yes | Yes | Yes | Yes | QKV + MLP |
| Ministral | 8B | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| Qwen2 / Qwen3 | 0.5B, 7B | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| SmolLM3 | 3B | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| OLMo2 | 1B, 7B (tiny config in tests) | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| OLMo3 | 1B, 7B (tiny config in tests) | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| GLM4 | 9B (tiny config in tests) | SwiGLU (Phi3-style) | Yes | -- | Yes | Yes | MLP only |
| Phi-3 | 3.8B | SwiGLU | Yes | Yes | Yes | -- | -- |
| Gemma | 2B, 7B | GeGLU Exact | Yes | Yes | Yes | Yes | QKV + MLP |
| Gemma2 | 2B, 7B | GeGLU Approx | Yes | Yes | Yes | Yes (softcap) | QKV + MLP |
| Gemma3 (text) | 1B, 4B (tiny config in tests) | GeGLU Approx | Yes | Yes | Yes | Yes | MLP only |
| Granite | 3B, 8B | SwiGLU | Yes | Yes | Yes | Yes | QKV + MLP |
| Cohere | 8B | SwiGLU | -- | Yes | Yes | Yes | MLP only |
| Cohere2 | 8B | SwiGLU | -- | Yes | Yes | Yes | QKV + MLP |
| Exaone4 | 1.2B, 32B (tiny config in tests) | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| GPT-2 | 124M, 355M | -- | -- | -- | -- | -- | -- |
| DeepSeek | 7B | SwiGLU | Yes | Yes | Yes | Yes | QKV + MLP |

Gemma3's `q_norm` / `k_norm` RMSNorms inside `Gemma3Attention` are picked up
automatically because the kernel patcher rebinds `Gemma3RMSNorm.forward` at
class level; the same trick covers `Exaone4RMSNorm` for Exaone4. Fused add +
RMSNorm is intentionally **off** for OLMo2 / OLMo3 / Cohere / Cohere2 /
Gemma3 / Exaone4 because their decoder layers apply
`post_attention_layernorm` *between* the attention output and the residual
add (the fused primitive expects residual-first ordering).

**Not supported by default patching:** expert-routed decoder stacks such as
GPT-OSS. **Deferred families:** Nemotron — `transformers.models.nemotron.modeling_nemotron`
in 4.57.1 ships only legacy `NemotronAttention` / `NemotronSdpaAttention` /
`NemotronFlashAttention2` (no `eager_attention_forward` symbol to swap), and
`NemotronMLP` is non-gated (`up_proj → act_fn → down_proj`, no
SwiGLU/GeGLU split). Adding it would require both a bespoke vmap attention
path and a new non-gated MLP kernel; revisit when a benchmark customer
needs it.

**What makes a model vmap-compatible:** The model must not use
`torch.nonzero`, data-dependent control flow (`if tensor.item() > 0`), or
operations that depend on the batch dimension being a specific index. Most
standard Transformer architectures work. Encoder-only models (BERT, RoBERTa)
typically work without patches.

**Sequence length and memory:** Attention is $O(\text{seq}^2)$ in memory
(for eager/SDPA). Combined with per-example gradients, long sequences
significantly increase memory. Use shorter sequences (512-1024) when possible,
or reduce the microbatch size.

## Functional model conversion

PyTorch models store parameters internally. To use them with
`clipped_grad`, convert to functional form with `make_functional`:

```python
from opaque.dpsgd.clipping import clipped_grad
from opaque.functional import make_functional

model = AutoModelForCausalLM.from_pretrained("gpt2")
fmodel, params = make_functional(model)

def loss_fn(params, input_ids, labels):
    out = fmodel(params, input_ids=input_ids, labels=labels)
    return out.loss

grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0,
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
    loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0,
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
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.functional import make_functional
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
    loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0,
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
  noise_multiplier=noise_multiplier, key=key(42),
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

## Patched operations

Opaque replaces standard PyTorch operations with fused Triton kernels for
supported models. These are applied by `apply_model_patches(model)` and
require no model-code changes. See [Memory Optimizations — Kernel benchmarks](memory-optimizations.md#kernel-benchmarks)
for performance numbers.

### Activation functions

| Kernel | Operation | Models |
|--------|-----------|--------|
| SwiGLU | `silu(gate) * up` | LLaMA, Mistral, Qwen2, Qwen3, Phi3, Granite, Cohere, Cohere2 |
| GeGLU Exact | `gelu(gate) * up` (erf-based) | Gemma |
| GeGLU Approx | `gelu_tanh(gate) * up` | Gemma2 |

Each fuses the forward activation into one kernel and provides a fused backward
that reads 3 tensors and writes 2-3 in a single pass.

### Rotary position embeddings (RoPE)

Fused RoPE applies rotary embeddings to Q and K tensors simultaneously. Supports
grouped-query attention (GQA) where Q and K have different head counts.

Patched models: LLaMA, Mistral, Qwen2, Qwen3, Phi3, Gemma, Gemma2, Granite.

### Cross-entropy loss

Chunked Triton cross-entropy processes the vocabulary dimension in blocks (up to
65536), avoiding materialization of the full `(batch*seq, vocab)` logits tensor.

### Fused linear cross-entropy

The most impactful memory optimization, and **opt-in**: pass
`fused_linear_cross_entropy=True` to `apply_model_patches(model, ...)`
to enable it. Standard cross-entropy materializes
`logits = hidden_states @ lm_head.T` — for Mellum-4b with 128K vocab,
~2 GB per forward pass. The fused kernel computes the loss directly
from hidden states and weight matrix using tiled matrix multiplication
inside Triton, never materializing the full logits tensor.

When enabled, `XForCausalLM.forward` returns `logits=None` on the fast
path (the kernel writes the loss directly). That is incompatible with
trainers or eval loops that read `outputs.logits` — `compute_metrics`,
`preprocess_logits_for_metrics`, generation eval, and similar consumers
all need the materialized tensor. The default is off so those cases
keep working; enable it explicitly when loss is the only consumer.
The bundled `train_causal_lm.py` and `train_dp_ftrl.py` examples
opt in.

The `cross_entropy=True` gate (default-on with `performance=True`)
still installs `Opaque_CrossEntropyLoss` via `loss_function` — that
swap keeps logits materialized and is safe everywhere.

Key design decisions:
- **Pre-shift in Python** — `hidden_states[..., :-1, :]` and `labels[..., 1:]`
  so vmap merge is a trivial reshape
- **Skip weight gradient when frozen** — in DP-SGD LoRA training, `lm_head`
  is frozen, so ~1/3 of backward compute is skipped entirely
- **Weight scaling outside kernel** — Cohere (multiplicative), Granite
  (divisive), and Gemma2 (softcapping) are handled correctly

Patched models: LLaMA, Mistral, Qwen2, Qwen3, Gemma, Gemma2, Granite,
Cohere, Cohere2.

### Fused LoRA operations

Three LoRA fusion levels, applied automatically when LoRA adapters are detected:

| Kernel | Description |
|--------|-------------|
| `Opaque_LoRA_W` | Single linear: `x @ W.T + x @ A @ B * s` — avoids intermediate `x @ A` |
| `Opaque_LoRA_QKV` | Fused Q+K+V: shares input across 3 projections in one call |
| `Opaque_LoRA_MLP` | Fused gate+up+down: 3 projections + activation in one call |

**LoRA_W** patches `peft.tuners.lora.Linear.forward` and applies to all LoRA
layers. **LoRA_QKV** and **LoRA_MLP** are auto-fused when all projections in an
attention block or MLP block have LoRA adapters with no bias.

QKV fusion eligible models: LLaMA, Mistral, Gemma, Gemma2, Granite, Cohere2.
Excluded: Qwen2 (bias on Q/K/V), Qwen3 (q_norm/k_norm), Phi3 (combined
qkv_proj), Cohere (no transpose).

### Using kernels directly

All kernels are available as standalone functions without patching:

```python
from opaque.patches.kernels import opaque_swiglu, opaque_cross_entropy_loss

h = opaque_swiglu(gate, up)
loss = opaque_cross_entropy_loss(logits, labels)
```

## Feature compatibility

| Feature | Status | Notes |
|---------|--------|-------|
| SDPA attention | Recommended | Default; up to 3.6x memory savings over eager |
| Mixed precision (fp16/bf16) | Supported | Pass `dtype` to `clipped_grad` for accumulation dtype |
| LoRA / PEFT adapters | Supported | Use `make_functional(partition_trainable=True)` |
| Kernel optimizations | Supported | Auto-applied on import; see [Patched operations](#patched-operations) |
| Microbatching | Supported | Works with both SDPA and eager attention |
| Gradient checkpointing | Supported | See [Memory Optimizations](memory-optimizations.md#gradient-checkpointing) |
| `torch.compile` | Supported | Works with vmap and patches |

## Configuration

Patching is configured through the explicit API rather than import-time side
effects. Three umbrella switches gate broad buckets of patches, plus per-concern
kwargs for fine control.

```python
from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches(
  compat=True,
  vmap_masking=True,
  empty_batches=True,
  vmap_checkpointing=True,
)

apply_model_patches(
  model,
  performance=True,  # kv_cache disabler + (by default) the kernels group
  compat=True,       # vmap-safe attention + batchify
  peft=True,         # LoRA / PEFT module patching
  kernels=True,      # Triton kernels (rope, rms_norm, activation, cross_entropy);
                     # defaults to performance, auto-False without CUDA + Triton
  fused_linear_cross_entropy=True,  # opt-in: skip lm_head materialization, fast path returns logits=None
)
```

Group → per-concern defaults:

- `performance` (memory / efficiency wins that run anywhere): `kv_cache`.
- `compat` (vmap safety): `eager_attention`, `batchify`.
- `kernels` (Triton kernels, need CUDA + Triton): `rope`, `rms_norm`,
  `activation`, `cross_entropy`. Defaults to `performance` when not passed;
  auto-forced to `False` when CUDA / Triton can't be imported, so
  `performance=True` keeps shipping `kv_cache` on CPU / MPS.

`fused_linear_cross_entropy` is the one kernel kwarg that is **always opt-in**:
the fused path returns `logits=None` and breaks any trainer that reads
`outputs.logits` (compute_metrics, preprocess_logits_for_metrics, eval loops
that inspect logits). Enable it only when loss is the sole consumer of the
forward output — see [Fused linear cross-entropy](#fused-linear-cross-entropy).

Common configurations:

```python
# Keep compat shims, drop everything performance-related.
apply_model_patches(model, performance=False)

# Keep kv_cache (pure-Python perf shim) but skip Triton kernels — what
# the router does on CPU / MPS automatically.
apply_model_patches(model, performance=True, kernels=False)

# Drop model-side compat wrappers, keep runtime patches.
apply_model_patches(model, compat=False)

# Maximize memory savings — loss-only consumer of forward outputs.
apply_model_patches(model, fused_linear_cross_entropy=True)

# Disable runtime checkpoint patching for debugging.
apply_runtime_patches(vmap_checkpointing=False)
```

The only environment-level switch in this area is
`OPAQUE_SKIP_TRANSFORMERS_DATA_PATCHES=all|collator`, which controls the empty-
batch collator wrapper used with Poisson sampling.

## Other models

Models not in the table above may work if their architecture follows
the standard Transformers pattern. If a model fails under `vmap`, the most
common cause is that its `forward` method assumes a batch dimension that
`vmap` has stripped.

### Wrapping the loss function

Use `with_batch_dim` to add a leading batch dimension to the arguments
that `vmap` unbatches:

```python
from opaque.functional import with_batch_dim

def loss_fn(params, input_ids, labels):
    out = fmodel(params, input_ids=input_ids, labels=labels)
    return out.loss

loss_fn = with_batch_dim(loss_fn, batch_argnums=(1, 2))

grad_fn, clip_state = clipped_grad(
    loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0,
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

**"vmap over _NoopSaveInputs" error:** This should not occur if `import opaque`
was called before enabling gradient checkpointing. Ensure Opaque is imported
first so its checkpoint compatibility patches are applied.

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

**Numerical differences after enabling kernel patches:** All kernels are
numerically equivalent to PyTorch reference implementations within
floating-point precision. If you see significant differences, file a bug report.

**Performance regression with kernel patches:** Some kernels (SwiGLU/GeGLU
forward) have higher dispatch overhead than native PyTorch for small tensors.
The net effect on end-to-end training is positive due to backward pass and
memory savings. Use `OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES` to isolate
which kernel causes the issue.

**LoRA fusion not applied:** Auto-fusion requires all projections in a group
(Q+K+V or gate+up+down) to have LoRA adapters with no bias. Check your
`LoraConfig.target_modules` and model architecture.

## API reference

See [Utilities API Reference](../reference/utilities.md) for
`make_functional` signatures and [Clipping API Reference](../reference/clipping.md)
for `clipped_grad` with `partition_trainable` examples.
See the [Kernels API](../reference/index.md) for complete kernel function signatures.
