# Model Patches and Kernels

`opaque.patches` is a standalone library that makes HuggingFace
Transformers models work under `torch.func.vmap(grad(...))` and
provides fused Triton kernels for the hot ops on the forward /
backward path.  It predates and operates independently of
`DPTrainer` — any code that drives DP-SGD over HF models (the
trainer, a hand-rolled training loop, a custom orchestration layer)
can use the same APIs.

Two concerns are handled:

- **Compat patches** — vmap-safety rewrites for attention, causal
  masks, KV-cache, batchify wrappers, gradient checkpointing, and
  collator behaviour under Poisson sampling.  Required so
  `vmap(grad(...))` over a per-example loss closure doesn't trip on
  hardcoded batch-dim indexing or data-dependent control flow.
- **Performance kernels** — fused Triton kernels for RoPE, RMSNorm,
  activations (SwiGLU / GeGLU), cross-entropy, optional fused linear
  CE, and fused LoRA.  Drop-in numerically-equivalent replacements;
  significant memory savings on long-sequence models.

## API surface

```python
from opaque.patches import apply_runtime_patches, apply_model_patches

apply_runtime_patches()                       # global HF shims, once at startup

model = AutoModelForCausalLM.from_pretrained(...)
# Optional: attach LoRA / PEFT adapters here.
apply_model_patches(
    model,
    compat=True,                              # vmap-safety (eager-attn, batchify, …)
    performance=True,                         # kv_cache (pure-Python; always-on)
    kernels=True,                             # CUDA + Triton group (rope, rms_norm, …)
    peft=True,                                # LoRA fusion when adapters detected
    fused_linear_cross_entropy=False,         # opt-in (fused forward returns logits=None)
)
```

`apply_runtime_patches()` should be called once near process startup,
before creating checkpointed models or HF data collators.
`apply_model_patches(model)` runs after the model is instantiated and
after any PEFT / LoRA wrapping, so the patcher sees the final module
graph.

### Umbrella flags

| Flag | Default | Effect |
|---|---|---|
| `compat` | `True` | vmap-safety wrappers — `eager_attention`, `batchify`, `vmap_masking`, `empty_batches`, `vmap_checkpointing`. |
| `performance` | `True` | Memory-efficiency patches that run on any host (currently `kv_cache`). |
| `kernels` | `performance` | CUDA + Triton kernel group — `rope`, `rms_norm`, `activation`, `cross_entropy`.  Forced `False` when CUDA + Triton aren't importable, so `performance=True` keeps `kv_cache` on CPU / MPS hosts. |
| `peft` | `True` | LoRA / PEFT module fusion (`opaque_lora_*`). |
| `fused_linear_cross_entropy` | `False` | Promoted kernel kwarg — opt-in because the fused forward returns `logits=None`, which is incompatible with callers that read `outputs.logits`. |

Each umbrella forwards to per-concern boolean kwargs in `**kwargs`,
so you can override individual patches without flipping the whole
group:

```python
apply_model_patches(
    model,
    kernels=True,
    rope=False,                  # disable just the RoPE kernel
)
apply_runtime_patches(
    vmap_checkpointing=False,    # debug: keep stock torch.utils.checkpoint
)
```

## Why patches are needed

`torch.func.vmap` removes the batch dimension: a function that
normally receives input of shape `(batch, seq_len, hidden)` instead
receives `(seq_len, hidden)`, with `vmap` handling batching
externally.  HuggingFace models use hardcoded dimension indices
(e.g. `x.shape[0]` for batch size) and data-dependent control flow
that break under vmap.

The patches replace these with:

- **Negative indexing** (`shape[-3]`, `transpose(-2, -1)`) — works
  regardless of how many leading dimensions exist.
- **Dynamic dimension detection** (`ndim == 2` for vmap vs `ndim >=
  3` for normal) to adapt mask shapes.
- **Broadcasting** instead of explicit batch-dimension manipulation.

Concrete patch targets:

- **Causal mask creation** (`create_causal_mask`,
  `_ignore_causal_mask_sdpa`) — handles arbitrary batch dimensions
  under vmap, including sliding-window models (Gemma2, Phi-3,
  Mistral).
- **Key-value head repetition** (`repeat_kv`) — uses negative
  indexing to work with both batched (4D) and unbatched (3D)
  tensors.
- **Eager attention forward** — replaces model-specific attention
  with vmap-compatible implementations using dynamic shapes.
- **Batchify wrappers** — automatically add / remove the batch
  dimension for model forward methods called under
  `vmap(grad(...))`.
- **Gradient checkpointing** — `torch.utils.checkpoint` is
  incompatible with `vmap(grad(...))` out of the box; the runtime
  patch installs the saved-tensors-hooks + non-reentrant + functional
  shim that makes it compose.

## Attention implementations

| Attention type | Status | Notes |
|---|---|---|
| `sdpa` | **Recommended** | Fused CUDA kernels (flash / efficient / cuDNN) with O(N) memory. |
| `eager` | Supported | Materialises full attention matrix — O(N²) memory. |
| `flash_attention_2` | **Not compatible** | Uses `torch.nonzero` for unpadding (dynamic shapes break vmap). |
| `flex_attention` | **Not compatible** | HigherOrderOperator has no vmap support (upstream PyTorch limitation). |

SDPA is the Transformers default and requires no configuration.  It
provides significant memory savings over eager because fused kernels
avoid materialising the `(heads, seq, seq)` attention matrix.
Measured at Qwen2-0.5B scale with LoRA:

| seq_len | Microbatch | Eager memory | SDPA memory | Savings |
|---|---|---|---|---|
| 512 | full batch | 7.38 GB | 4.22 GB | 1.75× |
| 1024 | full batch | 22.28 GB | 6.20 GB | 3.59× |
| 1024 | 2 | 11.80 GB | 4.42 GB | 2.67× |

If your model defaults to Flash Attention 2, override at load time:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="sdpa",
)
```

## Model compatibility

`apply_model_patches` dispatches per `config.model_type` to a
registered family handler.  The table shows vmap compatibility and
which fused Triton kernels are applied per model:

| Model | Tested sizes | Activation | RMSNorm | RoPE | CE | Fused Linear CE | LoRA Fusion |
|---|---|---|---|---|---|---|---|
| LLaMA / Llama 3 | 7B, 8B, 70B (LoRA) | SwiGLU | Yes | Yes | Yes | Yes | QKV + MLP |
| Mistral | 7B | SwiGLU | Yes | Yes | Yes | Yes | QKV + MLP |
| Ministral | 8B | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| Qwen2 / Qwen3 | 0.5B, 7B | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| SmolLM3 | 3B | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| OLMo2 | 1B, 7B (tiny config) | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| OLMo3 | 1B, 7B (tiny config) | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| GLM4 | 9B (tiny config) | SwiGLU (Phi-3 style) | Yes | — | Yes | Yes | MLP only |
| Phi-3 | 3.8B | SwiGLU | Yes | Yes | Yes | — | — |
| Gemma | 2B, 7B | GeGLU Exact | Yes | Yes | Yes | Yes | QKV + MLP |
| Gemma2 | 2B, 7B | GeGLU Approx | Yes | Yes | Yes | Yes (softcap) | QKV + MLP |
| Gemma3 (text) | 1B, 4B (tiny config) | GeGLU Approx | Yes | Yes | Yes | Yes | MLP only |
| Granite | 3B, 8B | SwiGLU | Yes | Yes | Yes | Yes | QKV + MLP |
| Cohere | 8B | SwiGLU | — | Yes | Yes | Yes | MLP only |
| Cohere2 | 8B | SwiGLU | — | Yes | Yes | Yes | QKV + MLP |
| Exaone4 | 1.2B, 32B (tiny config) | SwiGLU | Yes | Yes | Yes | Yes | MLP only |
| GPT-2 | 124M, 355M | — | — | — | — | — | — |

DeepSeek-Coder ships with `config.model_type == "llama"` and therefore
inherits the LLaMA family registration end-to-end — vmap compat,
SwiGLU / RMSNorm / RoPE kernels, fused linear CE, and QKV + MLP LoRA
fusion all apply.  Validated against
`deepseek-ai/deepseek-coder-1.3b-base`.  A DeepSeek variant shipping a
non-`llama` `model_type` would land in the "Other models" path below.

Gemma3's `q_norm` / `k_norm` RMSNorms inside `Gemma3Attention` are
picked up automatically because the kernel patcher rebinds
`Gemma3RMSNorm.forward` at class level; the same trick covers
`Exaone4RMSNorm` for Exaone4.  Fused add + RMSNorm is intentionally
**off** for OLMo2 / OLMo3 / Cohere / Cohere2 / Gemma3 / Exaone4
because their decoder layers apply `post_attention_layernorm`
*between* the attention output and the residual add (the fused
primitive expects residual-first ordering).

**Not supported** by default patching: expert-routed decoder stacks
such as GPT-OSS.

**Deferred families:** Nemotron —
`transformers.models.nemotron.modeling_nemotron` in `4.57.1` ships
only legacy `NemotronAttention` / `NemotronSdpaAttention` /
`NemotronFlashAttention2` (no `eager_attention_forward` symbol to
swap), and `NemotronMLP` is non-gated (`up_proj → act_fn →
down_proj`, no SwiGLU / GeGLU split).  Adding it would require both
a bespoke vmap attention path and a new non-gated MLP kernel;
revisit when a customer benchmark needs it.

**What makes a model vmap-compatible**: the model must not use
`torch.nonzero`, data-dependent control flow (`if tensor.item() >
0`), or operations that depend on the batch dimension being a
specific index.  Most standard Transformer architectures work.
Encoder-only models (BERT, RoBERTa) typically work without patches.

**Sequence length and memory**: attention is $O(\text{seq}^2)$ in
memory (for eager / SDPA).  Combined with per-example gradients,
long sequences significantly increase memory.  Use shorter
sequences (512–1024) when possible, or reduce the microbatch size.

## Triton kernels

Opaque ships fused Triton kernels for the hot ops in transformer
forward / backward.  All are numerically equivalent to PyTorch
reference implementations within floating-point precision; the
benchmarks live in [Memory Optimizations — Kernel benchmarks](../memory-optimizations.md#kernel-benchmarks).

### Activation functions

| Kernel | Operation | Models |
|---|---|---|
| SwiGLU | `silu(gate) * up` | LLaMA, Mistral, Qwen2, Qwen3, Phi-3, Granite, Cohere, Cohere2 |
| GeGLU Exact | `gelu(gate) * up` (erf-based) | Gemma |
| GeGLU Approx | `gelu_tanh(gate) * up` | Gemma2, Gemma3 |

Each fuses the forward activation into one kernel and provides a
fused backward that reads 3 tensors and writes 2–3 in a single pass.

### Rotary position embeddings (RoPE)

Fused RoPE applies rotary embeddings to Q and K tensors
simultaneously.  Supports grouped-query attention (GQA) where Q and
K have different head counts.

Patched models: LLaMA, Mistral, Qwen2, Qwen3, Phi-3, Gemma, Gemma2,
Granite.

### Cross-entropy

Chunked Triton cross-entropy processes the vocabulary dimension in
blocks (up to 65536), avoiding materialisation of the full
`(batch*seq, vocab)` logits tensor.

The kernel honours `label_smoothing` natively (`F.cross_entropy(...,
label_smoothing=...)` parity) — pass `label_smoothing=...` as a loss
kwarg and the kernel applies the smoothed formula directly.
Available standalone as `opaque.patches.kernels.opaque_cross_entropy_loss`.

### Fused linear cross-entropy

Computes the loss directly from hidden states and the `lm_head`
weight matrix using tiled matrix multiplication inside the Triton
kernel, never materialising the full logits tensor.  For Mellum-4b
with 128K vocab, this avoids the ~2 GB `logits = hidden_states @
lm_head.T` allocation that the non-fused path produces per forward
pass.

The patched `XForCausalLM.forward` returns `logits=None` on the
fast path, which is incompatible with callers that read
`outputs.logits` — `compute_metrics`,
`preprocess_logits_for_metrics`, and generation eval all need the
materialised tensor.  The patch is **opt-in** for this reason:
enable via `apply_model_patches(model,
fused_linear_cross_entropy=True)` when loss is the only consumer of
the forward output.  `examples/train_causal_lm.py` and
`examples/train_dp_ftrl.py` enable it.

`cross_entropy=True` (default when `kernels=True`) still installs
the non-fused chunked CE via `loss_function`, which operates on
materialised logits and leaves `outputs.logits` populated.

Key design decisions:

- **Pre-shift in Python** — `hidden_states[..., :-1, :]` and
  `labels[..., 1:]` so vmap merge is a trivial reshape.
- **Skip weight gradient when frozen** — in LoRA training,
  `lm_head` is frozen, so ~1/3 of backward compute is skipped
  entirely.
- **Weight scaling outside the kernel** — Cohere (multiplicative),
  Granite (divisive), and Gemma2 (softcapping) are handled
  correctly.

Patched models: LLaMA, Mistral, Qwen2, Qwen3, Gemma, Gemma2,
Granite, Cohere, Cohere2.

### Fused LoRA operations

Three LoRA fusion levels, applied automatically when LoRA adapters
are detected and `peft=True` is passed to `apply_model_patches`:

| Kernel | Description |
|---|---|
| `opaque_lora_w` | Single linear: `x @ W.T + x @ A @ B * s` — avoids intermediate `x @ A`. |
| `opaque_lora_qkv` | Fused Q+K+V: shares input across 3 projections in one call. |
| `opaque_lora_mlp` | Fused gate+up+down: 3 projections + activation in one call. |

`opaque_lora_w` patches `peft.tuners.lora.Linear.forward` and applies
to all LoRA layers.  `opaque_lora_qkv` and `opaque_lora_mlp` are
auto-fused when all projections in an attention block or MLP block
have LoRA adapters with no bias.

QKV fusion eligible models: LLaMA, Mistral, Gemma, Gemma2, Granite,
Cohere2.  Excluded: Qwen2 (bias on Q/K/V), Qwen3 (q_norm/k_norm),
Phi-3 (combined qkv_proj), Cohere (no transpose).

### Using kernels directly

All kernels are available as standalone functions without patching:

```python
from opaque.patches.kernels import opaque_swiglu, opaque_cross_entropy_loss

h = opaque_swiglu(gate, up)
loss = opaque_cross_entropy_loss(logits, labels)
```

## Other models

Models not in the compatibility matrix above may work if their
architecture follows the standard Transformers pattern.  If a model
fails under `vmap`, the most common cause is that its `forward`
method assumes a batch dimension that `vmap` has stripped.

### Wrapping the loss function

Use `with_batch_dim` to add a leading batch dimension to the
arguments that `vmap` unbatches:

```python
from opaque.functional import with_batch_dim

def loss_fn(params, input_ids, labels):
    out = fmodel(params, input_ids=input_ids, labels=labels)
    return out.loss

loss_fn = with_batch_dim(loss_fn, batch_argnums=(1, 2))
```

`batch_argnums` specifies which positional arguments get
`unsqueeze(0)`.  Under `vmap`, `input_ids` arrives as `(seq,)`; the
wrapper makes it `(1, seq)` before the model sees it.

### Wrapping the model forward

Alternatively, patch the model's forward method once.  This is what
Opaque does internally for supported HF models:

```python
model.forward = with_batch_dim(
    model.forward,
    batch_kwargs={"input_ids": 2, "attention_mask": 2, "labels": 2, "inputs_embeds": 3},
    min_ndim=2,
)
```

`batch_kwargs` maps keyword argument names to their expected
minimum number of dimensions.  When a tensor has fewer dimensions
than the threshold, `unsqueeze(0)` is applied on entry and
`squeeze(0)` on the output.  When the tensor already has the
expected shape, it is a no-op.

### Custom non-HF nn.Module

For an arbitrary `nn.Module` used downstream of `make_functional`,
the `forward` should return a dict-like `ModelOutput` (or any
`Mapping` with `"loss"` and optionally `"logits"`) so consumers like
`DPTrainer.prediction_step` can read named fields.  Wrap if needed:

```python
class MyModel(torch.nn.Module):
    main_input_name = "features"

    def forward(self, features, labels=None):
        logits = self.head(features)
        if labels is None:
            return {"logits": logits}
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
```

If the model is already vmap-safe and doesn't need opaque's compat
shims, pass `compat=False` to `apply_model_patches` (or
`use_compat_patches=False` to DPTrainer).

## DPTrainer integration

`DPTrainer` calls `apply_runtime_patches(...)` and
`apply_model_patches(model, ...)` during `__init__`, driven by three
`TrainingArguments` fields — using the trainer doesn't require calling
the patch APIs directly:

| Field | Default | Effect |
|---|---|---|
| `use_compat_patches` | `True` | Routed to `compat`.  Set `False` for custom models that don't need vmap-safety shims. |
| `use_performance_kernels` | `False` | Routed to `kernels`.  Auto-`False` on hosts without CUDA + Triton. |
| `performance_kernels_config` | `None` | Flat `dict[str, bool]` forwarded as-is to `apply_model_patches` / `apply_runtime_patches` kwargs.  Per-concern override. |

The trainer always passes `performance=True` (so `kv_cache` is on
regardless), and `peft=True` so LoRA fusion engages when adapters
are detected.  `performance_kernels_config` accepts any of the
per-concern keys discussed above:

```python
args = TrainingArguments(
    use_performance_kernels=True,
    performance_kernels_config={
        "fused_linear_cross_entropy": True,   # opt-in
        "kv_cache": False,                    # for HF DynamicCache-dependent models
    },
)
```

## See also

- [PEFT and LoRA](peft.md) — `make_functional`, the trainable / frozen
  partition, and the LoRA training recipe.
- [Memory Optimizations](../memory-optimizations.md) — kernel
  benchmarks and gradient checkpointing.
- [DPTrainer](dptrainer.md) — when the trainer drives the patch
  surface for you.
- [Distributed Training](../distributed-trainer.md) — DDP specifics.
