# Kernel Optimizations

DP-SGD training with `vmap(grad())` is memory-intensive: per-example gradients
require materializing one gradient copy per sample. Standard PyTorch operations
create unnecessary intermediate tensors that compound this cost. Opaque includes
fused Triton kernels that reduce memory usage and improve throughput without
changing training semantics.

## How it works

On `import opaque`, kernel patches are applied automatically to supported
HuggingFace Transformers models. No code changes are needed in your training
script. The patches replace standard PyTorch operations with fused Triton
kernels that:

- **Eliminate intermediate tensors** — fused forward passes compute results
  in a single kernel instead of multiple operations.
- **Recompute in backward** — instead of saving activations for the backward
  pass, recompute them from inputs (trading compute for memory).
- **Support vmap natively** — each kernel implements `autograd.Function` with
  custom `vmap()` static methods, so `vmap(grad())` works without fallbacks.

All kernels follow the same two-level dispatch pattern:

1. `Opaque_Foo(autograd.Function)` — main entry via `.apply()`
   - `forward()` / `setup_context()` / `backward()` for standard autograd
   - `vmap()` — Triton forward inline (called by functorch under `vmap`)

2. `_FooBackward(autograd.Function)` — backward wrapper
   - `forward()` — runs Triton backward kernel
   - `vmap()` — Triton backward with merged batch dims

When `vmap(grad(fn))` runs, functorch intercepts `.apply()` and routes to the
`vmap()` methods, where tensors are regular (unwrapped) and Triton kernels
work directly.

## Patched operations

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

The most impactful optimization. Standard cross-entropy requires materializing
`logits = hidden_states @ lm_head.T` — for Mellum-4b with 128K vocab, this is
~2 GB per forward pass. Fused linear cross-entropy computes the loss directly
from hidden states and weight matrix using tiled matrix multiplication inside
the Triton kernel, never materializing the full logits tensor.

Three Triton kernels (ported from Apple's cut_cross_entropy, simplified):

- **Forward**: 2D tiled grid, tiled matmul with per-block LSE and lock-based
  atomic `logaddexp` across vocabulary blocks
- **Backward**: recomputes logits, computes CE gradient, lock-based gradient
  accumulation for hidden states and weight
- **mm_backward**: helper for lock-based tiled matmul gradient accumulation

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

## Model compatibility

| Model | SwiGLU/GeGLU | RoPE | CE | Fused Linear CE | LoRA Fusion |
|-------|--------------|------|----|-----------------|-------------|
| LLaMA | SwiGLU | Yes | Yes | Yes | QKV + MLP |
| Mistral | SwiGLU | Yes | Yes | Yes | QKV + MLP |
| Qwen2 | SwiGLU | Yes | Yes | Yes | MLP only |
| Qwen3 | SwiGLU | — | Yes | Yes | MLP only |
| Phi3 | SwiGLU | Yes | Yes | — | — |
| Gemma | GeGLU Exact | Yes | Yes | Yes | QKV + MLP |
| Gemma2 | GeGLU Approx | Yes | Yes | Yes (softcap) | QKV + MLP |
| Granite | SwiGLU | Yes | Yes | Yes | QKV + MLP |
| Cohere | SwiGLU | — | Yes | Yes | MLP only |
| Cohere2 | SwiGLU | — | Yes | Yes | QKV + MLP |

## Performance

Measured at Mellum-4b scale (batch=4, seq=1024, vocab=128256, LoRA r=16):

### Kernel-level benchmarks

| Kernel | Forward | Backward | Memory | vmap(grad) speed | vmap(grad) memory |
|--------|---------|----------|--------|------------------|-------------------|
| SwiGLU | 0.69x | 0.83x | 1.20x | 1.19x | 2.10x |
| GeGLU Exact | 0.76x | 0.78x | 1.38x | 0.84x | 1.43x |
| GeGLU Approx | 0.81x | 0.72x | 1.38x | 0.77x | 1.43x |
| RoPE | 2.01x | 1.13x | 1.46x | 0.98x | 1.70x |
| CE (V=32K) | 1.56x | 1.33x | 1.67x | 2.63x | 2.00x |
| CE (V=128K) | 2.20x | 2.24x | 1.67x | 3.68x | 2.00x |

Values > 1.0 mean the kernel is faster (speed) or uses less memory (memory)
than the PyTorch baseline.

SwiGLU/GeGLU forward is slower than native PyTorch because
`autograd.Function.apply()` dispatch overhead (~30-50us) dominates the trivially
fast element-wise operation. The real value of these kernels is in the fused
backward pass and vmap memory savings.

### Fused linear cross-entropy benchmarks

| Metric | V=32K | V=128K |
|--------|-------|--------|
| Forward speedup | 8.73x | 9.46x |
| Forward memory | 2.85x | 3.19x |
| Backward speedup | 2.63x | 2.76x |
| Backward memory | 3.35x | 3.80x |
| vmap forward speedup | 8.86x | 8.88x |
| vmap forward memory | 12.10x | 22.67x |
| vmap(grad) speedup | 2.65x | 2.70x |
| vmap(grad) memory | 6.06x | 8.05x |

## Configuration

Opaque patches are controlled by environment variables. Each accepts `all`
to skip everything in that group, or a comma-separated list for selective skip.

| Variable | Scope | Values |
|----------|-------|--------|
| `OPAQUE_SKIP_COMPAT_PATCHES` | All patching | `all` |
| `OPAQUE_SKIP_TRANSFORMERS_PATCHES` | HF Transformers | `all`, or `vmap,kernels` |
| `OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES` | vmap compat | `all`, or `shared,standard,gemma2,phi3` |
| `OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES` | Triton kernels | `all`, or `swiglu,rope,ce,fused_ce,lora` |

### Disabling all patching

```python
import os
os.environ["OPAQUE_SKIP_COMPAT_PATCHES"] = "all"
import opaque
```

### Disabling kernel optimizations

```python
import os
os.environ["OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES"] = "all"
import opaque
```

This disables all kernel optimizations. The library still works — models use
standard PyTorch operations with vmap patches still applied.

### Selectively skipping kernels

```bash
# Skip only fused linear CE
OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=fused_ce python train.py

# Skip multiple
OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=swiglu,rope,fused_ce python train.py
```

Available kernel names: `swiglu`, `rope`, `ce`, `fused_ce`, `lora`.

### Disabling vmap patches

Vmap patches make HuggingFace models compatible with `vmap(grad())`. These
are required for DP-SGD training but can be skipped for debugging:

```bash
# Skip all vmap patches (models will NOT work with vmap)
OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=all python train.py

# Skip only Gemma2-specific patches
OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=gemma2 python train.py
```

Available vmap groups: `shared`, `standard`, `gemma2`, `phi3`.

### Using kernels directly

All kernels are available as standalone functions without patching:

```python
from opaque.compat.kernels import opaque_swiglu, opaque_cross_entropy_loss

# Direct usage
h = opaque_swiglu(gate, up)
loss = opaque_cross_entropy_loss(logits, labels)
```

## Troubleshooting

**Numerical differences after enabling patches:** All kernels are numerically
equivalent to PyTorch reference implementations within floating-point precision.
If you see significant differences, file a bug report.

**Performance regression with patches:** Some kernels (SwiGLU/GeGLU forward)
have higher dispatch overhead than native PyTorch for small tensors. The net
effect on end-to-end training is positive due to backward pass and memory
savings. Use `OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES` to isolate which kernel causes the issue.

**OOM with fused linear CE disabled:** Without fused linear CE, the full
`(batch*seq, vocab)` logits tensor is materialized. For 128K vocab models,
this uses ~2 GB per sample. Re-enable fused CE or reduce batch size.

**LoRA fusion not applied:** Auto-fusion requires all projections in a group
(Q+K+V or gate+up+down) to have LoRA adapters with no bias. Check your
`LoraConfig.target_modules` and model architecture.

## Implementation details

Source code: `packages/opaque/src/opaque/compat/kernels/`

| File | Lines | Description |
|------|-------|-------------|
| `swiglu.py` | 380 | SwiGLU forward + backward + vmap |
| `geglu.py` | 736 | GeGLU Exact and Approx variants |
| `rope_embedding.py` | 745 | RoPE with GQA support |
| `cross_entropy.py` | 478 | Chunked Triton cross-entropy |
| `linear_cross_entropy.py` | 962 | Fused linear + CE (ported from Apple CCE) |
| `lora.py` | 1295 | LoRA W, QKV, and MLP fusion |
| `utils.py` | 166 | Triton utilities and helpers |

Tests: `packages/opaque/tests/kernels/` — 5 test categories per kernel:
forward precision, backward precision, vmap forward, vmap(grad), performance.

For the detailed optimization plan and implementation history, see
`docs/development/MEMORY_OPTIMIZATION_PLAN.md`.

## API reference

See the [Kernels API](../api/index.md) in the API Reference for complete
function signatures.
