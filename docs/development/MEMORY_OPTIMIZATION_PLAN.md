# Memory Optimization Implementation Plan

## Goal
Improve memory efficiency of DP-SGD training on JetBrains Mellum-4b to enable:
- **Primary:** Train with all 7 LoRA modules (currently 2/7) for better model quality
- **Secondary:** Increase micro-batch size from 4 to 8-16 for faster training

## Current State
- **Model:** JetBrains/Mellum-4b-base (vocab=128,256)
- **Hardware:** H200 80GB GPU
- **Config:** micro-batch=4, seq_len=1024, bf16, LoRA r=16
- **LoRA:** Only q_proj + v_proj (2/7 modules) - memory constrained
- **Memory:** ~75 GB peak

## Target State
- **LoRA:** All 7 modules (q, k, v, o, gate, up, down) - better quality
- **Micro-batch:** 8-16 (2-4x throughput)
- **Memory:** ~65-70 GB peak (5-10 GB savings)
- **Speed:** 20-40% faster per step

---

## Architecture Principles

### Code Organization
Opaque uses a 2-layer architecture to stay framework-agnostic:

**Layer 1: Generic Operations** (`src/opaque/kernels/`)
- Pure Triton kernels and tensor operations
- Framework-agnostic (PyTorch, JAX, etc.)
- Reusable building blocks
- Example: `chunked_cross_entropy_loss(logits, labels)`

**Layer 2: Framework Integration** (`src/opaque/compat/transformers/`)
- HuggingFace model patching
- Monkey-patches model.forward() to use Layer 1 kernels
- Makes optimizations transparent (no train code changes)
- Example: Patches LlamaForCausalLM to use chunked CE automatically

**Constraint: Vmap Compatibility**
- Opaque uses `torch.func.vmap` for per-example gradients (DP-SGD requirement)
- Standard optimizations often break with vmap:
  - Gradient checkpointing (uses `requires_grad_()`)
  - Flash Attention (SDPA incompatible with vmap)
  - In-place operations (vmap needs functional purity)
- Our optimizations must use vmap-compatible patterns:
  - Triton kernels (operation-level fusion)
  - Functional transformations
  - `autograd.Function` with `Opaque_Foo.vmap()` + `_FooBackward.vmap()` pattern

### Autograd Pattern for vmap(grad())

All kernels follow the same two-level dispatch pattern (confirmed by ezyang, PyTorch #128020):

1. **`Opaque_Foo(autograd.Function)`** — main entry via `.apply()`
   - `forward()`: computes output
   - `setup_context()`: saves tensors for backward
   - `backward()`: delegates to `_FooBackward.apply()`
   - `vmap()`: Triton forward inline (not `.apply()`), returns `(result, bdim)`

2. **`_FooBackward(autograd.Function)`** — backward wrapper
   - `forward()`: runs Triton backward kernel
   - `vmap()`: Triton backward inline with merged batch dims

When `vmap(grad(fn))` runs, functorch intercepts `.apply()` calls and routes to the `vmap()` static methods, where tensors are regular (unwrapped) and Triton kernels work directly.

---

## Implementation Phases

### Phase 0: Baseline Measurement — COMPLETED

**Objective:** Establish reproducible baseline for comparison

Baseline established with Mellum-4b at batch=4, seq=1024, bf16, LoRA r=16.

---

### Phase 1: Fused Triton Kernels — COMPLETED

**Problem:**
- Standard ops create intermediate tensors
- RMSNorm: Creates variance tensor
- SwiGLU: Creates 2 intermediate activations
- 64 RMSNorm ops + 32 SwiGLU ops per forward = ~0.5 GB intermediates

**Implemented kernels** (`src/opaque/kernels/`):

| Kernel | File | Description |
|--------|------|-------------|
| SwiGLU | `swiglu.py` | `silu(gate) * up` — fused forward + backward, recomputes h in backward |
| GeGLU (exact) | `geglu.py` | `gelu(gate) * up` — erf-based GELU variant (Gemma) |
| GeGLU (approx) | `geglu.py` | `gelu_tanh(gate) * up` — tanh-based GELU variant (Gemma2) |
| RoPE | `rope_embedding.py` | Rotary position embeddings for Q+K, with GQA support |
| Cross-Entropy | `cross_entropy.py` | Chunked Triton CE for vocab up to 65536 |
| Utilities | `utils.py` | `calculate_settings`, `torch_gpu_device`, Triton helpers |

**Integration:** `_kernel_patches.py` patches at class level for all supported models.
Applied automatically at `import opaque` time. Disable with `OPAQUE_NO_KERNEL_PATCH=1`,
selectively skip with `OPAQUE_SKIP_PATCHES=swiglu,rope,...`.

**Supported models:** LLaMA, Mistral, Qwen2, Qwen3, Phi3, Gemma, Gemma2, Granite, Cohere, Cohere2.

**Test results** (105 tests, all pass at Mellum-4b scale: batch=4, seq=1024):

| Kernel | Forward speedup | Backward speedup | Memory reduction | vmap(grad) speedup | vmap(grad) memory |
|--------|-----------------|-------------------|------------------|--------------------|-------------------|
| SwiGLU | 0.69x | 0.83x | 1.20x | 1.19x | 2.10x |
| GeGLU Exact | 0.76x | 0.78x | 1.38x | 0.84x | 1.43x |
| GeGLU Approx | 0.81x | 0.72x | 1.38x | 0.77x | 1.43x |
| RoPE | 2.01x | 1.13x | 1.46x | 0.98x | 1.70x |
| CE (V=32K) | 1.56x | 1.33x | 1.67x | 2.63x | 2.00x |
| CE (V=128K) | 2.20x | 2.24x | 1.67x | 3.68x | 2.00x |

**Known limitation — element-wise kernel overhead:**
SwiGLU/GeGLU forward is slower than native PyTorch (`F.silu(gate)*up` = ~0.10ms) because `autograd.Function.apply()` dispatch overhead (~30-50us) dominates the trivially fast element-wise operation. This affects both the test and the real training path. The real value of these kernels is in the fused backward (reads 3, writes 2-3 in one kernel) and vmap (custom batching rules for DP-SGD). Increasing Triton BLOCK_SIZE does not help — the bottleneck is Python/autograd dispatch, not the GPU kernel.

---

### Phase 2: Fused LoRA Operations — COMPLETED

**Goal:** Enable training with all 7 LoRA modules efficiently

**Implemented kernels** (`src/opaque/kernels/lora.py`):

| Kernel | Description |
|--------|-------------|
| `Opaque_LoRA_W` | Single LoRA linear: `x @ W.T + x @ A @ B * s`. Avoids intermediate `x @ A` tensor. |
| `Opaque_LoRA_QKV` | Fused Q+K+V LoRA projection: shares `x` across 3 projections in one call. |
| `Opaque_LoRA_MLP` | Fused gate+up+down LoRA MLP: combines 3 projections + activation in one call. Uses SwiGLU/GeGLU Triton kernels internally via callbacks. |

**Integration:**
- `_kernel_patches.py` patches `peft.tuners.lora.Linear.forward` with `Opaque_LoRA_W`
- `get_peft_model()` hook auto-detects and fuses QKV (`Opaque_LoRA_QKV`) and MLP (`Opaque_LoRA_MLP`)
- `patch_lora_model()` public API for manual patching of pre-existing PEFT models

**Supported attention classes for QKV fusion:** LlamaAttention, MistralAttention, GemmaAttention, Gemma2Attention, GraniteAttention, Cohere2Attention. Excluded: Qwen2 (bias on Q/K/V), Qwen3 (q_norm/k_norm), Phi3 (combined qkv_proj), Cohere (no transpose).

**vmap backward for LoRA_W:** Uses per-sample `bmm` for weight gradients (not merged — correct for DP-SGD per-example grads).

---

### Phase 3: Chunked Cross-Entropy Loss — COMPLETED

**Problem:**
- Mellum has 128K vocab — loss computation uses ~1 GB
- Standard CE: allocate (batch*seq, vocab) = (4*1024, 128256) in memory
- Chunked CE: process vocab in chunks, only store logsumexp

**Implemented:** `src/opaque/kernels/cross_entropy.py`
- Triton forward+backward kernels with BLOCK_SIZE up to 65536
- `Opaque_CrossEntropyLoss` with vmap support
- Integrated via `LOSS_MAPPING` patch in `_kernel_patches.py`

**Results:** 1.6-2.2x forward speedup, 1.3-2.2x backward speedup, 1.67x memory reduction at Mellum-4b scale.

---

### Phase 3b: Fused Linear Cross-Entropy — COMPLETED

**Problem:**
- Standard CE: `logits = hidden @ lm_head.T` allocates [batch, seq, vocab] tensor
- For batch=4, seq=1024, vocab=128K: ~2 GB just for logits
- Apple's `cut_cross_entropy` (CCE) library has no vmap support — `autograd.Function` without `vmap()` static method
- Previous wrapper called CCE via Python API in a loop (12 kernel launches per step for B_vmap=4), causing 14% throughput degradation

**Solution:** Ported CCE Triton kernels into our codebase with native vmap support.

**Implemented:** `src/opaque/kernels/linear_cross_entropy.py` (846 lines)

Three Triton kernels (ported from Apple CCE, simplified):
- `_linear_ce_forward_kernel`: 2D tiled grid, tiled matmul E@C.T, per-block LSE with lock-based atomic `logaddexp`, batch grouping (GROUP_B=8)
- `_linear_ce_backward_kernel`: recomputes logits, computes CE gradient, lock-based `_mm_backward` for dE and dC accumulation
- `_mm_backward`: helper for lock-based tiled matmul gradient accumulation

Stripped from CCE (not needed): bias, logit_avg, gradient filtering, Kahan summation, vocab parallel, VocabOrdering, dLSE, shift.

**Key design decisions:**
1. **No shift in kernel** — pre-shift in Python (`h[..., :-1, :]`, `labels[..., 1:]`), so vmap merge is a trivial reshape
2. **Per-sample dC in kernel** — `sample_id = offs_b // tokens_per_sample` allows single kernel call for merged vmap batch, producing per-sample weight gradients
3. **Skip dC when weight frozen** — `ctx.needs_input_grad[1]` detects if `lm_head.weight` needs grad. In DP-SGD LoRA training, only LoRA params are trainable (`argnums=0`), so dC is skipped entirely (~1/3 of backward compute saved)
4. **Weight scaling in patch, not kernel** — Cohere (`weight * scale`), Granite (`weight / scale`), Gemma2 (`softcap` in kernel). Moving scaling outside the kernel ensures correct gradient chain for the original weight.

**Integration:** `_kernel_patches.py` replaces `ForCausalLM.forward()` for all 9 supported models. When labels present and hidden_states are bf16/fp16, computes loss directly without materializing logits. Falls back to standard CE for fp32.

**Test results** (28 tests, all pass):

| Test | V=32K | V=128K |
|------|-------|--------|
| Forward speedup | 8.73x | 9.46x |
| Forward memory | 2.85x | 3.19x |
| Backward speedup | 2.63x | 2.76x |
| Backward memory | 3.35x | 3.80x |
| vmap forward speedup | 8.86x | 8.88x |
| vmap forward memory | 12.10x | 22.67x |
| vmap(grad) speedup | 2.65x | 2.70x |
| vmap(grad) memory | 6.06x | 8.05x |

**Throughput progression** (10-step Mellum-4b training, samples/s):
- Without fused CE (materialized logits): 12.3 s/s
- Old CCE wrapper (vmap loop): 11.1 s/s
- Ported kernel + per-sample loop: 10.1 s/s
- Ported + per-sample dC in kernel: 10.8 s/s
- **Ported + skip dC (frozen weight): 11.8 s/s** (6.3% improvement over old CCE wrapper)
- Remaining gap to no-fused-CE baseline: 0.5 s/s (4%)

**External dependency removed:** `cut_cross_entropy` is no longer required at runtime.

---

### Phase 4: Dtype Precision Guards — NOT STARTED

**Problem:**
- PyTorch silently upcasts to fp32 in many ops
- Accidental fp32 gradients = 2x memory usage

**Memory Savings:** 0.5-2 GB (prevents accidental doubling)

**Tasks:**
1. Audit current dtypes in `clipped_fun.py`, `gaussian_noise.py`
2. Add explicit `dtype` parameter enforcement
3. Validate memory consistency

---

### Phase 5: CPU Offloading — NOT STARTED

**Research completed (Feb 2025):** Analysis of Unsloth's memory optimizations revealed that their 4-6x batch size improvement comes primarily from gradient checkpointing with CPU offloading (~50% of savings) — blocked by vmap.

**Applicable options:**
- **Frozen embedding offloading** (low risk): Move 128K vocab embeddings to CPU, ~1-2 GB savings, ~5-10% slower forward
- **Activation offloading for frozen layers** (medium risk): Offload base model activations to CPU via `autograd.Function` with vmap support

Both patterns confirmed working in vmap compatibility tests (Appendix B).

---

### Phase 6: Research — NOT STARTED

**High-risk explorations after Phase 5:**
- 4-bit quantization with vmap (bitsandbytes blocked, but simulated NF4 works)
- Hybrid checkpointing (standard checkpoint for embedding/output layers only)

---

## Testing Protocol

### Test Categories (per kernel)
1. **Forward precision** — numerical equivalence with PyTorch reference
2. **Backward precision** — gradient equivalence
3. **vmap forward** — Triton vmap vs PyTorch vmap
4. **vmap(grad)** — per-example gradients (the DP-SGD path)
5. **Performance** — speedup and/or memory reduction vs PyTorch

### Running Tests
```bash
# All kernel tests (105 tests)
uv run pytest packages/opaque/tests/kernels/ -v

# Specific kernel
uv run pytest packages/opaque/tests/kernels/test_linear_cross_entropy.py -v

# Training validation (10-step Mellum-4b)
PYTHONUNBUFFERED=1 uv run python examples/train_causal_lm.py \
  --preset mellum-kstack --max_steps 10 --wandb
```

### Disable/Skip Patches
```bash
# Disable all kernel patches
OPAQUE_NO_KERNEL_PATCH=1

# Selectively skip specific patches
OPAQUE_SKIP_PATCHES=fused_ce,swiglu,rope
```

---

## Rollback Plan

If issues arise:

**Numerical Issues:** Use `OPAQUE_SKIP_PATCHES=<kernel>` to disable specific patches

**Memory Regression:** Profile for leaks, check for fp32 upcasting

**Performance Regression:** Profile with `torch.profiler`, use skip flags to isolate

**Vmap Incompatibility:** Rewrite with vmap pattern, test standalone first

---

---

## Appendix A: Supported Models & Kernel Coverage

### Patched Components per Model

| Model | SwiGLU/GeGLU | RoPE | CE (LOSS_MAPPING) | Fused Linear CE | LoRA (auto-fuse) |
|-------|--------------|------|--------------------|-----------------|------------------|
| LLaMA | SwiGLU | Yes | Yes | Yes | QKV + MLP |
| Mistral | SwiGLU | Yes | Yes | Yes | QKV + MLP |
| Qwen2 | SwiGLU | Yes | Yes | Yes | MLP only (Q/K/V have bias) |
| Qwen3 | SwiGLU | - | Yes | Yes | MLP only (q_norm/k_norm) |
| Phi3 | SwiGLU (chunked) | Yes | Yes | - | - |
| Gemma | GeGLU Exact | Yes | Yes | Yes | QKV + MLP |
| Gemma2 | GeGLU Approx | Yes | Yes | Yes (softcap) | QKV + MLP |
| Granite | SwiGLU | Yes | Yes | Yes (div scaling) | QKV + MLP |
| Cohere | SwiGLU | - | Yes | Yes (mul scaling) | MLP only (no transpose) |
| Cohere2 | SwiGLU | - | Yes | Yes | QKV + MLP |

### Special Handling in Fused Linear CE
- **Gemma2**: `final_logit_softcapping` — `softcap * tanh(logits / softcap)` applied inside kernel
- **Granite**: `logits_scaling` (divisive) — `weight = weight / scale` applied before kernel
- **Cohere/Cohere2**: `logit_scale` (multiplicative) — `weight = weight * scale` applied before kernel
- All models: `bias=False`, `ignore_index=-100`, shift handled in Python

---

## Appendix B: Vmap Compatibility Test Results (Feb 2025)

### Test Summary

Ran comprehensive tests to validate which memory optimization techniques work with `torch.func.vmap`.
See `tests/research/test_memory_optimization_approaches*.py` for full test code.

### CONFIRMED WORKING

| Technique | Test | Result |
|-----------|------|--------|
| **CPU tensor in forward** | `test_cpu_tensor_in_forward` | Works |
| **Pinned memory transfer** | `test_pinned_memory_transfer` | Works |
| **Activation offload autograd.Function** | `test_activation_offload_autograd_function` | Works |
| **CUDA stream in autograd.Function** | `test_cuda_stream_in_autograd` | Works |
| **Non-blocking transfer in vmap** | `test_non_blocking_transfer_in_vmap` | Works |
| **Per-example grads to CPU** | `test_per_example_grad_to_cpu` | Works |
| **Chunked vmap accumulation** | `test_chunked_vmap_accumulation` | Works |
| **Embedding with CPU weight** | `test_embedding_indices_stay_on_gpu` | Works |
| **Frozen CPU + trainable GPU (LoRA pattern)** | `test_frozen_linear_on_cpu_trainable_on_gpu` | Works |
| **OffloadedEmbedding with hook** | `test_embedding_with_hook_for_offload` | Works |
| **Checkpoint with fixed weight** | `test_checkpoint_with_fixed_weight` | Works |
| **Layer recomputation** | `test_checkpoint_layer_recomputation` | Works |
| **Selective MLP checkpoint** | `test_selective_checkpoint_mlp_only` | Works |
| **CPU offload in backward (fixed)** | `test_cpu_offload_weight_stays_gpu` | Works |
| **Pinned memory offload backward** | `test_pinned_memory_offload_backward` | Works |
| **Fused linear+CE vectorized** | `test_fused_ce_vectorized` | Works |
| **Chunked logsumexp CE** | `test_chunked_logsumexp_ce` | Works |
| **Manual recompute MLP** | `test_manual_recompute_in_backward` | Works |
| **Checkpoint + offload combined** | `test_checkpoint_with_offload_combined` | Works |
| **Simulated NF4 dequant+matmul** | `test_simulated_nf4_with_vmap` | Works |
| **Dynamic int8 quantization** | `test_dynamic_quantization_forward_only` | Works |
| **torch.compile'd cross-entropy** | `test_torch_compile_cross_entropy` | Works |
| **Hybrid model (CPU frozen + GPU trainable)** | `test_model_with_cpu_frozen_layers` | Works |
| **Chunked batch processing** | `test_chunked_batch_processing` | Works |

### CONFIRMED NOT WORKING

| Technique | Test | Error |
|-----------|------|-------|
| `torch.utils.checkpoint(use_reentrant=True)` | `test_standard_checkpoint_reentrant` | Must override setup_context |
| `torch.utils.checkpoint(use_reentrant=False)` | `test_standard_checkpoint_non_reentrant` | `_NoopSaveInputs` no vmap support |
| `bitsandbytes.nn.Linear4bit` | `test_bitsandbytes_linear_4bit` | Must override setup_context |
| `bitsandbytes.nn.Linear8bitLt` | `test_bitsandbytes_linear_8bit` | Must override setup_context |
| `cut_cross_entropy.linear_cross_entropy` | `test_cut_cross_entropy_with_fixed_inputs` | Must override setup_context |

### Key Findings

1. **Manual checkpoint WORKS**: Using `torch.autograd.Function` with custom `vmap()` and manual recomputation in backward is vmap-compatible.

2. **CPU offloading WORKS**: Moving activations to CPU in `setup_context` and fetching in `backward` works with vmap.

3. **Fused CE WORKS (custom impl)**: We ported CCE Triton kernels with native vmap support. The original `cut_cross_entropy` library does NOT work (missing vmap support).

4. **Quantization PARTIAL**: bitsandbytes doesn't work, but simulated NF4/int8 dequantization works. Could implement custom quantized linear with vmap support.

5. **Combined techniques WORK**: Checkpoint + CPU offload can be combined in a single `autograd.Function`.

---

## References

- **Baseline config:** `examples/train_causal_lm.py` (mellum-kstack preset)
- **Kernel sources:** `packages/opaque/src/opaque/kernels/`
- **Kernel patches:** `packages/opaque/src/opaque/compat/transformers/_kernel_patches.py`
- **Tests:** `packages/opaque/tests/kernels/` (105 tests)
- **PyTorch vmap+autograd pattern:** PyTorch #128020 (ezyang)
- **Apple CCE paper:** "Linear Cross-Entropy Loss" (ICLR 2025)
