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
  - ❌ Gradient checkpointing (uses `requires_grad_()`)
  - ❌ Flash Attention (SDPA incompatible with vmap)
  - ❌ In-place operations (vmap needs functional purity)
- Our optimizations must use vmap-compatible patterns:
  - ✅ Triton kernels (operation-level fusion)
  - ✅ Functional transformations
  - ✅ New autograd.Function pattern (`setup_context`, `generate_vmap_rule`)

---

## Implementation Phases

### Phase 0: Baseline Measurement (1 day)

**Objective:** Establish reproducible baseline for comparison

**Tasks:**
1. Run mellum-kstack preset with memory profiling
2. Record peak memory, step time, loss trajectory
3. Use `find_max_microbatch_size()` to confirm limits
4. Document memory breakdown (model, activations, gradients, optimizer)

**Command:**
```bash
python examples/train_causal_lm.py \
  --preset mellum-kstack \
  --num_train_samples 1000 \
  --max_steps 100 \
  --log_steps 10
```

**Deliverable:** `baseline_measurements.md` with detailed metrics

---

### Phase 1: Chunked Cross-Entropy Loss (3-5 days)

**Problem:**
- Mellum has 128K vocab → loss computation uses ~1 GB
- Standard CE: allocate (batch×seq, vocab) = (4×1024, 128256) in memory
- Chunked CE: process vocab in 4 chunks, only store logsumexp

**Memory Savings:** ~0.5-1 GB

**Implementation:**

**Step 1.1: Generic Kernel** (1 day)
- Create: `src/opaque/kernels/__init__.py`
- Create: `src/opaque/kernels/cross_entropy.py`
- Adapt from unsloth: `/tmp/unsloth/unsloth/kernels/cross_entropy_loss.py`
- Components:
  - `_chunked_cross_entropy_forward_kernel` (Triton)
  - `_chunked_cross_entropy_backward_kernel` (Triton)
  - `ChunkedCrossEntropyFunction` (autograd.Function with vmap support)
  - `chunked_cross_entropy_loss()` (public API)
- Test: Numerical equivalence with `F.cross_entropy`

**Step 1.2: HuggingFace Integration** (2 days)
- Create: `src/opaque/compat/transformers/_loss_patches.py`
- Function: `apply_chunked_loss_patches(n_chunks=4)`
- Strategy: Monkey-patch model.forward() to use chunked loss internally
- Preserve API: `output = model(input_ids, labels=labels); loss = output.loss`
- Integrate: Call from `_global_patches.py`

**Step 1.3: Test & Benchmark** (1 day)
- Test loss equivalence (atol=1e-4)
- Benchmark memory savings
- Measure speed overhead (expect 5-10% slower)
- Document results

**Success Criteria:**
- Memory reduction: 0.5-1 GB
- Loss values match baseline
- No API changes needed in train_causal_lm.py

---

### Phase 2: Fused Triton Kernels (5-7 days)

**Problem:**
- Standard ops create intermediate tensors
- RMSNorm: Creates variance tensor
- SwiGLU: Creates 2 intermediate activations
- 64 RMSNorm ops + 32 SwiGLU ops per forward = ~0.5 GB intermediates

**Memory Savings:** ~0.5-1 GB
**Speed Improvement:** 20-30% faster ops

**Step 2.1: RMSLayerNorm** (2 days)
- Create: `src/opaque/kernels/rms_layernorm.py`
- Source: `/tmp/unsloth/unsloth/kernels/rms_layernorm.py`
- Single-pass computation, stores only inverse RMS for backward
- Vmap-compatible autograd.Function

**Step 2.2: SwiGLU** (2 days)
- Create: `src/opaque/kernels/swiglu.py`
- Source: `/tmp/unsloth/unsloth/kernels/swiglu.py`
- Fuses: `output = gate_proj * sigmoid(gate_proj) * up_proj`
- In-place computation, no intermediates

**Step 2.3: Integration** (1 day)
- Create: `src/opaque/compat/transformers/_kernel_patches.py`
- Patch model layers after loading
- Test full integration

**Success Criteria:**
- Numerical equivalence
- 1.5-2x faster individual ops
- 20-30% faster overall training
- Memory reduction: 0.5-1 GB

---

### Phase 3: Dtype Precision Guards (2-3 days)

**Problem:**
- PyTorch silently upcasts to fp32 in many ops
- Accidental fp32 gradients = 2x memory usage
- Must enforce bf16 throughout gradient computation

**Memory Savings:** 0.5-2 GB (prevents accidental doubling)

**Step 3.1: Audit Current Dtypes** (1 day)
- Check: `src/opaque/clipping/clipped_fun.py`
- Check: `src/opaque/noise/gaussian_noise.py`
- Create audit script to detect fp32 tensors

**Step 3.2: Add Dtype Guards** (1 day)
- Modify: `clipped_fun.py` gradient accumulation
- Add explicit `dtype` parameter enforcement
- Ensure: `torch.sum(..., dtype=bf16)` not default fp32

**Step 3.3: Validate** (1 day)
- Assert all gradients stay bf16
- Test memory consistency
- Document findings

**Success Criteria:**
- No fp32 gradients during training
- Memory stays consistent
- No silent upcasting detected

---

### Phase 4: Fused LoRA Operations (7-10 days)

**Goal:** Enable training with all 7 LoRA modules efficiently

**Problem:**
- Standard LoRA: `x @ base.T + x @ lora_A.T @ lora_B.T * scale`
- Creates intermediate tensor: `lora_result = x @ lora_A.T @ lora_B.T`
- For 7 modules × 32 layers: ~0.5-1 GB intermediates

**Memory Savings:** 0.5-1 GB
**Enables:** Full 7-module LoRA (only ~60-110 MB net cost)

**Step 4.1: Study Unsloth Implementation** (1 day)
- Read: `/tmp/unsloth/unsloth/kernels/fast_lora.py`
- Document: How fusion eliminates intermediates
- Note: Must adapt to vmap-compatible pattern

**Step 4.2: Design Vmap Pattern** (2 days)
- Create design doc: `lora_fusion_design.md`
- Key: Use new autograd.Function pattern
  - `generate_vmap_rule = True`
  - `setup_context()` staticmethod
  - Batched transposes: `.transpose(-2, -1)` not `.t()`

**Step 4.3: Implement** (3 days)
- Create: `src/opaque/kernels/fused_lora.py`
- Class: `FusedLoRALinear` (autograd.Function)
- Test: Forward equivalence, backward correctness, vmap compatibility

**Step 4.4: Integrate** (1-2 days)
- Hook into model creation pipeline
- Replace standard LoRA layers with fused versions
- Update: `examples/train_causal_lm.py` to support full LoRA

**Step 4.5: Full Test** (1 day)
- Test with all 7 LoRA modules
- Verify per-example gradients correct
- Benchmark memory and speed

**Success Criteria:**
- Can train with all 7 LoRA modules
- Memory increase: <110 MB (vs. 2/7 baseline)
- Speed improvement: 10-20%
- Vmap compatibility confirmed

---

### Phase 5: CPU Offloading (Medium Risk, 1-2 weeks)

**Research Summary (Feb 2025):**
Analysis of Unsloth's memory optimizations revealed that their 4-6x batch size improvement
(batch 4 → 16-24 on Qwen 7B) comes primarily from:
1. **Gradient checkpointing with CPU offloading** (~50% of savings) - ❌ Blocked by vmap
2. **Fused linear + cross-entropy** (~30%) - ⚠️ Partially applicable
3. **Embedding offloading** (~10%) - ✅ Applicable
4. **Triton kernels** (~10%) - ✅ Already implemented (Phases 1-2)

**Key Insight:** While full gradient checkpointing is blocked by vmap (uses `requires_grad_()`
mutations), several CPU offloading techniques ARE compatible with our architecture.

---

#### Option A: Frozen Embedding Offloading (Low Risk, 2-3 days)

**Problem:**
- Large vocab models have massive embedding tables (Mellum 128K vocab = ~1 GB)
- In LoRA training, embeddings are frozen but still consume GPU memory
- `lm_head` (output projection) often shares weights with embeddings

**Unsloth Pattern** (from `unsloth/models/llama.py:2743-3001`):
```python
# Offload frozen embeddings to CPU, keep trainable LoRA copy on GPU
def _offload_frozen_module_for_training(module, device_type, offload_device="cpu"):
    module.modules_to_save.default.to(device=device_type, non_blocking=True)
    module.original_module.to(device=offload_device, non_blocking=True)
```

**Implementation for Opaque:**
```python
# In src/opaque/compat/transformers/_embedding_offload.py
def offload_frozen_embeddings(model):
    """Offload frozen embed_tokens and lm_head to CPU."""
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    if not embed.weight.requires_grad:
        embed.weight.data = embed.weight.data.to("cpu", non_blocking=True)
    if lm_head is not None and not lm_head.weight.requires_grad:
        lm_head.weight.data = lm_head.weight.data.to("cpu", non_blocking=True)
```

**Vmap Compatibility:** ✅ Safe - embeddings are frozen, no per-example gradients needed

**Memory Savings:** ~1-2 GB (depends on vocab size)
**Speed Impact:** ~5-10% slower forward (CPU→GPU transfer per batch)
**Risk:** Low

---

#### Option B: Per-Example Gradient CPU Staging (Medium Risk, 1 week)

**Problem:**
- DP-SGD computes per-example gradients: `batch_size` copies of each gradient
- For batch=8, this is 8x the memory of standard training gradients
- Peak memory occurs when all per-example grads are on GPU before clipping

**Proposed Pattern:**
```python
# In src/opaque/clipping/clipped_fun.py - modified accumulation
def _microbatch_accumulate_with_offload(...):
    """Process microbatches, offloading per-example grads to pinned CPU memory."""

    # Pre-allocate pinned CPU buffers (like Unsloth's CPU_BUFFERS)
    cpu_grad_buffers = [
        torch.empty_like(p, device="cpu", pin_memory=True)
        for p in parameters if p.requires_grad
    ]

    for microbatch in microbatches:
        # Compute per-example grads on GPU
        per_example_grads = vmap(grad(loss_fn))(microbatch)

        # Immediately offload to pinned CPU (non-blocking hides latency)
        for cpu_buf, gpu_grad in zip(cpu_grad_buffers, per_example_grads):
            cpu_buf.copy_(gpu_grad, non_blocking=True)

        # Clip on CPU (surprisingly fast for small tensors)
        clipped = clip_per_example(cpu_grad_buffers, max_norm)

        # Accumulate clipped grads back to GPU
        for param, clipped_grad in zip(parameters, clipped):
            param.grad.add_(clipped_grad.to(param.device, non_blocking=True))
```

**Key Techniques from Unsloth:**
1. **Pinned memory** (`pin_memory=True`): Enables async CPU↔GPU transfer
2. **Non-blocking transfers**: `tensor.to(device, non_blocking=True)` overlaps with compute
3. **CUDA streams**: Separate stream for transfers to hide latency
4. **Buffer pooling**: Reuse pre-allocated buffers to avoid allocation overhead

**Vmap Compatibility:** ✅ Safe - offloading happens AFTER vmap computation

**Memory Savings:** ~3-5 GB (moves per-example grads off GPU during clipping)
**Speed Impact:** ~10-20% slower (CPU↔GPU transfers)
**Risk:** Medium (timing/synchronization complexity)

---

#### Option C: Activation Offloading for Frozen Layers (Medium Risk, 1-2 weeks)

**Problem:**
- In LoRA training, base model layers are frozen but activations still stored
- These activations are only needed for LoRA gradient computation
- Could offload to CPU and fetch back during backward

**Key Insight:** Unlike gradient checkpointing, this doesn't use `requires_grad_()`:
```python
class OffloadedActivation(torch.autograd.Function):
    generate_vmap_rule = True  # Vmap compatible!

    @staticmethod
    def forward(x):
        # Save to pinned CPU memory
        cpu_x = x.to("cpu", non_blocking=True)
        return x  # Pass through on GPU

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.cpu_activation = inputs[0].to("cpu", non_blocking=True)
        ctx.device = inputs[0].device

    @staticmethod
    def backward(ctx, grad_output):
        # Fetch activation back from CPU for backward
        x = ctx.cpu_activation.to(ctx.device, non_blocking=True)
        # ... compute gradient using x ...
        return grad_output
```

**Vmap Compatibility:** ⚠️ Needs testing - `generate_vmap_rule=True` should work
but CPU tensor handling in vmap backward is uncharted territory

**Memory Savings:** ~2-4 GB (layer activations)
**Speed Impact:** ~15-25% slower
**Risk:** Medium-High (vmap + CPU tensor interaction unknown)

---

#### Option D: Fused Linear Cross-Entropy (Low Risk, 3-5 days)

**Problem:**
- Standard CE: `logits = hidden @ lm_head.T` allocates [batch, seq, vocab] tensor
- For batch=8, seq=1024, vocab=128K: ~4 GB just for logits!

**Unsloth Solution:** Use `cut_cross_entropy` library
```python
from cut_cross_entropy import linear_cross_entropy

# Never materializes full logits tensor
loss = linear_cross_entropy(
    hidden_states,      # [batch, seq, hidden]
    lm_head.weight,     # [vocab, hidden]
    labels,             # [batch, seq]
    shift=True,
    reduction="mean"
)
```

**For DP-SGD:** Need to verify vmap compatibility
```python
# Test if this works with vmap
per_example_loss = vmap(
    lambda h, l: linear_cross_entropy(h.unsqueeze(0), lm_head.weight, l.unsqueeze(0))
)(hidden_states, labels)
```

**Vmap Compatibility:** ⚠️ Unknown - `cut_cross_entropy` may use custom CUDA kernels

**Memory Savings:** ~2-4 GB (no full logits tensor)
**Speed Impact:** +10-20% faster (fused kernel)
**Risk:** Low-Medium (need to test vmap compatibility)

---

#### Phase 5 Implementation Priority

| Option | Memory Savings | Speed Impact | Vmap Safe | Risk | Priority |
|--------|---------------|--------------|-----------|------|----------|
| A: Embedding Offload | 1-2 GB | -5-10% | ✅ Yes | Low | **1st** |
| D: Fused CE | 2-4 GB | +10-20% | ⚠️ Test | Low-Med | **2nd** |
| B: Grad CPU Staging | 3-5 GB | -10-20% | ✅ Yes | Medium | **3rd** |
| C: Activation Offload | 2-4 GB | -15-25% | ⚠️ Test | Med-High | **4th** |

**Recommended Approach:**
1. Start with Option A (embedding offload) - guaranteed safe, quick win
2. Test Option D (fused CE) - high potential, needs vmap validation
3. If more memory needed, implement Option B (grad staging)
4. Option C is research/experimental

---

#### Unsloth Reference Implementation

**Key files analyzed:**
- `unsloth-zoo/unsloth_zoo/gradient_checkpointing.py`: Smart CPU offloading with CUDA streams
- `unsloth/models/llama.py:154-204`: `_offload_frozen_module_for_training()`
- `unsloth-zoo/unsloth_zoo/loss_utils.py`: Fused CE integration
- `unsloth-zoo/unsloth_zoo/fused_losses/cross_entropy_loss.py`: Chunked loss with `torch.func.grad_and_value`

**Unsloth's CPU Buffer Pattern:**
```python
# From gradient_checkpointing.py - reusable buffer pool
CPU_BUFFERS = []
for i in range(200):
    x = torch.empty(128*1024, dtype=dtype, device="cpu", pin_memory=True)
    CPU_BUFFERS.append(x)

# Async transfer with CUDA streams
EXTRA_STREAM.wait_stream(MAIN_STREAM)
with torch.cuda.stream(EXTRA_STREAM):
    cpu_buffer.copy_(gpu_tensor, non_blocking=True)
```

---

### Phase 6: Research (Optional, 2-4 weeks)

**High-risk explorations after Phase 5:**

**Option A: 4-bit Quantization with Vmap** (1-2 weeks)
- Test if bitsandbytes NF4 works with vmap
- Potential: ~10 GB savings
- Risk: May be fundamentally incompatible

**Option B: Hybrid Checkpointing** (2-4 weeks)
- Apply standard checkpointing to embedding/output layers only (no vmap needed)
- Keep vmap-compatible path for transformer blocks
- Potential: 1-2 GB additional savings
- Risk: Architecture complexity

---

## Testing Protocol

After each phase:

### 1. Memory Profile
```bash
python examples/train_causal_lm.py \
  --preset mellum-kstack \
  --max_steps 20 \
  --profile_memory
```
Check: Peak memory, breakdown by component, no leaks

### 2. Numerical Validation
```bash
# Baseline
python examples/train_causal_lm.py --preset mellum-kstack --seed 42 --max_steps 50 > baseline.txt

# Optimized
python examples/train_causal_lm.py --preset mellum-kstack --seed 42 --max_steps 50 --use_optimizations > optimized.txt

# Compare
python scripts/compare_loss_curves.py baseline.txt optimized.txt
```
Tolerance: Loss values within 1e-3

### 3. Performance Benchmark
- Time 100 training steps
- Report steps/second
- Compare to baseline

### 4. Gradient Correctness
- Compute gradients individually per example
- Compute gradients with vmap
- Verify equivalence (atol=1e-5)

---

## Expected Results

| Phase | Memory Saved | Speed Change | Duration | Risk |
|-------|--------------|--------------|----------|------|
| 0: Baseline | 0 GB | 0% | 1 day | None |
| 1: Chunked CE | 0.5-1 GB | -5 to -10% | 3-5 days | Low |
| 2: Triton Kernels | 0.5-1 GB | +20-30% | 5-7 days | Low |
| 3: Dtype Guards | 0.5-2 GB | 0% | 2-3 days | Low |
| 4: Fused LoRA | 0.5-1 GB | +10-20% | 7-10 days | Medium |
| **Total (1-4)** | **2-5 GB** | **+20-40%** | **18-28 days** | **Low** |
| 5A: Embedding Offload | 1-2 GB | -5-10% | 2-3 days | Low |
| 5B: Grad CPU Staging | 3-5 GB | -10-20% | 1 week | Medium |
| 5C: Activation Offload | 2-4 GB | -15-25% | 1-2 weeks | Med-High |
| 5D: Fused CE | 2-4 GB | +10-20% | 3-5 days | Low-Med |
| **Total (1-5)** | **8-16 GB** | **+10-30%** | **4-6 weeks** | **Medium** |

### Milestone Achievements

**After Phase 1-3:**
- Memory: ~72 GB (3 GB saved)
- Speed: 20-30% faster
- LoRA: Still 2/7 (foundation ready)

**After Phase 4:**
- Memory: ~70 GB (5 GB saved)
- Speed: 30-40% faster
- **LoRA: 7/7 modules (KEY UNLOCK)**
- **Model quality: Significantly improved**

**After Phase 5 (CPU Offloading):**
- Memory: ~60-65 GB (10-15 GB saved)
- Speed: 20-30% faster (kernels offset offloading overhead)
- **Micro-batch: 8-16 (2-4x throughput)**
- **Approaches Unsloth-level memory efficiency for DP-SGD**

**Stretch (with Phase 6):**
- Memory: ~55-60 GB (15-20 GB saved)
- Micro-batch: 16-24 (4-6x throughput)

---

## Rollback Plan

If issues arise:

**Numerical Issues:** Add `--disable_<optimization>` flag, revert to baseline

**Memory Regression:** Profile for leaks, check for fp32 upcasting

**Performance Regression:** Profile with `torch.profiler`, consider making optional

**Vmap Incompatibility:** Rewrite with vmap pattern, test standalone first

---

## Key Implementation Details

### Vmap-Compatible autograd.Function Pattern

```python
class VmapCompatibleFunction(torch.autograd.Function):
    generate_vmap_rule = True  # Critical for vmap

    @staticmethod
    def forward(x, weight):  # No ctx parameter
        return x @ weight.t()

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Setup context - called after forward"""
        x, weight = inputs
        ctx.save_for_backward(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = grad_output @ weight
        grad_weight = grad_output.t() @ x
        return grad_x, grad_weight
```

### Loss Patching Pattern

```python
# Layer 1: Generic kernel (opaque.kernels.cross_entropy)
def chunked_cross_entropy_loss(logits, labels, n_chunks=4):
    """Generic function - works on any tensors"""
    return ChunkedCrossEntropyFunction.apply(logits, labels, n_chunks, -100)

# Layer 2: HF integration (opaque.compat.transformers._loss_patches)
def apply_chunked_loss_patches(model_class):
    original_forward = model_class.forward

    def forward_with_chunked_loss(self, input_ids=None, labels=None, **kwargs):
        outputs = original_forward(self, input_ids=input_ids, labels=None, **kwargs)
        if labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            outputs.loss = chunked_cross_entropy_loss(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                n_chunks=4
            )
        return outputs

    model_class.forward = forward_with_chunked_loss
```

---

## Appendix A: Unsloth Supported Models & Optimizations

### Supported Model Architectures

| Architecture | Class | Models |
|--------------|-------|--------|
| **LLaMA** | `FastLlamaModel` | LLaMA-2 (7B, 13B), LLaMA-3 (8B, 70B), LLaMA-3.1 (8B, 70B, 405B), LLaMA-3.2 (1B, 3B, 11B-Vision, 90B-Vision), TinyLlama, CodeLlama, Yi |
| **Mistral** | `FastMistralModel` | Mistral-7B (v0.1-v0.3), Mistral-Nemo-12B, Mistral-Large, Mixtral-8x7B, Codestral-22B |
| **Qwen** | `FastQwen2Model`, `FastQwen3Model`, `FastQwen3MoeModel` | Qwen2 (0.5B-72B), Qwen2.5, Qwen3, Qwen3-MoE |
| **Gemma** | `FastGemmaModel`, `FastGemma2Model` | Gemma (2B, 7B), Gemma-2 (2B, 9B, 27B), CodeGemma, Gemma-3, Gemma-3n |
| **Phi** | via `FastLlamaModel` | Phi-3-mini, Phi-3-medium, Phi-3.5 |
| **Cohere** | `FastCohereModel` | Command-R, Command-R+, Cohere2 |
| **Granite** | `FastGraniteModel` | IBM Granite models |
| **Falcon** | `FastFalconH1Model` | Falcon-H1 |
| **GLM** | `FastGLM47Model` | GLM-4-MoE |
| **DeepSeek** | via registry | DeepSeek models |
| **Vision** | `FastVisionModel` | Pixtral, LLaVA-Next, Aya-Vision |

### Unsloth Optimization Techniques

#### 1. Triton Kernels (`unsloth/kernels/`)

| Kernel | File | Description |
|--------|------|-------------|
| Cross-Entropy | `cross_entropy_loss.py` | Chunked CE avoiding full logits materialization |
| RMS LayerNorm | `rms_layernorm.py` | Fused single-pass normalization |
| LayerNorm | `layernorm.py` | Standard fused layernorm |
| RoPE | `rope_embedding.py` | Fused rotary position embeddings |
| SwiGLU | `swiglu.py` | Fused gate×sigmoid×up projection |
| GeGLU | `geglu.py` | Fused GELU gate activation |
| Fast LoRA | `fast_lora.py` | Fused QKV/MLP LoRA computation |
| FP8 | `fp8.py` | 8-bit floating point support |
| Flex Attention | `flex_attention.py` | Optimized attention with softcapping |

#### 2. Gradient Checkpointing with CPU Offloading (`unsloth-zoo/gradient_checkpointing.py`)

```python
# Key pattern: Async CPU offloading with CUDA streams
class UnslothCheckpointFunction(torch.autograd.Function):
    def forward(ctx, forward_function, hidden_states, *args):
        # Save to pinned CPU memory (non-blocking hides latency)
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        return output

    def backward(ctx, dY):
        # Fetch back from CPU for recomputation
        hidden_states = ctx.saved_tensors[0].to(device, non_blocking=True)
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
```

**Key techniques:**
- **Pinned memory**: `torch.empty(..., pin_memory=True)` for fast DMA transfers
- **CUDA streams**: Separate stream for CPU↔GPU transfers to overlap with compute
- **Buffer pooling**: Pre-allocated 200 buffers (128KB each) to avoid allocation overhead
- **Selective checkpointing**: Skip last layer for better VRAM/speed tradeoff

#### 3. Fused Linear Cross-Entropy (`unsloth-zoo/loss_utils.py`)

```python
# Uses cut_cross_entropy library - never materializes full logits
from cut_cross_entropy import linear_cross_entropy

loss = linear_cross_entropy(
    hidden_states,      # [batch, seq, hidden]
    lm_head.weight,     # [vocab, hidden]
    labels,             # [batch, seq]
    shift=True,
    reduction="mean"
)
```

#### 4. Embedding Offloading (`unsloth/models/llama.py:2743-2760`)

```python
def _offload_frozen_module_for_training(module, device_type, offload_device="cpu"):
    # Keep trainable LoRA copy on GPU
    module.modules_to_save.default.to(device=device_type, non_blocking=True)
    # Move frozen original to CPU
    module.original_module.to(device=offload_device, non_blocking=True)
```

#### 5. Model Patching (`FastLlamaModel.pre_patch()`)

- Replace `LlamaAttention.forward` → `LlamaAttention_fast_forward`
- Replace `LlamaDecoderLayer.forward` → `LlamaDecoderLayer_fast_forward`
- Replace `LlamaModel.forward` → `LlamaModel_fast_forward`
- Optimized KV cache handling with incremental updates

#### 6. Mixed Precision & Quantization

- 4-bit QLoRA (bitsandbytes NF4)
- 8-bit LoRA
- FP8 training (`load_in_fp8='block'` or `'row'`)
- Automatic bf16/fp16 selection based on GPU capability

### Memory Optimization Impact Summary

| Technique | Memory Savings | Speed Impact | Vmap Compatible |
|-----------|---------------|--------------|-----------------|
| Gradient Checkpointing + CPU Offload | ~5-10 GB | -10-15% | ⚠️ **Partial** (see below) |
| Fused Linear CE | ~2-4 GB | +10-20% | ✅ **Yes** (custom impl) |
| Embedding Offloading | ~1-2 GB | -5-10% | ✅ Yes |
| Triton Kernels | ~0.5-1 GB | +20-30% | ✅ Partially |
| 4-bit Quantization | ~10 GB | -5% | ⚠️ **Partial** (see below) |

### Opaque Compatibility Matrix (Updated with Test Results)

| Technique | Applicable | Status | Notes |
|-----------|------------|--------|-------|
| RMS LayerNorm kernel | ✅ | Ported | `opaque/kernels/rms_layernorm.py` |
| SwiGLU/GeGLU kernels | ✅ | Ported | `opaque/kernels/swiglu.py`, `geglu.py` |
| RoPE kernel | ✅ | Ported | `opaque/kernels/rope_embedding.py` |
| Chunked Cross-Entropy | ✅ | Ported | `opaque/kernels/cross_entropy.py` |
| LoRA fusion | ✅ | Ported | `opaque/kernels/lora.py` |
| Embedding offloading | ✅ | **Tested** | Frozen weights on CPU ✅ |
| Gradient CPU staging | ✅ | **Tested** | Post-vmap offloading ✅ |
| Fused Linear CE | ✅ | **Tested** | Custom chunked impl ✅ (`cut_cross_entropy` ❌) |
| Manual checkpoint (recompute) | ✅ | **Tested** | `generate_vmap_rule=True` pattern ✅ |
| Checkpoint + CPU offload combined | ✅ | **Tested** | Both work together ✅ |
| Standard `torch.utils.checkpoint` | ❌ | Blocked | Uses `requires_grad_()` mutations |
| bitsandbytes 4-bit | ❌ | Blocked | No vmap support in autograd.Function |
| Simulated NF4 dequant | ✅ | **Tested** | Manual dequant+matmul works ✅ |

---

## Appendix B: Vmap Compatibility Test Results (Feb 2025)

### Test Summary

Ran comprehensive tests to validate which memory optimization techniques work with `torch.func.vmap`.
See `tests/research/test_memory_optimization_approaches*.py` for full test code.

### ✅ CONFIRMED WORKING

| Technique | Test | Result |
|-----------|------|--------|
| **CPU tensor in forward** | `test_cpu_tensor_in_forward` | ✅ Works |
| **Pinned memory transfer** | `test_pinned_memory_transfer` | ✅ Works |
| **Activation offload autograd.Function** | `test_activation_offload_autograd_function` | ✅ Works |
| **CUDA stream in autograd.Function** | `test_cuda_stream_in_autograd` | ✅ Works |
| **Non-blocking transfer in vmap** | `test_non_blocking_transfer_in_vmap` | ✅ Works |
| **Per-example grads to CPU** | `test_per_example_grad_to_cpu` | ✅ Works |
| **Chunked vmap accumulation** | `test_chunked_vmap_accumulation` | ✅ Works |
| **Embedding with CPU weight** | `test_embedding_indices_stay_on_gpu` | ✅ Works |
| **Frozen CPU + trainable GPU (LoRA pattern)** | `test_frozen_linear_on_cpu_trainable_on_gpu` | ✅ Works |
| **OffloadedEmbedding with hook** | `test_embedding_with_hook_for_offload` | ✅ Works |
| **Checkpoint with fixed weight** | `test_checkpoint_with_fixed_weight` | ✅ Works |
| **Layer recomputation** | `test_checkpoint_layer_recomputation` | ✅ Works |
| **Selective MLP checkpoint** | `test_selective_checkpoint_mlp_only` | ✅ Works |
| **CPU offload in backward (fixed)** | `test_cpu_offload_weight_stays_gpu` | ✅ Works |
| **Pinned memory offload backward** | `test_pinned_memory_offload_backward` | ✅ Works |
| **Fused linear+CE vectorized** | `test_fused_ce_vectorized` | ✅ Works |
| **Chunked logsumexp CE** | `test_chunked_logsumexp_ce` | ✅ Works |
| **Manual recompute MLP** | `test_manual_recompute_in_backward` | ✅ Works |
| **Checkpoint + offload combined** | `test_checkpoint_with_offload_combined` | ✅ Works |
| **Simulated NF4 dequant+matmul** | `test_simulated_nf4_with_vmap` | ✅ Works |
| **Dynamic int8 quantization** | `test_dynamic_quantization_forward_only` | ✅ Works |
| **torch.compile'd cross-entropy** | `test_torch_compile_cross_entropy` | ✅ Works |
| **Hybrid model (CPU frozen + GPU trainable)** | `test_model_with_cpu_frozen_layers` | ✅ Works |
| **Chunked batch processing** | `test_chunked_batch_processing` | ✅ Works |

### ❌ CONFIRMED NOT WORKING

| Technique | Test | Error |
|-----------|------|-------|
| `torch.utils.checkpoint(use_reentrant=True)` | `test_standard_checkpoint_reentrant` | Must override setup_context |
| `torch.utils.checkpoint(use_reentrant=False)` | `test_standard_checkpoint_non_reentrant` | `_NoopSaveInputs` no vmap support |
| `bitsandbytes.nn.Linear4bit` | `test_bitsandbytes_linear_4bit` | Must override setup_context |
| `bitsandbytes.nn.Linear8bitLt` | `test_bitsandbytes_linear_8bit` | Must override setup_context |
| `cut_cross_entropy.linear_cross_entropy` | `test_cut_cross_entropy_with_fixed_inputs` | Must override setup_context |

### Key Findings

1. **Manual checkpoint WORKS**: Using `torch.autograd.Function` with `generate_vmap_rule=True`
   and manual recomputation in backward is vmap-compatible. This provides checkpoint-like
   memory savings without using `torch.utils.checkpoint`.

2. **CPU offloading WORKS**: Moving activations to CPU in `setup_context` and fetching in
   `backward` works with vmap. Requires proper device handling.

3. **Fused CE WORKS (custom impl)**: Custom chunked logsumexp implementation works with vmap.
   The `cut_cross_entropy` library does NOT work (missing vmap support).

4. **Quantization PARTIAL**: bitsandbytes doesn't work, but simulated NF4/int8 dequantization
   works. Could implement custom quantized linear with vmap support.

5. **Combined techniques WORK**: Checkpoint + CPU offload can be combined in single
   `autograd.Function` with `generate_vmap_rule=True`.

### Recommended Implementation Pattern

```python
class VmapCheckpointWithOffload(torch.autograd.Function):
    """
    Combines activation checkpointing with CPU offloading.
    Vmap-compatible via generate_vmap_rule=True.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(x, *layer_weights):
        # Compute forward, don't save intermediates
        h = layer_forward(x, layer_weights)
        return h

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, *layer_weights = inputs
        # Offload input to CPU (saves GPU memory)
        ctx.x_cpu = x.detach().to("cpu", non_blocking=True)
        ctx.layer_weights = layer_weights

    @staticmethod
    def backward(ctx, grad_output):
        # Fetch input from CPU
        x = ctx.x_cpu.to(grad_output.device, non_blocking=True)

        # Recompute intermediate activations
        h = layer_forward_with_intermediates(x, ctx.layer_weights)

        # Compute gradients
        grad_x = backward_through_layer(grad_output, h, ctx.layer_weights)
        return (grad_x,) + (None,) * len(ctx.layer_weights)
```

---

## References

- **Baseline config:** `examples/train_causal_lm.py:332-355` (mellum-kstack)
- **Unsloth source:** `../unsloth/` and `../unsloth-zoo/`
- **Vmap limitations:** `docs/limitations.md`
- **Current constraints:** `torch.func.vmap` requires functional purity
- **PyTorch vmap+checkpoint issue:** https://github.com/pytorch/pytorch/issues/165880

---

## Next Steps

1. Review this plan
2. Begin Phase 0: Baseline measurement
3. Implement phases 1-4 sequentially
4. Document results after each phase
5. Decide on Phase 5 based on Phases 1-4 outcomes
6. Test `cut_cross_entropy` vmap compatibility for Phase 5D
