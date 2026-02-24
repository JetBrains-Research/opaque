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

### Phase 5: Research (Optional, 2-4 weeks)

**High-risk, high-reward explorations**

**Option A: 4-bit Quantization with Vmap** (1-2 weeks)
- Test if bitsandbytes NF4 works with vmap
- Potential: ~10 GB savings
- Risk: May be fundamentally incompatible

**Option B: Functional Gradient Checkpointing** (2-4 weeks)
- Implement checkpointing using `torch.func.vjp`
- Avoid `requires_grad_()` mutations
- Potential: 2-4 GB savings
- Risk: May conflict with Opaque's gradient computation

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

**Stretch (with Phase 5):**
- Memory: ~60-65 GB (10-15 GB saved)
- Micro-batch: 12-16 (3-4x throughput)

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

## References

- **Baseline config:** `examples/train_causal_lm.py:332-355` (mellum-kstack)
- **Unsloth source:** `/tmp/unsloth/` and `/tmp/unsloth-zoo/`
- **Vmap limitations:** `docs/limitations.md`
- **Current constraints:** `torch.func.vmap` requires functional purity

---

## Next Steps

1. Review this plan
2. Begin Phase 0: Baseline measurement
3. Implement phases 1-4 sequentially
4. Document results after each phase
5. Decide on Phase 5 based on Phases 1-4 outcomes
