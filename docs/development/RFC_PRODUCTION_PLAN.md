# RFC: Opaque Production Architecture & Implementation Plan

**Status:** Design Document
**Author:** Based on JAX-Privacy and jbr-fed-accounting analysis
**Date:** 2026-02-12
**Supersedes:** All previous architecture RFCs

---

## Executive Summary

This RFC presents a unified plan to evolve Opaque from a functional prototype (Stages 1-2 complete) to a production-ready DP training library. Key decisions:

1. **Architecture**: Adopt **functional design** (higher-order functions) following JAX-Privacy patterns
2. **Production Focus**: Prioritize memory efficiency and validation over feature breadth
3. **Timeline**: 4-phase plan over ~16 weeks to production readiness
4. **Accounting**: Keep training library independent; integrate with jbr-fed-accounting later via events

**Current State**: 111 tests passing (55 accounting + 56 optimizer), GPT-2 integration working, but lacks memory optimization and Opacus validation.

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Design Decision: Functional Architecture](#2-design-decision-functional-architecture)
3. [Production Readiness Gaps](#3-production-readiness-gaps)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Implementation Plan](#5-implementation-plan)
6. [Validation Strategy](#6-validation-strategy)
7. [Migration Path](#7-migration-path)
8. [Future Work](#8-future-work)

---

## 1. Current State Assessment

### 1.1 What Works Today ✅

**Core Primitives** (Stages 1-2 Complete):
- ✅ **Clipping**: `clip_pytree()`, `clipped_grad()`, `clipped_fun()` - Full JAX-Privacy API parity
- ✅ **Noise**: `add_gaussian_noise()` - Stateless noise injection
- ✅ **Accounting**: Functional API with `compose_*()` and `get_epsilon/beta/advantage()` queries
- ✅ **Optimizers**: `adaptive_clipping()` wrapper for TorchOpt optimizers

**Test Coverage**:
- 111 tests passing (55 accounting + 56 optimizer)
- 76% coverage on accounting module
- Numerical equivalence with JAX-Privacy validated (atol=1e-5)
- Parallel test execution enabled (3.17× speedup with pytest-xdist)

**Real Model Validation**:
- ✅ GPT-2 (124M params) integration test passes
- ✅ Custom TinyLLaMA (2-layer transformer) works
- ✅ HuggingFace models compatible via `torch.func.functional_call`

**Key Achievement**: `functional_call` successfully traces through messy HuggingFace code!

### 1.2 Critical Gaps ⚠️

**1. Memory Efficiency** 🔥 **BLOCKER**
- `vmap` over batch_size=64 causes OOM on GPT-2 (~31GB gradient memory)
- No microbatching implementation yet
- No memory profiling tools

**2. Production Validation** ❌ **HIGH PRIORITY**
- Zero cross-validation against Opacus
- No utility benchmarks (accuracy comparisons)
- No end-to-end training tests on realistic workloads

**3. Scale Testing** ❌
- Not tested on models >1B parameters
- LoRA integration unverified
- Gradient checkpointing not implemented

**4. Documentation** ❌
- Limited production usage guides
- No migration guide from Opacus
- No troubleshooting documentation

### 1.3 Bottom Line

**Opaque has correct functional primitives but lacks production hardening for real-world use.**

---

## 2. Design Decision: Functional Architecture

### 2.1 Why Functional Over Class-Based?

After analyzing JAX-Privacy's design (the gold standard for functional DP) and comparing with class-based alternatives, **functional architecture wins** on:

1. ✅ **Natural composition**: `noise_fn(grad_fn(...))` reads like mathematical composition
2. ✅ **Proven pattern**: JAX-Privacy uses this successfully for 3+ years
3. ✅ **Alignment**: Mirrors jbr-fed-accounting's compositional Rust API
4. ✅ **PyTorch idioms**: Perfect fit with `torch.func` (vmap, grad, functional_call)
5. ✅ **Less boilerplate**: No class hierarchies, just write functions
6. ✅ **Research flexibility**: Quick iteration on new mechanisms

### 2.2 JAX-Privacy Pattern Analysis

**Core Pattern**: `BoundedSensitivityCallable` dataclass wraps functions with sensitivity tracking

```python
@dataclass(frozen=True)
class BoundedSensitivityCallable:
    """Lightweight wrapper tracking sensitivity of a function."""
    fun: Callable[..., Any]
    l2_norm_bound: float
    has_aux: bool

    def sensitivity(self, neighboring: str) -> float:
        """Return sensitivity based on neighboring relation."""
        multiplier = 2.0 if neighboring == "replace_one" else 1.0
        return self.l2_norm_bound * multiplier

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)
```

**Higher-Order Functions**: Functions return wrapped callables

```python
def clipped_grad(
    loss_fn: Callable,
    *,
    l2_clip_norm: float,
    batch_argnums: int = 1,
    ...
) -> BoundedSensitivityCallable:
    """Create a function that computes clipped gradients."""

    def grad_fn(*args, **kwargs):
        # ... compute, clip, sum ...
        return clipped_grads, aux

    norm_bound = 1.0 if rescale_to_unit_norm else l2_clip_norm
    return BoundedSensitivityCallable(grad_fn, norm_bound, has_aux)
```

**Usage**:

```python
# Create wrapped function
grad_fn = jax_privacy.clipped_grad(loss_fn, l2_clip_norm=1.0)

# Use in training
grads = grad_fn(params, batch)

# Access sensitivity for noise calibration
sensitivity = grad_fn.sensitivity("replace_one")
noise_fn = gaussian_privatizer(stddev=noise_multiplier * sensitivity)
```

**Key Insights**:
- Only **noise addition is stateful** (for PRNG key management)
- Everything else is **pure functions**
- Composition is **built-in** (no custom API needed)

### 2.3 Comparison: Current vs Proposed

**Current API** (flat functional):

```python
# Current: Direct calls, no composition
grads = clipped_grad(loss_fn, l2_clip_norm=1.0)(params, batch)
noisy = add_gaussian_noise(grads, noise_multiplier=1.1, clip_norm=1.0)
```

**Proposed API** (higher-order functional):

```python
# Proposed: Configure once, compose naturally
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = gaussian(noise_multiplier=1.1, sensitivity=grad_fn.sensitivity())

# Training loop
grads = grad_fn(params, batch)
noisy = noise_fn(grads)
```

**Benefits**:
- Configure once, use many times
- Sensitivity tracked automatically
- Natural composition: `noise_fn(grad_fn(...))`
- Aligns with jbr-fed-accounting: `gaussian(nm).poisson(rate).repeat(count)`

---

## 3. Production Readiness Gaps

### 3.1 Gap 1: Memory Management 🔥 **CRITICAL**

**Problem**: `torch.func.vmap` materializes all per-example gradients

**Example** (GPT-2, batch_size=64):
- Model: 124M parameters × 4 bytes = 496 MB
- Per-example grads: 124M × 64 × 4 bytes = **31 GB**
- Result: **OOM on most GPUs**

**Solutions Needed**:

1. **Microbatching with AccumulationType**
   ```python
   # Process batch in chunks
   clipped_fn = clipped_grad(
       loss_fn,
       l2_clip_norm=1.0,
       microbatch_size=8,  # Process 8 at a time
   )
   grads = clipped_fn(params, batch)  # Uses 8× less memory
   ```

2. **Memory Profiling Tool**
   ```python
   from opaque.profiling import estimate_memory, recommend_microbatch_size

   est = estimate_memory(model, batch_size=64)
   print(f"Peak memory: {est.peak_gb:.1f} GB")

   if est.will_oom(available_gb=16):
       rec = recommend_microbatch_size(model, batch_size=64, available_gb=16)
       print(f"Use microbatch_size={rec}")
   ```

3. **Gradient Checkpointing Integration**
   ```python
   from torch.utils.checkpoint import checkpoint_sequential

   # Wrap expensive layers
   model.transformer.layers = checkpoint_sequential(
       model.transformer.layers,
       segments=4,
   )
   ```

**Implementation Priority**: Phase 1 (immediate)

### 3.2 Gap 2: Cross-Validation ❌ **HIGH PRIORITY**

**Problem**: No systematic validation that Opaque matches Opacus

**Required Tests**:

```python
# tests/validation/test_opaque_vs_opacus.py

def test_gradient_equivalence():
    """Same model, same batch → same gradients."""
    opacus_grads = compute_with_opacus(model, batch, clip=1.0)
    opaque_grads = compute_with_opaque(model, batch, clip=1.0)

    assert torch.allclose(opacus_grads, opaque_grads, atol=1e-5)

def test_privacy_consumption_equivalence():
    """Train same steps → same epsilon."""
    opacus_epsilon = train_with_opacus(model, data, steps=1000, noise=1.1)
    opaque_epsilon = train_with_opaque(model, data, steps=1000, noise=1.1)

    assert abs(opacus_epsilon - opaque_epsilon) < 0.1

def test_utility_equivalence():
    """Same training → same accuracy."""
    opacus_model = train_with_opacus(model, data, epochs=10)
    opaque_model = train_with_opaque(model, data, epochs=10)

    opacus_acc = evaluate(opacus_model, test_set)
    opaque_acc = evaluate(opaque_model, test_set)

    assert abs(opacus_acc - opaque_acc) < 0.02  # Within 2%
```

**Test Matrix**:

| Model | Dataset | Metric | Status |
|-------|---------|--------|--------|
| Linear | MNIST | Gradient ≈ | ❌ |
| CNN | CIFAR-10 | Epsilon ≈ | ❌ |
| GPT-2 | Wikitext | Utility ≈ | ❌ |

**Implementation Priority**: Phase 1 (critical for trust)

### 3.3 Gap 3: Large Model Support ❌

**Current State**: Only tested on GPT-2 (124M params)

**Required Testing**:

| Model | Params | LoRA | Gradient Checkpointing | Status |
|-------|--------|------|------------------------|--------|
| GPT-2 | 124M | ❌ | ❌ | ✅ Works |
| GPT-2-Large | 774M | ❌ | ❌ | ❌ Untested |
| Llama-7B | 7B | ✅ | ✅ | ❌ Untested |
| Llama-13B | 13B | ✅ | ✅ | ❌ Untested |

**Blockers**:
1. Memory profiling not implemented
2. Microbatching not implemented
3. LoRA integration not implemented
4. Gradient checkpointing not integrated

**Implementation Priority**: Phase 2

### 3.4 Gap 4: Documentation ❌

**Current State**: Developer-focused internal docs

**Required User Documentation**:

1. **Getting Started**
   - Installation
   - 10-minute tutorial
   - Basic examples (linear, CNN, transformer)

2. **Migration Guide**
   - From Opacus to Opaque
   - API mapping table
   - Common patterns

3. **Production Guide**
   - Memory management strategies
   - Microbatching guidelines
   - Debugging OOM errors
   - Performance tuning

4. **Known Limitations**
   - What works / what doesn't
   - Workarounds
   - Future roadmap

**Implementation Priority**: Phase 3

---

## 4. Proposed Architecture

### 4.1 Module Structure

```
src/opaque/
├── __init__.py              # Public API exports
├── clipping/                # Gradient clipping
│   ├── __init__.py          # clipped_grad, clipped_fun, l2_clipper, ...
│   ├── types.py             # BoundedSensitivityCallable
│   ├── pytree.py            # clip_pytree (low-level)
│   ├── clipped_fun.py       # clipped_fun (mid-level)
│   ├── clipped_grad.py      # clipped_grad (high-level)
│   ├── microbatching.py     # NEW: Microbatching implementation
│   └── strategies.py        # NEW: per_layer_clipper, adaptive_clipper
├── noise/                   # Noise injection
│   ├── __init__.py          # gaussian, laplace, gaussian_stateful
│   ├── gaussian.py          # Gaussian noise (stateless + stateful)
│   └── laplace.py           # NEW: Laplace noise (pure DP)
├── sampling/                # Batch sampling
│   ├── __init__.py          # poisson, truncated_poisson
│   ├── poisson.py           # Poisson subsampling
│   └── truncated.py         # Truncated Poisson (variable batch size)
├── accounting/              # Privacy accounting (functional)
│   ├── __init__.py          # create, compose_*, get_epsilon/beta/advantage
│   ├── composition.py       # Composition functions
│   ├── queries.py           # Privacy queries
│   └── calibration.py       # Calibration using riskcal
├── optimizers/              # Optimizer wrappers
│   ├── __init__.py          # adaptive_clipping
│   └── adaptive_clipping.py # Adaptive clipping wrapper for TorchOpt
├── profiling/               # NEW: Memory profiling & diagnostics
│   ├── __init__.py          # estimate_memory, recommend_microbatch_size
│   ├── memory.py            # Memory estimation
│   └── benchmarks.py        # Performance benchmarking
├── integration/             # NEW: High-level integrations
│   ├── __init__.py          # DPTrainer (optional high-level API)
│   ├── lora.py              # LoRA adapter detection & training
│   └── checkpointing.py     # Gradient checkpointing helpers
└── utils/                   # Utilities
    ├── __init__.py
    ├── pytree.py            # PyTree operations
    └── functional.py        # make_functional, etc.
```

### 4.2 Core API Design

See DESIGN_COMPARISON_EXAMPLES.md for detailed code examples. Key functional patterns:

1. **BoundedSensitivityCallable wrapper** for sensitivity tracking
2. **Higher-order functions** returning configured callables
3. **Explicit state passing** for reproducibility (noise PRNG)
4. **Natural composition** via function chaining

---

## 5. Implementation Plan

### Phase 1: Production Hardening (4-6 weeks) 🎯 **CRITICAL**

**Goal**: Make Opaque reliable for single-GPU, medium-scale models (≤1B params)

**Week 1-2: Memory Management**
- [ ] Implement microbatching in `clipped_grad()`
  - Add `microbatch_size` parameter
  - Use `AccumulationType.SUM` for gradients
  - Use `AccumulationType.CONCAT` for auxiliary outputs
  - Test: Verify results identical to full batch
- [ ] Implement memory profiling tools
  - `estimate_memory()` function
  - `recommend_microbatch_size()` function
  - Test on GPT-2, GPT-2-Large

**Week 3-4: Cross-Validation**
- [ ] Create `tests/validation/test_opaque_vs_opacus.py`
  - Test 1: Gradient equivalence (single step)
  - Test 2: Privacy consumption equivalence (full training)
  - Test 3: Utility equivalence (final accuracy)
- [ ] Test matrix: Linear (MNIST), CNN (CIFAR-10), GPT-2 (Wikitext)
- [ ] Document any discrepancies and root causes

**Week 5-6: Large Model Testing**
- [ ] Test GPT-2-Large (774M params) with microbatching
- [ ] Test GPT-2-XL (1.5B params) with gradient checkpointing
- [ ] Create troubleshooting guide for OOM errors
- [ ] Document recommended settings per model size

**Success Criteria**:
- ✅ Train GPT-2 on Wikitext without OOM (batch_size=32, microbatch_size=8)
- ✅ Match Opacus privacy accounting within 0.1 epsilon
- ✅ Match Opacus utility within 2% accuracy on MNIST/CIFAR-10
- ✅ Memory profiler predicts actual usage within 20%

**Deliverables**:
- Microbatching implementation
- Memory profiling tools
- Opacus validation tests
- Production troubleshooting guide

---

### Phase 2: Functional API Refactoring (3-4 weeks)

**Goal**: Adopt functional architecture following JAX-Privacy patterns

**Week 1: BoundedSensitivityCallable**
- [ ] Add `BoundedSensitivityCallable` dataclass to `clipping/types.py`
- [ ] Update `clipped_grad()` to return wrapped function
- [ ] Add `sensitivity()` method with neighboring relation support
- [ ] Tests: Verify backward compatibility

**Week 2: Higher-Order Noise Functions**
- [ ] Implement `gaussian()` returning function (stateless)
- [ ] Implement `gaussian_stateful()` with explicit state
- [ ] Implement `laplace()` for pure DP
- [ ] Tests: Equivalence with current `add_gaussian_noise()`

**Week 3: Compositional Sampling**
- [ ] Implement `poisson()` returning sampler function
- [ ] Implement `truncated_poisson()` for fixed max batch size
- [ ] Integration tests with clipping + noise
- [ ] Update tutorial notebook

**Week 4: API Migration**
- [ ] Update all examples to use new API
- [ ] Deprecate old API with warnings
- [ ] Update documentation
- [ ] Create migration guide

**Success Criteria**:
- ✅ New API matches JAX-Privacy patterns
- ✅ All existing tests pass with new API
- ✅ Tutorial demonstrates compositional usage
- ✅ Migration guide shows old→new mapping

**Deliverables**:
- Functional API implementation
- Updated tests and examples
- Migration guide
- Tutorial notebook

---

### Phase 3: Scale to Billions (4-6 weeks)

**Goal**: Support Llama-7B/13B LoRA fine-tuning

**Week 1-2: LoRA Integration**
- [ ] Implement LoRA adapter detection
- [ ] Create `opaque.integration.lora.make_lora_functional()`
- [ ] Test: Fine-tune Llama-7B on Alpaca
- [ ] Verify memory usage < 24GB (single A100)

**Week 3-4: Gradient Checkpointing**
- [ ] Integrate `torch.utils.checkpoint` with `functional_call`
- [ ] Create helper: `opaque.integration.checkpointing.wrap_layers()`
- [ ] Test: Llama-13B with checkpointing fits in 40GB
- [ ] Measure compute vs memory tradeoff

**Week 5-6: Large-Scale Validation**
- [ ] End-to-end LoRA fine-tuning experiments
- [ ] Compare utility against published baselines
- [ ] Performance benchmarking (tokens/sec)
- [ ] Memory optimization guide

**Success Criteria**:
- ✅ LoRA fine-tune Llama-7B on Alpaca (single A100, <24GB)
- ✅ Achieve published utility benchmarks (within 5% accuracy)
- ✅ Document memory-compute tradeoffs

**Deliverables**:
- LoRA integration module
- Gradient checkpointing helpers
- Large-scale validation results
- Performance benchmarks

---

### Phase 4: Production Polish (2-3 weeks)

**Goal**: Release-ready library with comprehensive documentation

**Week 1: Documentation**
- [ ] Getting Started guide (10-minute tutorial)
- [ ] User Guide (DP basics, clipping strategies, memory management)
- [ ] API Reference (auto-generated from docstrings)
- [ ] Migration Guide (Opacus → Opaque)
- [ ] Troubleshooting (common errors, solutions)

**Week 2: Testing & CI**
- [ ] Expand test coverage to >80%
- [ ] Add integration tests for large models (CI with GPU)
- [ ] Performance regression tests
- [ ] Memory usage tests

**Week 3: Release Preparation**
- [ ] Finalize API (no breaking changes after 1.0)
- [ ] Code review and cleanup
- [ ] Prepare release notes
- [ ] Create release checklist

**Success Criteria**:
- ✅ Documentation covers all user scenarios
- ✅ Test coverage >80%
- ✅ CI passes on CPU and GPU
- ✅ Ready for public release

**Deliverables**:
- Complete documentation site
- Comprehensive test suite
- Release-ready codebase
- v1.0.0 release

---

## 6. Validation Strategy

### Tier 1: Numerical Correctness ✅ **COMPLETE**

**Goal**: Verify primitives match JAX-Privacy mathematically

**Status**: 111 tests passing, numerical equivalence validated (atol=1e-5)

### Tier 2: Integration Testing ⚠️ **IN PROGRESS (Phase 1)**

**Goal**: Real models work end-to-end

**Test Matrix**:

| Model | Params | Microbatching | Status |
|-------|--------|---------------|--------|
| TinyLLaMA | 2M | ❌ | ✅ Works |
| GPT-2 | 124M | ❌ | ✅ Works |
| GPT-2-Large | 774M | ✅ | ❌ Phase 1 |
| GPT-2-XL | 1.5B | ✅ | ❌ Phase 1 |
| Llama-7B LoRA | 7B | ✅ | ❌ Phase 3 |

### Tier 3: End-to-End Parity ❌ **PLANNED (Phase 1)**

**Goal**: Match Opacus on privacy and utility

**Test Matrix**:

| Model | Dataset | Metric | Opacus | Opaque | Status |
|-------|---------|--------|--------|--------|--------|
| Linear | MNIST | Acc @ ε=3 | ? | ? | ❌ Phase 1 |
| CNN | CIFAR-10 | Acc @ ε=8 | ? | ? | ❌ Phase 1 |
| GPT-2 | Wikitext | PPL @ ε=3 | ? | ? | ❌ Phase 1 |

### Tier 4: Published Baselines ❌ **FUTURE (Phase 4)**

**Goal**: Reproduce published DP-SGD results

**Targets**:
- Abadi et al. (2016): MNIST + CNN (ε=8, acc~95%)
- Yu et al. (2021): CIFAR-10 + ResNet (ε=3, acc~68%)
- Li et al. (2021): BERT fine-tuning on GLUE

---

## 7. Migration Path

### 7.1 From Current API → Functional API

**Current API** (flat):

```python
# Before: Direct function calls
from opaque import clipped_grad, add_gaussian_noise

grads = clipped_grad(loss_fn, l2_clip_norm=1.0)(params, batch)
noisy = add_gaussian_noise(grads, noise_multiplier=1.1, clip_norm=1.0)
```

**Functional API** (proposed):

```python
# After: Higher-order functions
from opaque import clipped_grad, gaussian

grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = gaussian(noise_multiplier=1.1, sensitivity=grad_fn.sensitivity())

grads = grad_fn(params, batch)
noisy = noise_fn(grads)
```

**Backward Compatibility**:

Keep old API as deprecated wrappers for 2 releases:

```python
# opaque/__init__.py

def compute_clipped_grads(loss_fn, params, batch, l2_clip_norm):
    """Deprecated: Use clipped_grad() instead."""
    warnings.warn(
        "compute_clipped_grads() is deprecated. "
        "Use grad_fn = clipped_grad(...); grads = grad_fn(params, batch)",
        DeprecationWarning,
        stacklevel=2,
    )
    grad_fn = clipping.clipped_grad(loss_fn, l2_clip_norm=l2_clip_norm)
    return grad_fn(params, batch)
```

### 7.2 From Opacus → Opaque

**API Mapping Table**:

| Opacus | Opaque Equivalent | Notes |
|--------|-------------------|-------|
| `PrivacyEngine` | No equivalent | Low-level API only |
| `GradSampleModule` | `clipped_grad()` | Functional, no module wrapping |
| `make_private()` | `clipped_grad()` + `gaussian()` | Explicit composition |
| `DPOptimizer` | `adaptive_clipping()` | Wrapper for TorchOpt |
| `PrivacyAccountant` | `opaque.accounting` | Functional API |

See DESIGN_COMPARISON_EXAMPLES.md for detailed migration examples.

---

## 8. Future Work (Beyond v1.0)

### Short-Term (Months 5-6)

1. **Advanced Optimizers**
   - DP-FTRL (Follow-The-Regularized-Leader)
   - DP-SGD with momentum
   - Per-layer learning rates

2. **Advanced Sampling**
   - Secure shuffle (cryptographic privacy amplification)
   - Stratified sampling (by user/group)

3. **Debugging Tools**
   - Privacy auditing (empirical epsilon estimation)
   - Gradient flow visualization
   - Sensitivity analysis tools

### Medium-Term (Months 7-12)

1. **Multi-GPU Support** (Research)
   - Distributed DP-SGD
   - Gradient aggregation strategies
   - Privacy accounting for distributed training

2. **Advanced Noise Mechanisms**
   - Correlated noise (matrix factorization methods)
   - Adaptive noise (noise scheduling)

3. **Production Infrastructure**
   - Model checkpointing with privacy state
   - Incremental training (resume from checkpoint)
   - Privacy budget management across experiments

### Long-Term (Year 2+)

1. **New Privacy Definitions**
   - User-level DP (multiple examples per user)
   - Local DP (on-device training)
   - Shuffle DP (amplification via secure shuffle)

2. **Cross-Framework Integration**
   - JAX interoperability (shared accounting)
   - TensorFlow Privacy integration
   - Export to ONNX with privacy metadata

3. **Federated Learning**
   - Integration with federated-compute/federated-research
   - Client-side DP-SGD
   - Secure aggregation protocols

---

## 9. Open Questions & Decisions Needed

### 9.1 State Management Philosophy

**Question**: Should we use closures or explicit state passing for stateful components?

**Recommendation**: Use **closures for convenience** (adaptive clipping), **explicit state for reproducibility** (noise PRNG). Document both patterns.

### 9.2 Microbatching API Design

**Question**: How to expose microbatching control?

**Recommendation**: Start with **manual** (user specifies `microbatch_size`), add automatic in Phase 4 if users request it.

### 9.3 Integration with jbr-fed-accounting

**Question**: How should Opaque integrate with jbr-fed-accounting (once Python bindings exist)?

**Recommendation**: Start **separate** (no integration), add event emission in Phase 4 if needed. Avoid tight coupling.

### 9.4 High-Level API

**Question**: Should Opaque provide a high-level `DPTrainer` API (like HuggingFace Trainer)?

**Recommendation**: **Defer to Phase 4**. Focus on low-level functional API first. Add high-level wrapper if users request it.

---

## 10. Timeline Summary

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| **Phase 1** | 4-6 weeks | Production Hardening | Microbatching, profiling, Opacus validation |
| **Phase 2** | 3-4 weeks | Functional API | BoundedSensitivityCallable, higher-order functions |
| **Phase 3** | 4-6 weeks | Scale to Billions | LoRA integration, gradient checkpointing |
| **Phase 4** | 2-3 weeks | Production Polish | Documentation, testing, release |
| **Total** | **13-19 weeks** | **~4 months** | **Production-ready v1.0.0** |

---

## 11. Conclusion

This RFC presents a comprehensive plan to evolve Opaque from a functional prototype to a production-ready DP training library. The plan prioritizes:

1. **Memory efficiency** (Phase 1) - Blocker for large models
2. **Validation** (Phase 1) - Critical for trust
3. **Functional architecture** (Phase 2) - Aligns with JAX-Privacy and jbr-fed-accounting
4. **Scale** (Phase 3) - Support Llama-7B/13B LoRA
5. **Polish** (Phase 4) - Release-ready

**Estimated timeline**: 4 months to production-ready v1.0.0 release.

**Next Steps**:
1. Review and approve this RFC
2. Begin Phase 1 Week 1: Microbatching implementation
3. Set up weekly progress reviews
