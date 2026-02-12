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

### 2.2 Opaque Pattern: Plain Functions with Attributes

**Core Pattern**: Higher-order functions return plain callables with metadata attributes

**Design Decision**: After analysis, we **simplified** JAX-Privacy's `BoundedSensitivityCallable` wrapper:
- **Sensitivity is a privacy accounting concern**, not a training concern
- **Neighboring relations** (ADD_OR_REMOVE, REPLACE_ONE) belong in accounting layer
- **Plain functions with attributes** are simpler and equally composable

**Implementation**:

```python
def clipped_grad(
    loss_fn: Callable,
    *,
    l2_clip_norm: float,
    batch_argnums: int = 1,
    ...
) -> Callable:
    """Create a function that computes clipped gradients."""

    def grad_fn(*args, **kwargs):
        # ... compute, clip, sum ...
        return clipped_grads, aux

    # Store clip_norm as function attribute (not a method)
    grad_fn.clip_norm = 1.0 if rescale_to_unit_norm else l2_clip_norm

    return grad_fn
```

**Usage**:

```python
# Create clipping function
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)

# Access clip norm as simple attribute
print(grad_fn.clip_norm)  # 1.0

# Create noise function (user does multiplication)
noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)

# Training loop
for batch in dataloader:
    grads = grad_fn(params, batch)
    noisy = noise_fn(grads)
    params = optimizer.step(params, noisy)
```

**Key Insights**:
- **Plain functions** - No wrapper classes needed
- **Simple attributes** - Use Python's function attributes for metadata
- **Explicit math** - User does `stddev = noise_mult * clip_norm` (clearer)
- **Full composability** - Easy to swap any component
- **Research flexibility** - No abstraction barriers for experimenting

### 2.3 Comparison: Old vs New

**Old API** (PoC - flat functional):

```python
# Old: Direct calls, no composition
grads = clipped_grad(loss_fn, l2_clip_norm=1.0)(params, batch)
noisy = add_gaussian_noise(grads, stddev=1.1)
```

**New API** (Production - higher-order functional):

```python
# New: Configure once, compose naturally
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)

# Training loop
for batch in dataloader:
    grads = grad_fn(params, batch)
    noisy = noise_fn(grads)
    params = optimizer.step(params, noisy)
```

**Benefits**:
- ✅ **Configure once, use many times** - Define components outside loop
- ✅ **Natural composition** - `noise_fn(grad_fn(...))` reads like math
- ✅ **Simple & explicit** - No hidden complexity, clear data flow
- ✅ **Research flexibility** - Easy to swap clipping/noise mechanisms:
  ```python
  # Swap clipping
  grad_fn = per_layer_clipped_grad(loss_fn, clip_norms={'layer1': 1.0, 'layer2': 0.5})

  # Swap noise
  noise_fn = correlated_gaussian(stddev=1.1, rank=10)  # Matrix factorization
  noise_fn = clipped_gaussian(stddev=1.1, clip_at=3.0)  # Truncated noise
  noise_fn = laplace(scale=1.1)  # Pure DP
  ```

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

### Phase 1: Memory & Core Production (6-8 weeks) 🎯 **CRITICAL**

**Goal**: Make Opaque reliable for single-GPU, medium-scale models (≤1B params)

**Week 1-2: Memory Management**
- [ ] Implement microbatching in `clipped_grad()` and `clipped_fun()`
  - Actually implement `microbatch_size` parameter (currently declared but not implemented)
  - Use `AccumulationType.SUM` for gradients
  - Use `AccumulationType.CONCAT` for auxiliary outputs
  - Pattern: Split batch into chunks, accumulate clipped gradients
  - Test: Verify results identical to full batch
- [ ] Implement memory profiling tools (`opaque/profiling/`)
  - `estimate_memory(model, batch_size)` - Predict memory usage
  - `recommend_microbatch_size(model, batch_size, available_gb)` - Suggest optimal size
  - Test on GPT-2, GPT-2-Large

**Week 3-4: Cross-Validation**
- [ ] Create `tests/validation/test_opaque_vs_opacus.py`
  - Test 1: Gradient equivalence (single step)
  - Test 2: Privacy consumption equivalence (full training)
  - Test 3: Utility equivalence (final accuracy)
- [ ] Test matrix: Linear (MNIST), CNN (CIFAR-10), GPT-2 (Wikitext)
- [ ] Document any discrepancies and root causes

**Week 5-6: Gradient Checkpointing & Large Models**
- [ ] Integrate `torch.utils.checkpoint` with `functional_call`
  - Verify compatibility with `torch.func.vmap`
  - Create helper: `opaque.integration.checkpointing.wrap_layers()`
  - Test: GPT-2-XL (1.5B params) with checkpointing fits in 40GB
  - Measure compute vs memory tradeoff
- [ ] Test GPT-2-Large (774M params) with microbatching
- [ ] Create troubleshooting guide for OOM errors
- [ ] Document recommended settings per model size

**Week 7-8: Empirical Privacy Auditing**
- [ ] Implement `opaque.auditing` module (~1,146 lines from JAX-Privacy)
  - `CanaryScoreAuditor(in_scores, out_scores)` - Core auditing class
  - `epsilon_raw_counts()` - Simplest method (start here)
  - `epsilon_clopper_pearson()` - Standard method
  - `epsilon_one_run()` - State-of-the-art single-run method (Steinke 2024)
  - Bootstrap confidence intervals with BCa
  - Utility metrics: AUROC, TPR@FPR, max accuracy
- [ ] Validation tests against JAX-Privacy auditing
- [ ] Tutorial: How to audit your DP training

**Success Criteria**:
- ✅ Train GPT-2 on Wikitext without OOM (batch_size=32, microbatch_size=8)
- ✅ Match Opacus privacy accounting within 0.1 epsilon
- ✅ Match Opacus utility within 2% accuracy on MNIST/CIFAR-10
- ✅ Memory profiler predicts actual usage within 20%
- ✅ Gradient checkpointing works with vmap (GPT-2-XL fits in 40GB)
- ✅ Empirical auditing validates theoretical epsilon bounds

**Deliverables**:
- Microbatching implementation (actually working)
- Memory profiling tools
- Gradient checkpointing integration
- Opacus validation tests
- Empirical privacy auditing module
- Production troubleshooting guide

---

### Phase 2: Batch Selection & Advanced Accounting (3-4 weeks)

**Goal**: Complete batch sampling strategies and multi-epoch accounting

**Week 1: Batch Selection Strategies**
- [ ] Implement `opaque.sampling.batch_selection` module
  - `CyclicPoissonSampling` - Core DP-SGD with cyclic sampling (BandMF amplification)
  - `BallsInBinsSampling` - Alternative to Poisson (arxiv.org/abs/2410.06266)
  - `FixedBatchSampling` - Uniform random with/without replacement
  - `UserSelectionStrategy` - Federated learning (user-level DP)
  - `split_and_pad_global_batch()` - Split with padding (-1 indices)
  - `pad_to_multiple_of()` - Pad to batch size multiple
- [ ] Integration with DataLoader patterns
- [ ] Tests: Validate sampling distributions, padding strategies

**Week 2: Multi-Epoch Privacy Accounting**
- [ ] Extend accounting module for multi-epoch training
  - Without-replacement sampling across epochs
  - Amplification from epoch boundaries
  - Integration with `dp_accounting` library
- [ ] Event constructors for multi-epoch mechanisms
- [ ] Tests: Validate amplification vs JAX-Privacy

**Week 3: Functional API Refinement**
- [ ] **Consider**: Stateful noise API wrapper (if needed)
  - `gaussian_stateful(stddev, seed)` with PRNG state management
  - **Defer if not critical** - stateless is clearer for DP
- [ ] Add composition examples (grad_fn + noise_fn + optimizer)
- [ ] Document functional patterns

**Week 4: Documentation & Migration**
- [ ] Update tutorial notebook with batch selection
- [ ] Multi-epoch training examples
- [ ] Update README examples
- [ ] Document sampling strategies trade-offs

**Success Criteria**:
- ✅ All batch selection strategies implemented and tested
- ✅ Multi-epoch accounting matches JAX-Privacy
- ✅ Cyclic sampling amplification validated
- ✅ Integration examples with DataLoader

**Deliverables**:
- Batch selection strategies module
- Multi-epoch accounting
- Updated tutorials
- Sampling strategy comparison docs

---

### Phase 3: Matrix Factorization & DP-FTRL (6-8 weeks)

**Goal**: Implement correlated noise mechanisms (BandMF, DP-FTRL) for 10-50% utility improvement

**Week 1-2: Matrix Factorization Core**
- [ ] Implement `opaque.matrix_factorization` module structure
  - `streaming_matrix.py` - Abstract interface for matrix multiplication
  - `dense.py` - Small-scale exact factorizations (start here)
  - `banded.py` - General banded matrix support
  - `sensitivity.py` - L2 sensitivity computation under participation patterns
    - `single_participation_sensitivity(C)` - Single-epoch
    - `minsep_sensitivity(C, min_sep, max_participations)` - Multi-epoch
    - `max_participation_for_linear_fn(x, min_sep, max_participations)` - Dynamic programming
  - `optimization.py` - Finding optimal strategy matrices (L-BFGS)
- [ ] Tests: Validate sensitivity computations, matrix operations

**Week 3-4: BandMF (Banded Matrix Factorization)**
- [ ] Implement `toeplitz.py` - Banded Toeplitz matrices
  - BandMF mechanism (arxiv.org/abs/2306.08153)
  - Cyclic sampling integration for amplified privacy
  - Optimal band selection via optimization
- [ ] Implement `buffered_toeplitz.py` - Memory-efficient streaming
  - BLT mechanisms (arxiv.org/abs/2404.16706)
  - Multi-epoch support (arxiv.org/abs/2408.08868)
- [ ] Tests: Validate utility improvement vs independent noise

**Week 5-6: DP-FTRL Integration**
- [ ] Implement DP-FTRL optimizer wrapper
  - Follow-The-Regularized-Leader with matrix mechanisms
  - State management for correlated noise across steps
  - Integration with TorchOpt optimizers
- [ ] Noise addition transforms using matrix factorization
  - `matrix_factorization_privatizer()` wrapper
  - Stateful noise generation with strategy matrices
- [ ] Tests: Multi-epoch training with correlated noise

**Week 7-8: Validation & Optimization**
- [ ] End-to-end validation: BandMF vs standard DP-SGD
  - Compare utility at same privacy budget
  - Measure 10-50% accuracy improvement (as per papers)
  - Test on MNIST, CIFAR-10, Wikitext
- [ ] Performance optimization for online noise generation
- [ ] Documentation: When to use matrix mechanisms

**Success Criteria**:
- ✅ BandMF achieves 10-50% utility improvement over independent noise (matching papers)
- ✅ Sensitivity computations validated against JAX-Privacy
- ✅ DP-FTRL works for multi-epoch training
- ✅ Efficient online noise generation (<10% overhead)

**Deliverables**:
- Matrix factorization module (~2,000+ lines from JAX-Privacy)
- BandMF and BLT mechanisms
- DP-FTRL optimizer wrapper
- Utility improvement benchmarks
- Usage guide: When to use correlated noise

---

### Phase 4: Scale to Billions & LoRA (4-6 weeks)

**Goal**: Support Llama-7B/13B LoRA fine-tuning with all production features

**Week 1-2: LoRA Integration**
- [ ] Implement `opaque.integration.lora` module
  - Automatic detection of PEFT/LoRA parameters (`.lora_A`, `.lora_B`)
  - Helper: `make_lora_functional(model)` for PEFT integration
  - Memory-efficient handling (only compute gradients for adapters)
  - **Note**: No LoRA-specific optimizations in JAX-Privacy, just convenience wrappers
- [ ] Test: Fine-tune Llama-7B on Alpaca with DP-SGD
- [ ] Test: Fine-tune with BandMF (compare utility improvement)
- [ ] Verify memory usage < 24GB (single A100)

**Week 3-4: Large-Scale Validation**
- [ ] End-to-end LoRA fine-tuning experiments
  - Llama-7B on Alpaca (instruction tuning)
  - Llama-13B with gradient checkpointing
  - Compare: DP-SGD vs DP-FTRL/BandMF utility
- [ ] Compare utility against published baselines
- [ ] Performance benchmarking (tokens/sec)
- [ ] Memory optimization guide

**Week 5-6: High-Level API (Optional)**
- [ ] **Consider**: DP Execution Plans (`opaque.execution_plan`)
  - High-level API packaging all DP components
  - `DPExecutionPlan` dataclass with validated composition
  - Config classes: `DPSGDConfig`, `BandMFConfig`, etc.
  - Automatic noise calibration from (ε, δ) budget
  - **Decision**: Implement if users request simplified API, otherwise defer
- [ ] Alternative: Document manual composition patterns
- [ ] Tutorial: Composing DP components

**Success Criteria**:
- ✅ LoRA fine-tune Llama-7B on Alpaca (single A100, <24GB)
- ✅ Llama-13B works with gradient checkpointing (<40GB)
- ✅ BandMF shows utility improvement over DP-SGD on LoRA task
- ✅ Achieve published utility benchmarks (within 5% accuracy)

**Deliverables**:
- LoRA integration module
- Large-scale validation results
- Performance benchmarks
- Memory optimization guide
- (Optional) High-level execution plan API

---

### Phase 5: Distributed Training (4-6 weeks)

**Goal**: Support multi-GPU/multi-node DP training for models >10B parameters

**Week 1-2: Sharding Utilities**
- [ ] Implement `opaque.distributed.sharding_utils` module
  - `flatten_with_zero_redundancy()` - ZeRO-style sharding for gradients
  - `local_reshape_add()` - Add noise without cross-device communication
  - `compute_early_stopping_order()` - Optimal data ordering for microbatching
- [ ] Integration with PyTorch distributed primitives
  - DDP (DistributedDataParallel) compatibility
  - FSDP (FullyShardedDataParallel) integration
  - Device mesh and sharding strategies
- [ ] Tests: Validate sharding, noise distribution

**Week 3-4: Distributed Noise Generation**
- [ ] Implement distributed BandMF noise generation
  - No cross-device communication required (arxiv.org/abs/2405.15913)
  - Per-device PRNG seeding strategy
  - Gradient accumulation across devices
- [ ] Integration with matrix factorization mechanisms
- [ ] Tests: Verify privacy guarantees under distributed training

**Week 5-6: Large-Scale Validation**
- [ ] Multi-GPU training experiments
  - 2-8 GPU training on single node
  - Llama-7B/13B full fine-tuning (not just LoRA)
  - Measure scaling efficiency
- [ ] Multi-node training (if infrastructure available)
- [ ] Performance benchmarking vs single-GPU
- [ ] Documentation: Distributed training best practices

**Success Criteria**:
- ✅ DDP/FSDP compatibility validated
- ✅ Distributed noise generation maintains privacy guarantees
- ✅ Linear scaling efficiency up to 8 GPUs
- ✅ Llama-13B full fine-tuning works (multi-GPU)

**Deliverables**:
- Distributed training utilities
- DDP/FSDP integration
- Distributed BandMF implementation
- Scaling benchmarks
- Multi-GPU training guide

---

### Phase 6: Production Polish & Release (3-4 weeks)

**Goal**: Production-ready library with comprehensive documentation and tooling

**Week 1: Documentation**
- [ ] Update committed docs (see review checklist below)
- [ ] Getting Started guide (10-minute tutorial)
- [ ] User Guide updates (new features: matrix factorization, distributed)
- [ ] API Reference (auto-generated from docstrings)
- [ ] Migration Guide (Opacus → Opaque)
- [ ] Troubleshooting (common errors, solutions)
- [ ] Performance tuning guide

**Week 2: Testing & CI**
- [ ] Expand test coverage to >80%
- [ ] Add integration tests for large models (CI with GPU)
- [ ] Performance regression tests
- [ ] Memory usage tests
- [ ] Distributed training tests (if CI supports multi-GPU)

**Week 3: Advanced Features Documentation**
- [ ] Matrix factorization tutorial (BandMF, DP-FTRL)
- [ ] Empirical auditing guide
- [ ] Distributed training tutorial
- [ ] Batch selection strategies comparison
- [ ] When to use which mechanism (decision tree)

**Week 4: Release Preparation**
- [ ] Finalize API (no breaking changes after 1.0)
- [ ] Code review and cleanup
- [ ] Prepare release notes
- [ ] Create release checklist
- [ ] Security review (DP guarantees audit)

**Success Criteria**:
- ✅ Documentation covers all user scenarios
- ✅ Test coverage >80%
- ✅ CI passes on CPU and GPU
- ✅ All committed docs updated and accurate
- ✅ Security review complete
- ✅ Ready for production use

**Deliverables**:
- Complete documentation site
- Comprehensive test suite
- Production troubleshooting guide
- Release-ready codebase
- **v1.0.0 release**

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
| **Phase 1** | 6-8 weeks | Memory & Core Production | Microbatching, profiling, checkpointing, auditing, Opacus validation |
| **Phase 2** | 3-4 weeks | Batch Selection & Accounting | Sampling strategies, multi-epoch accounting |
| **Phase 3** | 6-8 weeks | Matrix Factorization & DP-FTRL | BandMF, BLT, correlated noise, sensitivity analysis |
| **Phase 4** | 4-6 weeks | Scale to Billions & LoRA | LoRA integration, large-scale validation, execution plans |
| **Phase 5** | 4-6 weeks | Distributed Training | Sharding, distributed noise, multi-GPU/multi-node |
| **Phase 6** | 3-4 weeks | Production Polish & Release | Documentation, testing, security review, v1.0.0 |
| **Total** | **26-36 weeks** | **~6-9 months** | **Production-ready v1.0.0 with all features** |

**Note**: This is the complete v1.0 implementation. There is no v2.0 - all features are in this effort.

---

## 11. Committed Documentation Review Checklist

**Committed docs requiring updates** (run `git ls-files docs/`):

### User-Facing Docs (High Priority)
- [ ] `docs/getting-started/quickstart.md` - Add memory profiling, microbatching examples
- [ ] `docs/user-guide/clipping.md` - Document microbatching parameter
- [ ] `docs/user-guide/noise.md` - Add matrix factorization noise mechanisms
- [ ] `docs/user-guide/accounting.md` - Multi-epoch accounting, batch selection integration
- [ ] `docs/user-guide/optimizers.md` - Add DP-FTRL optimizer
- [ ] `docs/user-guide/sampling.md` - Complete batch selection strategies
- [ ] `docs/user-guide/lora.md` - Update with Phase 4 LoRA helpers
- [ ] `docs/tutorials/05_sampling_and_microbatching.ipynb` - Actually implement microbatching examples
- [ ] `docs/tutorials/06_lora_huggingface_dp_training.ipynb` - Update with production patterns

### API Docs (Medium Priority)
- [ ] `docs/api/core/clipping.md` - Document microbatching, gradient checkpointing
- [ ] `docs/api/noise.md` - Add matrix factorization mechanisms
- [ ] `docs/api/accounting.md` - Multi-epoch, batch selection integration
- [ ] `docs/api/optimizers.md` - Add DP-FTRL
- [ ] `docs/api/sampling.md` - Complete batch selection API

### Development Docs (Keep as Reference, Update for Accuracy)
- [ ] `docs/development/STATUS.md` - Update with new phase plan
- [ ] `docs/development/RFC_PRODUCTION_PLAN.md` - **This file** (being updated now)
- [ ] `docs/development/DESIGN_COMPARISON_EXAMPLES.md` - Add matrix factorization examples
- [ ] `docs/development/tdd-workflow.md` - No changes needed

### New Docs Required
- [ ] `docs/user-guide/matrix-factorization.md` - NEW: BandMF, DP-FTRL guide
- [ ] `docs/user-guide/auditing.md` - NEW: Empirical privacy auditing
- [ ] `docs/user-guide/distributed.md` - NEW: Multi-GPU/multi-node training
- [ ] `docs/user-guide/memory-optimization.md` - NEW: Profiling, checkpointing, microbatching
- [ ] `docs/tutorials/07_matrix_factorization_bandmf.ipynb` - NEW: BandMF tutorial
- [ ] `docs/tutorials/08_empirical_auditing.ipynb` - NEW: Auditing tutorial
- [ ] `docs/tutorials/09_distributed_training.ipynb` - NEW: Multi-GPU tutorial

**Note**: `docs/development/` files are **local reference only** - accuracy is important but they won't be in user-facing documentation builds.

---

## 12. Conclusion

This RFC presents a comprehensive plan to evolve Opaque from a functional prototype to a **production-ready, research-grade DP training library** with all advanced features. The plan prioritizes:

1. **Memory efficiency** (Phase 1) - Microbatching, checkpointing, profiling
2. **Validation** (Phase 1) - Opacus parity, empirical auditing
3. **Batch selection** (Phase 2) - Complete sampling strategies
4. **Advanced mechanisms** (Phase 3) - Matrix factorization, BandMF, DP-FTRL
5. **Scale** (Phase 4) - Llama-7B/13B LoRA fine-tuning
6. **Distributed** (Phase 5) - Multi-GPU/multi-node support
7. **Polish** (Phase 6) - Documentation, testing, security review

**Estimated timeline**: 6-9 months to feature-complete v1.0.0 release.

**Key Insight**: This is the **complete v1.0 implementation** - no v2.0 planned. All features from JAX-Privacy (except JAX-specific utilities) will be implemented.

**Next Steps**:
1. Review and approve this RFC
2. Begin Phase 1 Week 1: Microbatching implementation (fix the declared-but-not-implemented parameter)
3. Set up weekly progress reviews
4. Track progress against 26-36 week timeline
