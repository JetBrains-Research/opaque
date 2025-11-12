# Project Roadmap

This document outlines the complete implementation roadmap for Opaque.

---

## Stage 0: Planning & Setup ✅ **COMPLETED**

**Timeline**: Initial setup

**Deliverables**:
- [x] Repository structure created
- [x] Dependencies configured (`pyproject.toml`)
- [x] Testing framework set up (pytest + JAX validation)
- [x] Code quality tools (ruff for linting/formatting)
- [x] Documentation framework (Material for MkDocs)
- [x] TDD workflow defined

**Planning Documents**:
- [x] JAX-Privacy architecture analysis
- [x] PyTorch port strategy
- [x] Detailed Stage 1 implementation plan
- [x] Design decisions documented

---

## Stage 1: Core Clipping Module ✅ **COMPLETED**

**Timeline**: 3 weeks (Completed 2025-11-11)

**Goal**: Implement per-example gradient clipping without noise

**Deliverables**:
- [x] `opaque.pytree_utils` - PyTree operations (152 LOC)
- [x] `opaque.clipping` - Full clipping API (700 LOC)
  - `clip_pytree()` - Low-level PyTree clipping
  - `clipped_fun()` - Primary API for clipping function outputs
  - `clipped_grad()` - High-level gradient clipping API
  - `BoundedSensitivityCallable` - Wrapper with sensitivity tracking
  - `AuxiliaryOutput` - Named tuple for auxiliary outputs
- [x] Tests: 79 tests passing (34 unit + 45 JAX validation)
- [x] JAX-Privacy numerical validation (atol=1e-5)
- [x] 80% code coverage

**Implementation Highlights**:
- Full API parity with JAX-Privacy main branch (single-device features)
- All parameters implemented except `microbatch_size`, `prng_argnum`, `spmd_axis_name` (documented as tech debt)
- Created `_value_and_grad()` helper to bridge PyTorch/JAX API differences
- Workaround for PyTorch vmap None-handling limitation
- Numerical validation against JAX-Privacy within 1e-5 tolerance

**Known Limitations** (documented as tech debt):
- `microbatch_size` - Deferred until Stage 3 (requires sophisticated implementation)
- `prng_argnum` - Deferred (requires PRNG key splitting, no PyTorch equivalent)
- `spmd_axis_name` - Deferred (distributed training feature)

**See**: [Stage 1 Detailed Plan](stage1-plan.md)

---

## Stage 2: Noise Injection & Privacy Accounting 📋 **READY**

**Timeline**: 3 weeks

**Goal**: Add Gaussian noise to clipped gradients and track privacy budget

**Deliverables**:

1. `opaque.noise` - Simple i.i.d. Gaussian noise generation
  - `add_gaussian_noise()` - Stateless functional API
  - Reproducible noise with `torch.Generator`
  - Statistical validation tests (normality, stddev)
  - JAX-Privacy numerical equivalence validation
2. `opaque.accounting` - Privacy budget tracking
  - Wrap Google's `dp-accounting` library
  - RDP (Rényi Differential Privacy) accounting
  - Noise calibration for target (ε, δ)
  - Budget validation during training

**Key Features**:

- Simple i.i.d. Gaussian noise N(0, stddev²)
- PyTree support via `tree_map()`
- Stateless API (no state management)
- Integration example with `clipped_grad()`
- Privacy Loss Distribution (PLD) accounting
- Automatic noise multiplier calibration
- Training step privacy validation
- Composition theorems for multi-epoch training

**See**: [Stage 2 Detailed Plan](stage2-plan.md)

---

## Stage 3: Functional Optimizers & Advanced Clipping 🔜 **FUTURE**

**Timeline**: 3 weeks

**Goal**: Implement functional optimizers and advanced clipping mechanisms

**Deliverables**:

1. `opaque.optimizers` - Functional optimizer implementations
  - SGD with DP-SGD support
  - Adam with DP-Adam support
  - AdaClip (adaptive clipping)
2. Integration with clipping and noise
3. Support for DP-FTRL (Follow-The-Regularized-Leader)
4. Per-layer clipping strategies

**Key Features**:

- Functional optimizer interface matching `torch.func`
- AdaClip: Adaptive clipping based on gradient quantiles
- Stateless optimizer updates
- Integration with existing `clipped_grad()` API

---

## Stage 4: Privacy Amplification by Sampling 🔜 **FUTURE**

**Timeline**: 2 weeks

**Goal**: Implement privacy amplification through subsampling

**Deliverables**:

1. `opaque.sampling` - Privacy-aware data loading
  - Poisson sampling for batch selection
  - Secure shuffling with fixed privacy cost
  - Batch size accounting
2. `opaque.microbatching` - Memory-efficient microbatching
  - Gradient accumulation over microbatches
  - Memory vs. privacy tradeoffs
  - Automatic microbatch size selection
3. Integration with privacy accounting (from Stage 2)

**Key Features**:

- Poisson sampling with sampling probability q
- Privacy amplification analysis (tighter ε bounds)
- Microbatching to handle large logical batch sizes
- Memory-efficient gradient accumulation
- Compatible with PyTorch DataLoader
- Automatic batch/microbatch sizing
- Update accounting to use amplification

---

## Stage 5: High-Level API 🔜 **FUTURE**

**Timeline**: 2 weeks

**Goal**: User-friendly API for common use cases

**Deliverables**:
1. `opaque.api.make_private()` - One-line DP wrapper
2. `DPConfig` - Configuration dataclass
3. Integration with Hugging Face `peft` library
4. LoRA parameter detection
5. End-to-end training examples

**Key Features**:
- Automatic LoRA adapter detection
- Simple configuration with sensible defaults
- Validation and error checking
- Examples with real LLMs
- Integration with HuggingFace Trainer

---

## Stage 6: Optimization & Polish 🔜 **FUTURE**

**Timeline**: 2-3 weeks

**Goal**: Production-ready quality

**Deliverables**:
1. Performance optimizations
2. Comprehensive documentation
3. Tutorial notebooks
4. PyPI package publication

**Key Features**:
- Profile and optimize bottlenecks
- Add GPU-specific optimizations
- Write migration guide from Opacus
- Create video tutorials
- Publish to PyPI

---

## Long-term Goals

### Beyond Stage 6

**Potential Future Work**:

- **Correlated Noise**: Matrix factorization privatizer from JAX-Privacy
  - Dense matrix implementation
  - StreamingMatrix for memory-efficient temporal correlation
  - Advanced correlation structures (prefix-sum, banded matrices)
- **Distributed Training**: Multi-GPU and multi-node support
  - SPMD (Single Program Multiple Data) training
  - Distributed noise generation with sharding
  - Cross-device gradient aggregation
- **Advanced Features**:
  - Support for more layer types (Conv, Attention)
  - Integration with other PEFT methods (Prefix Tuning, Adapters)
  - Virtual sequences for long-context models
  - Gradient checkpointing integration
- **Research Extensions**:
  - DP-FTRL variants and optimizations
  - Adaptive privacy budgets
  - Privacy-utility tradeoff visualization

**Community Building**:
- Example gallery
- Blog posts and tutorials
- Conference presentations
- Research collaborations

---

## Timeline Summary

| Stage                          | Duration      | Status                  |
|--------------------------------|---------------|-------------------------|
| Stage 0: Planning              | 1 week        | ✅ Complete              |
| Stage 1: Clipping              | 3 weeks       | ✅ Complete (2025-11-11) |
| Stage 2: Noise & Accounting    | 3 weeks       | 📋 Ready                |
| Stage 3: Functional Optimizers | 3 weeks       | 🔜 Future               |
| Stage 4: Privacy Amplification | 2 weeks       | 🔜 Future               |
| Stage 5: High-Level API        | 2 weeks       | 🔜 Future               |
| Stage 6: Polish                | 2-3 weeks     | 🔜 Future               |
| **Total**                      | **~16 weeks** | **Stage 1 Complete**    |

---

## Contribution Opportunities

Want to help? See areas where contributions are welcome:

- **Documentation**: Improve tutorials and examples
- **Testing**: Add edge case tests and property-based tests
- **Validation**: Cross-check with JAX-Privacy on more examples
- **Examples**: Create real-world use case examples
- **Performance**: Profile and optimize bottlenecks

See [Contributing Guide](contributing.md) for how to get started.
