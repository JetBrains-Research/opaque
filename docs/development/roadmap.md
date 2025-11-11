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

## Stage 2: Noise Injection 🔜 **FUTURE**

**Timeline**: 2 weeks

**Goal**: Add Gaussian noise to clipped gradients

**Deliverables**:
1. `opaque.core.noise` - Noise generation module
2. Reproducible noise with `torch.Generator`
3. Integration with `clipped_grad()`
4. Tests validating noise properties

**Key Features**:
- Gaussian noise with correct scaling
- PRNG for reproducibility
- Noise multiplier calibration
- Integration with clipping

---

## Stage 3: Privacy Accounting 🔜 **FUTURE**

**Timeline**: 2 weeks

**Goal**: Track privacy budget (ε, δ) across training

**Deliverables**:
1. `opaque.accounting` - Privacy budget tracking
2. Wrap Google's `dp-accounting` library
3. Noise calibration for target (ε, δ)
4. Budget validation during training

**Key Features**:
- RDP (Rényi Differential Privacy) accounting
- Privacy Loss Distribution (PLD)
- Automatic noise calibration
- Training step validation

---

## Stage 4: High-Level API 🔜 **FUTURE**

**Timeline**: 2 weeks

**Goal**: User-friendly API for common use cases

**Deliverables**:
1. `opaque.api.make_private()` - One-line DP wrapper
2. `DPConfig` - Configuration dataclass
3. Integration with Hugging Face `peft` library
4. LoRA parameter detection

**Key Features**:
- Automatic LoRA adapter detection
- Simple configuration
- Validation and error checking
- Examples with real LLMs

---

## Stage 5: Optimization & Polish 🔜 **FUTURE**

**Timeline**: 2-3 weeks

**Goal**: Production-ready quality

**Deliverables**:
1. Performance optimizations
2. Comprehensive documentation
3. Tutorial notebooks
4. PyPI package publication

**Tasks**:
- Profile and optimize bottlenecks
- Add GPU-specific optimizations
- Write migration guide from Opacus
- Create video tutorials
- Publish to PyPI

---

## Long-term Goals

### Beyond Stage 5

**Potential Future Work**:
- Support for more layer types (Conv, Attention)
- Distributed training support
- Integration with other PEFT methods (Prefix Tuning, Adapters)
- DP-FTRL implementation
- Privacy amplification by sampling
- Adaptive clipping

**Community Building**:
- Example gallery
- Blog posts and tutorials
- Conference presentations
- Research collaborations

---

## Timeline Summary

| Stage | Duration | Status |
|-------|----------|--------|
| Stage 0: Planning | 1 week | ✅ Complete |
| Stage 1: Clipping | 3 weeks | ✅ Complete (2025-11-11) |
| Stage 2: Noise | 2 weeks | 📋 Ready |
| Stage 3: Accounting | 2 weeks | 🔜 Future |
| Stage 4: High-Level API | 2 weeks | 🔜 Future |
| Stage 5: Polish | 2-3 weeks | 🔜 Future |
| **Total** | **~12 weeks** | **Stage 1 Complete** |

---

## Contribution Opportunities

Want to help? See areas where contributions are welcome:

- **Documentation**: Improve tutorials and examples
- **Testing**: Add edge case tests and property-based tests
- **Validation**: Cross-check with JAX-Privacy on more examples
- **Examples**: Create real-world use case examples
- **Performance**: Profile and optimize bottlenecks

See [Contributing Guide](contributing.md) for how to get started.
