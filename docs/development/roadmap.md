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

## Stage 2: Noise Injection & Privacy Accounting ✅ **COMPLETED**

**Timeline**: 3 weeks (Completed 2025-11-12)

**Goal**: Add Gaussian noise to clipped gradients and track privacy budget

**Deliverables**:

1. [x] `opaque.noise` - Simple i.i.d. Gaussian noise generation
  - `add_gaussian_noise()` - Stateless functional API
  - Reproducible noise with `torch.Generator`
  - Statistical validation tests (normality, stddev)
  - JAX-Privacy numerical equivalence validation

2. [x] `opaque.accounting` - Privacy budget tracking

- Uses Google's `dp-accounting` library
- `PLDAccountant` - PLD (Privacy Loss Distribution) accounting for Poisson sampling
- `RDPAccountant` - RDP (Rényi Differential Privacy) accounting for fixed-size mini-batches
- Truncated Poisson sampling support
- Three calibration functions: `calibrate_noise_multiplier`, `calibrate_steps`, `calibrate_batch_size`

**Implementation Highlights**:

- [x] Simple i.i.d. Gaussian noise N(0, stddev²)
- [x] PyTree support via `tree_map()`
- [x] Stateless API (no state management)
- [x] Privacy Loss Distribution (PLD) accounting
- [x] Automatic noise multiplier calibration
- [x] Training step privacy validation
- [x] Flexible accountant instantiation (string/class/callable)
- [x] **Truncated Poisson sampling** - Solves variable batch size problem
- [x] 30 unit tests + 13 JAX validation tests = 43 tests passing
- [x] Numerical equivalence with JAX-Privacy confirmed (tolerance < 0.01-0.1 epsilon)

**See**: [Stage 2 Detailed Plan](stage2-plan.md)

---

## Stage 3: Functional Optimizers & DP-Adam-AC 🔜 **READY TO START**

**Timeline**: 4-5 weeks

**Goal**: Implement functional optimizers with TorchOpt, including state-of-the-art DP-Adam-AC

**Deliverables**:

1. `opaque.optimizers` - Functional optimizer implementations (~400 LOC)

- DP-SGD using TorchOpt
- DP-Adam using TorchOpt
- **DP-Adam-AC** (Adaptive Clipping) from [arxiv:2510.05288](https://arxiv.org/abs/2510.05288)

2. `opaque.adaptive` - Adaptive clipping infrastructure (~150 LOC)

- Gradient norm buffer with percentile tracking
- Clip-rate-based learning rate scaling

3. Tutorial 03 - Updated with DP-Adam-AC comparison
4. Comprehensive tests (~400 LOC)

**Key Features**:

- **TorchOpt Integration**: JAX-like functional optimizers (Optax pattern)
- **DP-Adam-AC**: Adaptive clipping that adjusts threshold based on gradient percentiles
- **Dynamic LR Scaling**: Learning rate adjusts based on clip rate
- **EMA Smoothing**: Exponential moving average for better privacy-utility tradeoff
- Stateless optimizer updates matching our functional design
- Full integration with existing `clipped_grad()`, `add_gaussian_noise()`, accounting APIs

**Why DP-Adam-AC?**:

- 📈 1-3% accuracy improvement over fixed clipping
- 🎯 Self-tuning: adapts to gradient scale during training
- 🔒 Same privacy guarantees as standard DP-Adam
- 🚀 State-of-the-art for LLM fine-tuning (October 2024 paper)

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

- **Advanced Privacy Accounting**:
  - **User-level privacy accounting**: Privacy for users with multiple examples in dataset
    - Requires mixture of Gaussians analysis
    - Uses hypergeometric distribution for sampling
    - Reference: `DpsgdTrainingUserLevelAccountant` in JAX-Privacy
  - **Single-release analysis for DP-FTRL**: Un-amplified analysis treating training as single DP event
    - No subsampling amplification
    - Reference: `SingleReleaseTrainingAccountant` in JAX-Privacy
  - **Cyclic Poisson sampling**: For correlated noise with BandMF
    - Supports partitioned sampling across cycles
    - Reference: JAX-Privacy `analysis.py` cycle_length parameter
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

| Stage                          | Duration      | Status                   |
|--------------------------------|---------------|--------------------------|
| Stage 0: Planning              | 1 week        | ✅ Complete               |
| Stage 1: Clipping              | 3 weeks       | ✅ Complete (2025-11-11)  |
| Stage 2: Noise & Accounting    | 3 weeks       | ✅ Complete (2025-11-12)  |
| Stage 3: Functional Optimizers | 3 weeks       | 🔜 Future                |
| Stage 4: Privacy Amplification | 2 weeks       | 🔜 Future                |
| Stage 5: High-Level API        | 2 weeks       | 🔜 Future                |
| Stage 6: Polish                | 2-3 weeks     | 🔜 Future                |
| **Total**                      | **~16 weeks** | **Stage 1 & 2 Complete** |

---

## Contribution Opportunities

Want to help? See areas where contributions are welcome:

- **Documentation**: Improve tutorials and examples
- **Testing**: Add edge case tests and property-based tests
- **Validation**: Cross-check with JAX-Privacy on more examples
- **Examples**: Create real-world use case examples
- **Performance**: Profile and optimize bottlenecks

See [Contributing Guide](contributing.md) for how to get started.
