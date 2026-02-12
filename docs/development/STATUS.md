# Opaque Status & Roadmap

**Last Updated**: 2026-02-12
**Current Version**: v0.1.0-alpha
**Status**: Prototype → Production Transition

---

## Quick Status

**Completed** (Stages 1-2):
- ✅ Core clipping API (111 tests passing)
- ✅ Functional accounting API
- ✅ GPT-2 integration validated

**In Progress** (Phase 1):
- 🔄 Memory optimization (microbatching)
- 🔄 Opacus cross-validation
- 🔄 Production hardening

**Planned**:
- 📋 Phase 2: Functional API refactoring
- 📋 Phase 3: Scale to Llama-7B/13B
- 📋 Phase 4: v1.0.0 release

---

## For Complete Plan

See **[RFC: Production Plan](RFC_PRODUCTION_PLAN.md)** for:
- Complete production readiness assessment
- 4-phase implementation plan (~16 weeks)
- Architectural decisions (functional design)
- Validation strategy
- Timeline and deliverables

---

## Stage History

### Stage 1: Core Clipping ✅ COMPLETE (Nov 2025)

**Deliverables**:
- `opaque.clipping` module (700 LOC)
  - `clip_pytree()`, `clipped_fun()`, `clipped_grad()`
  - Full JAX-Privacy API parity (single-device)
- Tests: 79 passing (34 unit + 45 JAX validation)
- Coverage: 80%

**Achievements**:
- Numerical equivalence with JAX-Privacy (atol=1e-5)
- All single-device parameters implemented
- Tech debt documented: `microbatch_size`, `prng_argnum`, `spmd_axis_name`

### Stage 2: Noise & Accounting ✅ COMPLETE (Nov 2025)

**Deliverables**:
- `opaque.noise` module
  - `add_gaussian_noise()` - Stateless Gaussian noise
- `opaque.accounting` module (functional API)
  - Composition: `compose_gaussian()`, `compose_poisson_gaussian()`, etc.
  - Queries: `get_epsilon()`, `get_beta()`, `get_advantage()`
  - Calibration: Using riskcal primitives
- `opaque.optimizers` module
  - `adaptive_clipping()` wrapper for TorchOpt
- Tests: 111 passing (55 accounting + 56 optimizer)

**Achievements**:
- Functional API (immutable state, pure functions)
- Truncated Poisson sampling (solves variable batch problem)
- Parallel test execution (3.17× speedup)
- Tutorial notebook restructured

---

## Next: Phase 1 (Production Hardening)

**Timeline**: 4-6 weeks
**Focus**: Memory efficiency + validation

**Week 1-2**: Memory Management
- Implement microbatching in `clipped_grad()`
- Memory profiling tools (`estimate_memory()`, `recommend_microbatch_size()`)

**Week 3-4**: Cross-Validation
- Validate against Opacus (gradient, privacy, utility)
- Test matrix: MNIST, CIFAR-10, Wikitext

**Week 5-6**: Large Model Testing
- GPT-2-Large (774M) with microbatching
- GPT-2-XL (1.5B) with gradient checkpointing
- Troubleshooting guide

**Success Criteria**:
- ✅ Train GPT-2 without OOM (batch=32, microbatch=8)
- ✅ Match Opacus within 0.1 epsilon
- ✅ Match Opacus within 2% accuracy

---

## Production Readiness Gaps

### 🔥 BLOCKER: Memory Efficiency
- `vmap` materializes all per-example grads (31GB for GPT-2 batch=64)
- **Solution**: Microbatching + memory profiling (Phase 1)

### ❌ HIGH PRIORITY: Validation
- No Opacus cross-validation yet
- No utility benchmarks
- **Solution**: Systematic validation tests (Phase 1)

### ❌ Scale Testing
- Only tested GPT-2 (124M params)
- LoRA integration untested
- **Solution**: Large model testing (Phase 3)

### ❌ Documentation
- Limited production guides
- No migration guide from Opacus
- **Solution**: Comprehensive docs (Phase 4)

---

## Timeline to v1.0.0

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 4-6 weeks | Microbatching, Opacus validation |
| Phase 2 | 3-4 weeks | Functional API (JAX-Privacy patterns) |
| Phase 3 | 4-6 weeks | Llama-7B LoRA, gradient checkpointing |
| Phase 4 | 2-3 weeks | Documentation, v1.0.0 release |
| **Total** | **~16 weeks** | **Production-ready library** |

---

## References

- **[RFC: Production Plan](RFC_PRODUCTION_PLAN.md)** - Complete implementation plan
- **[Design Comparison](DESIGN_COMPARISON_EXAMPLES.md)** - API design examples
- **[TDD Workflow](tdd-workflow.md)** - Development process
- **[CLAUDE.md](../../CLAUDE.md)** - Agent briefing (current context)
