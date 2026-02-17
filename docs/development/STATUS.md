# Opaque Status & Roadmap

**Last Updated**: 2026-02-17
**Current Version**: v0.1.0-alpha
**Status**: Stages 1-2 + Distributed Training Complete → Ready for Team Review

---

## Quick Status

**Completed**:
- ✅ Core clipping API (JAX-Privacy parity)
- ✅ Noise injection (Gaussian + MF via functional API)
- ✅ Functional accounting API (to be migrated - see RFC_ACCOUNTING_MIGRATION.md)
- ✅ TorchOpt optimizer wrappers with adaptive clipping
- ✅ **570 tests passing** (35 distributed + 5 HF models + core tests)
- ✅ GPT-2 (124M) & real HF models integration validated
- ✅ **Distributed DDP training** - Multi-GPU LoRA fine-tuning with deterministic noise
- ✅ **Spawn-based testing** - No torchrun dependencies, mp.spawn-based workers
- ✅ HF model compatibility (Qwen2, TinyLlama, Phi-3) with vmap patches

**Production Ready for**:
- 🎯 Multi-GPU LoRA fine-tuning with DP-SGD
- 🎯 Real transformer models (8B-1.5B scale validated)
- 🎯 Deterministic distributed noise (synchronized across ranks)

**In Scope for Future Work** (not blockers):
- 📋 Memory profiling & optimization
- 📋 FSDP support (currently DDP-only, not planned for v1.0)
- 📋 Streaming matrix CUDA device handling (BandMF/BLT on GPU)
- 📋 Multi-epoch correlated noise tests
- 📋 v1.0.0 release when ready

---

## For Agents: Starting Phase 1A

**If you're an agent tasked with Phase 1A implementation**, see:

1. **[RFC: Production Plan](RFC_PRODUCTION_PLAN.md)** - Section 5, Phase 1A for complete task breakdown
2. **Key tasks**:
   - Choose LoRA model: Llama-3-8B or Mistral-7B
   - Task: Instruction tuning (Alpaca/Dolly)
   - Get non-DP LoRA baseline working
   - Integrate Opaque DP training (`clipped_grad()` + `gaussian_noise()`)
   - Cross-validate with JAX-Privacy (gradient + utility equivalence)
   - Document all issues encountered
   - Implement basic microbatching if OOM occurs
   - Create end-to-end tutorial notebook

3. **Success criteria**:
   - ✅ 8B LoRA model fine-tunes end-to-end with DP
   - ✅ Matches JAX-Privacy gradient output (atol=1e-5)
   - ✅ Achieves reasonable utility (comparable to JAX-Privacy)
   - ✅ Tutorial demonstrates complete workflow

**Environment**: H200 GPU with 80GB memory

---

## Complete Plan

See **[RFC: Production Plan](RFC_PRODUCTION_PLAN.md)** for:
- Complete 6-phase implementation plan (~6-9 months to v1.0.0)
- Phase 1A: LoRA validation at scale (4-6 weeks) 🎯 **START HERE**
- Phase 1B: Memory profiling & optimization (2-3 weeks)
- Phase 1C: Empirical privacy auditing (2-3 weeks)
- Phase 2: Functional API polish (1-2 weeks)
- Phase 3: BandMF & DP-FTRL (6-8 weeks)
- Phase 4: Larger scale validation - optional (2-3 weeks)
- Phase 5: Distributed training (4-6 weeks)
- Phase 6: Production polish & v1.0.0 (3-4 weeks)

**Key decisions**:
- ✅ LoRA validation moved to Phase 1A (validate on real use case, not toy models)
- ✅ JAX-Privacy comparison (not Opacus - more flexible)
- ✅ Memory profiler = context manager (observe, not predict)
- ✅ Batch selection deferred to Phase 3 (only CyclicPoisson for BandMF)
- ❌ No DP Execution Plans (out of scope for functional v1.0)

---

## Stage History

### Stage 1: Core Clipping ✅ COMPLETE (Nov 2025)

**Deliverables**:
- `opaque.clipping` module (modularized structure)
  - `clip_pytree()`, `clipped_fun()`, `clipped_grad()`
  - Full JAX-Privacy API parity (single-device)
- Tests: 70 passing with JAX validation
- Coverage: ~90%

**Achievements**:
- Numerical equivalence with JAX-Privacy (atol=1e-5)
- All single-device parameters implemented
- Tech debt documented: `microbatch_size`, `prng_argnum`, `spmd_axis_name`

### Stage 2: Noise & Accounting ✅ COMPLETE (Nov 2025)

**Deliverables**:
- `opaque.noise` module
  - `gaussian_noise()` - Stateless Gaussian noise
- `opaque.accounting` module (functional API)
  - Composition: `compose_gaussian()`, `compose_poisson_gaussian()`, etc.
  - Queries: `get_epsilon()`, `get_beta()`, `get_advantage()`
  - Calibration: Using riskcal primitives
  - **Note**: Will be migrated to separate library (see RFC_ACCOUNTING_MIGRATION.md)
- `opaque.optimizers` module
  - `adaptive_clipping()` wrapper for TorchOpt
- Tests: 111 passing (55 accounting + 56 optimizer)

**Achievements**:
- Functional API (immutable state, pure functions)
- Truncated Poisson sampling (solves variable batch problem)
- Parallel test execution (3.17× speedup with pytest-xdist)
- Tutorial notebook restructured

**Archived Documentation**:
- Stage 1-2 detailed progress docs moved to `archive/stages_1_2_complete/`

---

## Production Readiness Assessment

### Current State
- ✅ **Core primitives work** (clipping, noise, accounting)
- ✅ **Small model validation** (GPT-2 124M)
- ❌ **No large-scale validation** (8B+ models untested)
- ❌ **No LoRA integration testing**
- ❌ **Memory optimization missing** (microbatching not implemented)
- ❌ **Limited cross-validation** (JAX-Privacy not tested)

### Critical Path: Phase 1A
**Phase 1A is the blocker for everything else.** Must prove library works on production workload (LoRA @ 8B scale) before proceeding with advanced features.

---

## Timeline to v1.0.0

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| **Phase 1A** 🎯 | 4-6 weeks | **LoRA @ 8B Scale** | Validation, JAX comparison, tutorial |
| Phase 1B | 2-3 weeks | Memory Optimization | Microbatching, profiler |
| Phase 1C | 2-3 weeks | Empirical Auditing | Auditing module |
| Phase 2 | 1-2 weeks | API Polish | Based on Phase 1A learnings |
| Phase 3 | 6-8 weeks | BandMF & DP-FTRL | Matrix factorization |
| Phase 4 | 2-3 weeks | Optional Scaling | 13B+ if needed |
| Phase 5 | 4-6 weeks | Distributed | Multi-GPU |
| Phase 6 | 3-4 weeks | Polish & Release | Docs, tests, v1.0.0 |
| **Total** | **~6-9 months** | | **Production-ready library** |

---

## References

- **[RFC: Production Plan](RFC_PRODUCTION_PLAN.md)** - Complete implementation plan with all phases
- **[Design Comparison](DESIGN_COMPARISON_EXAMPLES.md)** - Functional API design examples
- **[TDD Workflow](tdd-workflow.md)** - Development process
- **[CLAUDE.md](../../CLAUDE.md)** - Agent briefing (current context)
- **[Archive](archive/stages_1_2_complete/)** - Completed Stages 1-2 documentation
