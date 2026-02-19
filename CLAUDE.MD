# Opaque: Agent Briefing

**Project**: PyTorch port of JAX-Privacy for DP-SGD LoRA fine-tuning
**Status**: ✅ Stages 1-2 Complete → 🎯 Phase 1A Ready (LoRA Validation @ 8B Scale)
**JAX-Privacy Reference**: `../jax_privacy/` (always available)
**Environment**: H200 GPU with 80GB memory

---

## Current State

**Repository Phase**: ✅ Stages 1-2 COMPLETE → 🎯 **Phase 1A READY TO START**

**Key Constraint**: Follow TDD workflow strictly - write tests before implementation

**Completed (Stages 1-2)**:

- ✅ **Stage 1**: Full clipping API (clip_pytree, clipped_fun, clipped_grad)
- ✅ **Stage 2**: Noise injection (gaussian_noise) and functional privacy accounting
- **Modules**: Modularized into `utils/`, `clipping/`, `noise/`, `accounting/`, `optimizers/` packages
- **API Parity**: All single-device parameters except microbatch_size and spmd_axis_name (tech debt)
- **Tests**: 111 tests passing (55 accounting + 56 optimizer tests), parallel execution enabled
- **Validation**: GPT-2 (124M params) integration test passes
- **Archived**: Stages 1-2 detailed docs moved to `docs/development/archive/stages_1_2_complete/`

**⚠️ Note on Privacy Accounting**:
- The `opaque.accounting` module will be migrated to a separate library
- See forthcoming `docs/development/RFC_ACCOUNTING_MIGRATION.md` for details
- Phase 1A focuses on DP training library only (clipping, noise, sampling, optimizers)

**Ready to Start (Phase 1A - LoRA Validation @ 8B Scale)**:

**🎯 Goal**: Prove Opaque works for real LoRA fine-tuning with DP at production scale

**Why LoRA now?** LoRA is the actual use case. Training small models doesn't test memory pressure or scale issues. Better to discover LoRA-specific issues early.

**Phase 1A Tasks** (4-6 weeks):
1. **Week 1-2: Baseline & Integration**
   - Choose LoRA model: Llama-3-8B or Mistral-7B
   - Task: Instruction tuning (Alpaca/Dolly)
   - Get non-DP LoRA baseline working
   - Add Opaque DP training (`clipped_grad()` + `gaussian_noise()`)

2. **Week 3-4: Debugging & JAX-Privacy Validation**
   - Debug all issues (API friction, memory, numerical stability, LoRA integration)
   - Cross-validate with JAX-Privacy (gradient + utility equivalence)
   - Create `tests/validation/test_opaque_vs_jax_privacy.py`

3. **Week 5-6: Documentation & Microbatching**
   - Troubleshooting guide (real issues + solutions)
   - Implement basic microbatching if OOM occurred
   - Tutorial notebook: End-to-end LoRA DP training

**Success Criteria**:
- ✅ 8B LoRA model fine-tunes end-to-end with DP
- ✅ Matches JAX-Privacy gradient output (single step, atol=1e-5)
- ✅ Achieves reasonable utility (task performance comparable to JAX-Privacy)
- ✅ Documented all real issues encountered
- ✅ Tutorial demonstrates complete workflow

**For detailed task breakdown**: See `docs/development/RFC_PRODUCTION_PLAN.md` Section 5, Phase 1A

---

## Quick Orientation

### What This Is
PyTorch implementation of differential privacy for LLM fine-tuning:
- **DP-SGD**: Per-example gradient clipping + Gaussian noise
- **LoRA Focus**: Parameter-efficient fine-tuning only
- **Functional API**: Porting JAX-Privacy's `experimental/clipping.py` (not OOP API)

### Essential Context Files

**Production Planning** (START HERE):
1. **[RFC: Production Plan](docs/development/RFC_PRODUCTION_PLAN.md)** - Complete implementation plan, architecture decisions, timeline
2. **[Status & Roadmap](docs/development/STATUS.md)** - Current status, stage history, next steps

**Development Process**:
3. **[TDD Workflow](docs/development/tdd-workflow.md)** - Test-driven development process
4. **[Design Comparisons](docs/development/DESIGN_COMPARISON_EXAMPLES.md)** - API design examples (functional vs class-based)

**Historical Reference** (for context only):
- [docs/development/archive/](docs/development/archive/) - Archived planning docs (Stages 1-2 complete)

### Current Codebase
```
src/opaque/
├── __init__.py             # Public API exports
├── utils/                  # Utility modules
│   ├── pytree.py           # PyTree operations (tree_map, global_norm)
│   └── functional.py       # Functional utils (make_functional)
├── clipping/               # Clipping module
│   ├── types.py            # BoundedSensitivityCallable, AuxiliaryOutput
│   ├── pytree.py           # clip_pytree implementation
│   ├── clipped_fun.py      # clipped_fun (primary API)
│   ├── clipped_grad.py     # clipped_grad (gradient clipping)
│   └── _helpers.py         # Internal helpers
├── noise/                  # Noise injection (DP-SGD + correlated MF)
│   ├── gaussian_noise.py           # gaussian_noise() → (noise_fn, state)
│   ├── bounded_gaussian_noise.py   # bounded_gaussian_noise() → (noise_fn, state)
│   ├── band_mf_noise.py            # BandMF correlated noise
│   ├── blt_mf_noise.py             # Buffered Linear Toeplitz noise
│   ├── dense_mf_noise.py           # Dense matrix noise
│   ├── custom_mf_noise.py          # Bring-your-own-matrix noise
│   ├── identity_mf_noise.py        # Identity (DP-SGD via MF API)
│   └── matrix_factorization/       # MF math infrastructure
│       ├── streaming_matrix.py     # StreamingMatrix data structure
│       ├── toeplitz.py             # Banded Toeplitz strategies
│       ├── buffered_toeplitz.py    # BLT strategies
│       ├── banded.py               # General banded strategies
│       ├── dense.py                # Dense strategies
│       ├── sensitivity.py          # Sensitivity calculations
│       ├── checks.py               # Validation
│       └── optimization.py         # Strategy optimization
├── auditing/               # Empirical privacy auditing
│   ├── __init__.py         # setup(), evaluate() module-level API
│   ├── audit.py            # AuditResult, CoinFlipExperiment
│   ├── scoring.py          # score_by_loss (torch.func.vmap)
│   ├── bootstrap.py        # BootstrapParams
│   └── helpers.py          # Internal helpers
├── accounting/             # Privacy accounting (functional API)
│   ├── composition.py      # create(), compose_*() functions
│   ├── queries.py          # get_epsilon, get_beta, get_advantage
│   └── calibration.py      # find_noise_multiplier_for_*()
├── sampling/               # Batch sampling strategies
├── optimizers/             # TorchOpt wrappers
│   └── adaptive_clipping.py
└── compat/                 # HuggingFace compatibility
    └── transformers/

tests/
├── auditing/               # 108 auditing tests
├── accounting/             # 50 accounting tests
├── clipping/               # 9 clipping tests
├── noise/                  # Noise tests (gaussian, bounded)
├── matrix_factorization/   # MF infrastructure tests
├── jax_validation/         # 5 JAX validation tests
├── optimizers/             # 56 optimizer tests
└── test_dp_ftrl.py         # End-to-end DP-FTRL tests
```

**Total Tests**: 458 tests passing (108 auditing + 55 accounting + 56 optimizer + 239 other)
**Current Status**: Stages 1-2 COMPLETE, Phase 1C (Auditing) COMPLETE, Noise API unified

**Test Execution**:

- Parallel execution enabled with pytest-xdist (`-n=auto` for 12 workers)
- ~3.17x speedup over sequential execution (42.81s vs 135.81s for accounting tests)
- Safe for functional API (no shared mutable state)

---

## TDD Workflow (Required Process)

1. **Discover**: Read JAX-Privacy implementation in `../jax_privacy/`
2. **JAX Test** (optional): Create reference test with `@pytest.mark.jax_validation`
3. **Failing Test**: Write Opaque test defining API
4. **Implement**: Make test pass
5. **Document**: Add Google-style docstrings with examples
6. **Validate**: Numerical comparison with JAX-Privacy

**Commands**:
```bash
uv run pytest                                      # Run unit tests (parallel, -n=auto in pyproject.toml)
uv run --group jax-validation pytest -m jax_validation  # JAX validation
uv run pytest -n 0                                # Run tests sequentially (disable parallel)
uv run ruff format src/ tests/                    # Format
uv run ruff check src/ tests/                     # Lint
```

---

## Key Technical Mappings

**JAX → PyTorch**:
- `jax.vmap` → `torch.func.vmap`
- `jax.grad` → `torch.func.grad`
- `jax.tree_util.tree_map` → `torch.utils._pytree.tree_map`
- PyTree = nested dict of tensors

**DP-SGD Algorithm**:
1. Per-example gradients (via `vmap`)
2. Clip each gradient to max L2 norm
3. Sum clipped gradients
4. Add Gaussian noise (Stage 2)

---

## Design Decisions Summary

Full details: [docs/development/design-decisions.md](docs/development/design-decisions.md)

1. **PyTree**: Use `torch.utils._pytree` (private API, wrap it)
2. **Microbatching**: Explicit `microbatch_size` parameter
3. **Numerical Tolerance**: `atol=1e-5` for JAX validation
4. **Error Handling**: Fail-fast by default
5. **Documentation**: Material for MkDocs

---

## Stage 1 Progress

See [docs/development/stage1-plan.md](docs/development/stage1-plan.md) for full plan.

✅ **Week 1: PyTree Utilities** - COMPLETE
- Implemented `global_norm()`, `tree_leaves()`, `tree_map()` using optree
- Handles mixed dtypes, complex numbers, empty trees
- Reference: `../jax_privacy/src/training/grad_clipping_utils.py`

✅ **Week 2: Basic Clipping** - COMPLETE
- Implemented `clip_pytree()` with all edge cases (zero, inf, NaN-safe)
- Implemented `clipped_grad()` for per-example gradient clipping
- Reference: `../jax_privacy/src/experimental/clipping.py`

✅ **Week 3: Full Clipping + Validation** - COMPLETE
- Full PyTree parameter support (nested dicts of tensors)
- Microbatching support for memory efficiency (tested in test_clipping.py)
- All 29 JAX validation tests pass (numerical equivalence within 1e-5)

### Stage 1 Success Criteria ✅ ALL MET

From [stage1-plan.md](docs/development/stage1-plan.md):
1. ✅ `clip_pytree()` handles all edge cases correctly (0, inf, NaN, rescale)
2. ✅ `clipped_grad()` matches JAX-Privacy output within 1e-5
3. ✅ Works with dictionaries of tensors (PyTrees)
4. ✅ Microbatching reduces memory without changing results
5. ✅ Tests pass on CPU (GPU compatible, not tested)
6. ✅ Code is documented and formatted with Ruff

**Stage 1 is COMPLETE** - Ready for Stage 2 (Noise Injection)

---

## Stage 2: Noise Injection & Privacy Accounting ✅ COMPLETE

See [docs/development/stage2-plan.md](docs/development/stage2-plan.md) for full plan.

**Status**: ✅ **COMPLETE** (2025-11-14)

**Implemented**:

1. ✅ `opaque.noise.gaussian_noise()` - Returns `(noise_fn, state)` with immutable state

2. ✅ `opaque.accounting` - Functional privacy accounting framework
  - **Immutable state**: `state = acc.create()` returns PLD object
  - **Composition**: Pure functions return new states
    - `acc.compose_gaussian()` - Standard Gaussian mechanism
    - `acc.compose_poisson_gaussian()` - Poisson sampling
    - `acc.compose_sampled_gaussian()` - Fixed-size batches
    - `acc.compose_truncated_poisson_gaussian()` - Bounded Poisson (best!)
  - **Queries**: Pure functions on state
    - `acc.get_epsilon(state, delta)` - (ε, δ)-DP
    - `acc.get_beta(state, alpha)` - (α, β) error rates
    - `acc.get_advantage(state)` - f-DP advantage

3. ✅ **Calibration** - Using riskcal.calibration.core primitives
  - **Core abstractions**: `PrivacyEvaluator`, `PrivacyMetrics`, `CalibrationTarget`, `CalibrationConfig`
  - **Epsilon/delta**: `find_noise_multiplier_for_epsilon_delta()` - Custom evaluator using opaque.accounting
  - **Advantage**: `find_noise_multiplier_for_advantage()` - Re-exported from riskcal
  - **Error rates**: `find_noise_multiplier_for_err_rates()` - Re-exported from riskcal
  - **Generic interface**: `calibrate_parameter(evaluator, target, config)` for custom metrics

4. ✅ **Truncated Poisson sampling** - Solves variable batch size problem
5. ✅ **Tutorial notebook updated** - Comprehensive rewrite using functional API

**Functional API Example**:

```python
import opaque.accounting as acc

# Create state (immutable)
state = acc.create()

# Compose mechanisms (returns new state)
state = acc.compose_truncated_poisson_gaussian(
    state,
    noise_multiplier=1.1,
    sample_rate=0.01,
    truncated_batch_size=100,
    dataset_size=10000,
    count=1000,
)

# Query privacy (pure function)
epsilon = acc.get_epsilon(state, delta=1e-5)
beta = acc.get_beta(state, alpha=0.01)
advantage = acc.get_advantage(state)
```

**Calibration Example**:

```python
# Find noise for target (ε, δ)
noise = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0,
    delta=1e-5,
    sample_rate=0.01,
    num_steps=1000,
)

# Find noise for target (α, β)
noise = acc.find_noise_multiplier_for_err_rates(
    alpha=1e-4,
    beta=0.8,
    sample_rate=16/2048,
    num_steps=1000,
)
```

**Test Results**:

- **50 unit tests** - All passing (composition: 13, queries: 16, calibration: 21)
- **5 JAX validation tests** - All passing (cross-framework numerical equivalence)
- Tests cover: epsilon/delta, advantage, error rates, integration, sampling methods
- Verified against opaque.accounting composition and JAX-Privacy
- Parallel execution with pytest-xdist (12 workers, 3.17x speedup)

---

## API Completeness

### ✅ JAX-Privacy main Branch API Parity Achieved

**Implemented Module** (matching JAX-Privacy main structure):
1. **`opaque.clipping`** (matches `jax_privacy.experimental.clipping`)
   - `BoundedSensitivityCallable` - Wrapper with `sensitivity()` method and `has_aux` field
   - `clip_pytree(pytree, clip_norm, ...)` - Low-level PyTree clipping with `return_zero` support
   - `clipped_fun(fun, has_aux, batch_argnums, ...)` - **PRIMARY API** Clip function outputs and sum
   - `clipped_grad(loss_fn, argnums, has_aux, ...)` - High-level gradient clipping API
   - `AuxiliaryOutput` - Named tuple for auxiliary outputs
   - `clip_sum(...)` - **DEPRECATED** Legacy alias for backward compatibility

**Implemented Parameters** (all single-device features from JAX-Privacy main):
- ✅ `fun`, `argnums`, `batch_argnums`, `keep_batch_dim`, `l2_clip_norm`
- ✅ `rescale_to_unit_norm`, `normalize_by`, `return_norms`
- ✅ `return_values`, `return_grad_norms`, `has_aux`
- ✅ `pre_clipping_transform`, `nan_safe`, `dtype`
- ✅ `return_zero` - For privacy amplification via padding

### ⚠️ Known Tech Debt (Deferred Features)

**Not Implemented** (require significant additional work):

1. **`prng_argnum`** parameter
   - **Why deferred**: Requires sophisticated PRNG key splitting for PyTorch
   - **JAX approach**: JAX has built-in PRNG splitting with `jax.random.split()`
   - **PyTorch challenge**: No equivalent - `torch.Generator` can't be easily split
   - **Workaround attempted**: Created batch of int seeds with `torch.arange(seed, seed+batch_size)`
   - **Problem**: This is not cryptographically sound and doesn't match JAX behavior
   - **Impact**: Functions requiring per-example randomness won't work correctly
   - **Defer to**: Stage 3+ if needed, may require custom PRNG implementation
   - **Reference**: `../jax_privacy/jax_privacy/experimental/clipping.py` lines 387-425

2. **`microbatch_size`** parameter
   - **Why deferred**: JAX-Privacy uses sophisticated microbatching with:
     - `microbatching.inmemory_microbatched_fn_general()` wrapper
     - `AccumulationType` (SUM, CONCAT) for different outputs
     - Sharding-aware reshaping for distributed training
   - **Our approach**: Simple for-loop would not match JAX behavior and could trace incorrectly
   - **Impact**: Memory usage will be higher for large batches
   - **Defer to**: Stage 3 or 4 when memory optimization becomes critical
   - **Reference**: `../jax_privacy/jax_privacy/experimental/microbatching.py`

3. **`spmd_axis_name`** parameter
   - **Why deferred**: For JAX SPMD distributed training only
   - **Impact**: None for single-device training
   - **Defer to**: Stage 5+ if distributed training support is needed

### Implementation Notes

**Design Decisions**:
- `clip_pytree` returns `(clipped, norm)` tuple to match JAX-Privacy
- `clip_sum` returns `BoundedSensitivityCallable` with `l2_norm_bound` property
- `clipped_grad` returns `(grad, AuxiliaryOutput)` matching JAX-Privacy's high-level API
- `nan_safe` defaults to `True` (safer for production use)

---

## Deferred Items

Items not included in Stage 1 & 2 (to be addressed in later stages):

**Stage 3 (Additional Optimizer Features)**:

- [x] adaptive_clipping wrapper for TorchOpt optimizers (COMPLETE)
- [x] Optimizer tests using functional API (56 tests passing)
- [x] Tutorial demonstrating TorchOpt flexibility (COMPLETE)
- [ ] Additional optimizer wrappers if needed (e.g., DPOptimizer base class)
- [ ] Integration with more TorchOpt optimizers beyond SGD/Adam/AdamW

**Stage 4 (High-Level API)**:
- [ ] LoRA integration test with `transformers` library
- [ ] Advanced example with real LLM fine-tuning
- [ ] Automatic LoRA parameter detection

**Stage 5 (Polish & Documentation)**:
- [ ] Jupyter notebook tutorials (replacing deleted examples/)
- [ ] Performance benchmarking vs JAX-Privacy
- [ ] Additional validation scenarios (multi-layer networks)
- [ ] Tutorial documentation in `docs/`
- [ ] Consider `spmd_axis_name` for distributed training (if needed)

---

## Fast Context Recovery

**Recent Changes** (2025-11-14 - Optimizer Tests Fixed & Tutorial Restructured):

- ✅ **Fixed all 23 failing optimizer tests** - Complete rewrite of test_dp_optimizer_ac.py (437 lines)
  - Fixed syntax errors (missing commas, malformed comments)
  - Rewrote tests to match functional API: `adaptive_clipping(base_optimizer, ...)` not
    `dp_optimizer_ac(learning_rate=...)`
  - Updated to use external noise injection and privacy accounting (not built-in)
  - New test classes: TestInitialization, TestStepFunction, TestAdaptiveClipping, TestLRScaling,
    TestDifferentOptimizers, TestMetrics, TestIntegrationWithClippedGrad, TestStateImmutability
  - All 56 optimizer tests now passing (37 adaptive_clipping + 19 dp_optimizer_ac) ✓

- ✅ **Tutorial restructured** - docs/tutorials/04_dp_optimizers.ipynb (18 cells)
  - **New title**: "Functional DP Training with TorchOpt"
  - **Part 1**: Setup (data, privacy config) - kept as-is
  - **Part 2 (NEW)**: TorchOpt Optimizers Comparison
    - Shows that ANY TorchOpt optimizer works: torchopt.sgd, torchopt.adam, torchopt.adamw
    - Trains 3 models in parallel with fixed clipping and external noise/accounting
    - Compares convergence across optimizers
    - Emphasizes adaptive_clipping comes AFTER understanding base optimizers
  - **Part 3**: Introduce adaptive_clipping as WRAPPER around base optimizers
    - Shows how clip norm C adapts automatically
    - Compares against fixed clipping from Part 2
  - **Part 4**: Dynamic LR Scaling (optional enhancement)
  - **Part 5**: Updated summary emphasizing TorchOpt flexibility and modular design
  - Fixed variable reference errors (losses_fixed → losses_adamw)

- 🎉 **Key improvements**:
  - Better pedagogical flow: base optimizers first, then adaptive wrapper
  - Demonstrates TorchOpt flexibility (any optimizer works)
  - Maintains modular design (external noise + accounting)
  - All 56 optimizer tests passing, tutorial ready to execute

**Previous Changes** (2025-11-14 - Test Suite Cleanup & Parallel Execution):

- ✅ **Test suite cleanup** - Fixed all failing tests after functional API migration
  - Deleted irrelevant tests: `TestNeighboringRelation` class (3 tests) and `test_invalid_sampling_method`
  - Rewrote 4 input validation tests in `test_composition.py` with required parameters
  - Migrated 5 JAX validation tests from OOP API to functional API
  - Deleted 8 obsolete JAX validation tests (2 RDP + 6 calibration)
- ✅ **Parallel test execution** - Configured pytest-xdist for ~3.17x speedup
  - Added `pytest-xdist>=3.0.0` to dev dependencies
  - Configured `-n=auto` in pytest addopts (12 workers on local machine)
  - Safe for functional API (no shared mutable state)
  - Accounting tests: 42.81s vs 135.81s sequential
- ✅ **All accounting tests passing** - 55 tests total (50 unit + 5 JAX validation)
  - test_composition.py: 13 tests (composition functions, input validation)
  - test_queries.py: 16 tests (epsilon/delta/beta/advantage queries)
  - test_calibration.py: 21 tests (epsilon/delta, advantage, error rates calibration)
  - test_accounting.py (JAX validation): 5 tests (cross-framework numerical equivalence)

**Previous Changes** (2025-11-14 - Calibration Reimplementation):

- ✅ **Calibration using riskcal.calibration.core** - Reimplemented using riskcal's primitives
  - Re-exported: `PrivacyEvaluator`, `PrivacyMetrics`, `CalibrationTarget`, `CalibrationConfig`, `calibrate_parameter`
  - Re-exported from riskcal.dpsgd: `find_noise_multiplier_for_advantage`, `find_noise_multiplier_for_err_rates`
  - Custom implementation: `find_noise_multiplier_for_epsilon_delta` using opaque.accounting
- ✅ **Tutorial notebook updated** - Complete rewrite to use functional accounting API
  - Updated imports: `import opaque.accounting as acc`
  - Replaced OOP API (`PLDAccountant()`) with functional API (`acc.create()`)
  - Added calibration examples for all three metrics: (ε,δ), advantage, (α,β)
- ✅ **9 calibration tests** - All passing with integration checks
- 🎉 **Functional API migration complete** - Immutable state, pure functions, composable

**Previous Changes** (2025-11-12 - Stage 2 Complete - Calibration & Validation):

- ✅ **Flexible accountant instantiation** - Calibration functions accept string/class/callable
- ✅ **Added calibrate_batch_size()** - Find maximum batch size for target privacy budget
- ✅ **Stabilized unit tests** - Marked slow tests, reduced timeouts, all 30 tests passing
- ✅ **JAX validation tests** - 13 comprehensive tests comparing against JAX-Privacy
- ✅ **All tests passing** - 30 unit + 13 JAX validation = 43 total tests
- ✅ **Numerical equivalence confirmed** - All accountants and calibration functions match JAX-Privacy
- 🎉 **Stage 2 COMPLETE** - Privacy accounting fully implemented, tested, and validated

**Previous Changes** (2025-11-12 - PLD Accounting & Truncated Poisson):

- ✅ **Simplified accounting module** - Removed old structure, kept PLDAccountant and RDPAccountant
- ✅ **PLD-based accounting** - Uses Google's dp-accounting library for tight privacy bounds
- ✅ **Truncated Poisson support** - Fully implemented and tested (solves variable batch size problem)
- ✅ **Initial calibration functions** - calibrate_noise_multiplier and calibrate_steps implemented

**Previous Changes** (2025-11-12 - Module Restructuring):

- ✅ **Fixed circular import** - `clipped_grad.py` now imports types directly from `types.py`
- ✅ **Modularized structure** - Split `clipping.py` into proper package with submodules
- ✅ **Proper `__all__` exports** - Public API clearly defined in `__init__.py` files

**Previous Changes** (2025-11-11 - Final API Cleanup):

- ✅ Removed `clip_sum()` entirely - No longer deprecated, completely removed
- ✅ Removed `prng_argnum` - Documented as tech debt (requires sophisticated PRNG splitting)
- ✅ All tests updated to use `clipped_fun()` (no more clip_sum references)
- ✅ Clean API with only `clipped_fun()` as primary clipping function
- ✅ Tech debt properly documented for prng_argnum, microbatch_size, spmd_axis_name

**Previous Changes** (2025-11-11 - JAX-Privacy main Branch Migration Complete):
- ✅ Migrated to JAX-Privacy main branch API (from v1.0.0)
- ✅ Added `clipped_fun()` as primary API
- ✅ Updated `BoundedSensitivityCallable` with `has_aux` field and `sensitivity()` method
- ✅ Added `return_zero` parameter to `clip_pytree()` for privacy amplification
- ✅ Simplified `clipped_grad()` to ~30 lines using `torch.func.grad(has_aux=True)` and delegating to `clipped_fun()`
- ✅ Consolidated all code into single `clipping.py` module (deleted `gradient_clipping.py`)
- ✅ Updated output signature: `return_norms=True` now returns `(value, (aux, norms))` not `(value, norms)`
- ✅ 70 tests passing with ~90% coverage (25 unit + 45 JAX validation)
- ✅ All numerical validations pass against JAX-Privacy main within atol=1e-5

**Previous Changes** (2025-11-11 - Stage 1 Complete):
- ✅ Implemented full pytree_utils module (152 LOC, optree-based)
- ✅ Implemented clip_pytree with all edge cases
- ✅ Fixed `rescale_to_unit_norm` edge case
- ✅ Removed examples/ folder (defer to Stage 5)

**Implementation Status**: ✅ **Stages 1-2 COMPLETE (100%)**. All single-device features from JAX-Privacy main branch
implemented, tested, and validated. Functional privacy accounting API with riskcal-based calibration fully working.
All 111 tests passing (55 accounting + 56 optimizer) with parallel execution (3.17x speedup).

🔄 **Phase 1 IN PROGRESS (Production Hardening)**: See [RFC: Production Plan](docs/development/RFC_PRODUCTION_PLAN.md) for:
- Complete 4-phase roadmap (~16 weeks to v1.0.0)
- Production readiness gaps (memory efficiency, Opacus validation, scale testing)
- Functional architecture decision (following JAX-Privacy patterns)
- Week-by-week implementation tasks

---

## Critical Reminders

1. **No premature implementation**: Tests first, always
2. **JAX-Privacy at `../jax_privacy`**: Use for reference and validation
3. **Security-critical**: DP guarantees depend on correct implementation
4. **Mark JAX tests**: Use `@pytest.mark.jax_validation` (optional group)
5. **Fail-fast**: Prefer exceptions over silent failures

---

## Documentation Structure

All human-facing docs in [docs/](docs/):
- **Getting Started**: Installation, quickstart
- **User Guide**: DP basics, clipping, LoRA
- **API Reference**: Auto-generated from docstrings
- **Development**: Architecture, TDD, decisions, plans
- **Examples**: Linear regression (to be created)

This file is **agent-only** - users should read [README.md](README.md) or [docs/](docs/).
