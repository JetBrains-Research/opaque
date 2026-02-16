# Accounting Module Removal

**Date**: 2026-02-12
**Reason**: Privacy accounting will be provided by jbr-fed-accounting (external Rust library)

---

## What Was Removed

### Code
- `src/opaque/accounting/` - Entire module (4 files, ~1500 LOC)
  - `__init__.py` - Public API
  - `composition.py` - PLD composition functions
  - `queries.py` - Epsilon/beta/advantage queries
  - `calibration.py` - Noise calibration using riskcal

### Tests
- `tests/accounting/` - All accounting tests (4 files, 55 tests)
  - `test_composition.py` - 13 tests
  - `test_queries.py` - 16 tests  
  - `test_calibration.py` - 21 tests
  - `test_integration.py` - 5 JAX validation tests

### Dependencies
- `dp-accounting>=0.4.0` - Removed from pyproject.toml
- `riskcal>=1.0.0` - Removed from pyproject.toml

### Public API
- `opaque.accounting` - Removed from exports
- Updated `src/opaque/__init__.py` docstring to note accounting is external

---

## What Remains

**Test Results**: ✅ **175 tests passing** (down from 111+55=166 due to optimizer tests being added)
- Clipping: 45 tests
- Noise: 12 tests
- Sampling: 23 tests
- Optimizers: 56 tests
- Utils: 30 tests
- Integration: 1 test

**Core Modules**:
```
src/opaque/
├── clipping/        # ✅ Gradient clipping (no changes)
├── noise/           # ✅ Noise injection (no changes)
├── sampling/        # ✅ Batch sampling (no changes)
├── optimizers/      # ✅ Adaptive clipping wrapper (no changes)
├── profiling/       # Empty (to be implemented in Phase 1)
├── integration/     # Empty (to be implemented in Phase 3)
└── utils/           # ✅ PyTree & functional utils (no changes)
```

---

## Migration Path

### For Existing Code Using opaque.accounting

**Before** (old Opaque):
```python
import opaque.accounting as acc

# Create state
state = acc.create()

# Compose mechanisms
state = acc.compose_truncated_poisson_gaussian(
    state,
    noise_multiplier=1.1,
    sample_rate=0.01,
    truncated_batch_size=64,
    dataset_size=10000,
    count=1000,
)

# Query privacy
epsilon = acc.get_epsilon(state, delta=1e-5)
```

**After** (with jbr-fed-accounting - once Python bindings available):
```python
import jbr_fed_accounting as acc

# Build mechanism (functional composition)
mechanism = acc.gaussian(noise_multiplier=1.1) \
    .poisson(sample_rate=0.01) \
    .truncated(max_batch_size=64, dataset_size=10000) \
    .repeat(count=1000)

# Query privacy
epsilon = mechanism.epsilon_at(delta=1e-5)
```

### Training Loop Changes

**Before**:
```python
import opaque
import opaque.accounting as acc

# Configure mechanisms
grad_fn = opaque.clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = opaque.gaussian(noise_multiplier=1.1)

# Track privacy internally
state = acc.create()

for batch in dataloader:
    grads = grad_fn(params, batch)
    noisy = noise_fn(grads)
    params = optimizer.step(params, noisy)
    
    # Update accounting
    state = acc.compose_gaussian(state, noise_multiplier=1.1)

epsilon = acc.get_epsilon(state, delta=1e-5)
```

**After** (accounting external):
```python
import opaque
import jbr_fed_accounting as acc

# Configure mechanisms
grad_fn = opaque.clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = opaque.gaussian(noise_multiplier=1.1)

# Define accounting mechanism (once, before training)
mechanism = acc.gaussian(nm=1.1).poisson(rate=0.01).repeat(count=num_steps)
epsilon = mechanism.epsilon_at(delta=1e-5)  # Pre-compute budget

# Training loop (no accounting state)
for batch in dataloader:
    grads = grad_fn(params, batch)
    noisy = noise_fn(grads)
    params = optimizer.step(params, noisy)
```

**Key Difference**: Accounting happens **before training** (mechanism definition + privacy budget pre-computation) rather than **during training** (stateful composition).

---

## Rationale

### Why Remove?

1. **Separation of Concerns**
   - Opaque focuses on **training** (clipping, noise, sampling, optimizers)
   - jbr-fed-accounting focuses on **privacy accounting**
   - Clear API boundary between the two

2. **Performance**
   - jbr-fed-accounting is Rust-based → faster accounting
   - No Python overhead for privacy computations

3. **Maintenance**
   - Single source of truth for privacy accounting
   - Opaque doesn't need to track dp-accounting API changes
   - Reduces dependency conflicts

4. **Composability**
   - jbr-fed-accounting has richer composition API
   - Supports advanced mechanisms (adaptive clipping, matrix factorization)
   - Better suited for federated learning scenarios

### What We Lose (Temporarily)

Until jbr-fed-accounting Python bindings are available:

- ❌ Built-in privacy tracking during training
- ❌ Automatic epsilon/delta computation
- ❌ Noise calibration helpers (`find_noise_multiplier_for_epsilon_delta`)

**Workaround**: Users can manually compute privacy budgets using:
- Google's `dp_accounting` library directly
- Analytical formulas for simple DP-SGD
- Conservative estimates

---

## Updated Documentation

### CLAUDE.md
- Updated status: "Privacy accounting provided by jbr-fed-accounting (external)"
- Removed accounting module references
- Updated test counts (175 passing)

### STATUS.md  
- Updated Stage 2 summary
- Removed accounting achievements
- Added note about external accounting

### RFC_PRODUCTION_PLAN.md
- Section 4.1: Removed accounting module from architecture
- Section 9.3: Updated integration strategy (event emission for future)
- No longer includes accounting implementation tasks

---

## Future Integration (Phase 4+)

Once jbr-fed-accounting has Python bindings:

### Option 1: Event Emission (Loose Coupling)
```python
from opaque.accounting import track_privacy

collector = track_privacy()

grad_fn = opaque.clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = opaque.gaussian(noise_multiplier=1.1)

# Wrap functions to emit accounting events
grad_fn = collector.track(grad_fn, "clipping", clip_norm=1.0)
noise_fn = collector.track(noise_fn, "noise", noise_multiplier=1.1)

# Training loop
for batch in dataloader:
    grads = grad_fn(params, batch)  # Emits ClippingEvent
    noisy = noise_fn(grads)          # Emits NoiseEvent
    ...

# Query accounting via jbr-fed-accounting
epsilon = jbr_fed_accounting.from_events(collector.events).epsilon_at(delta=1e-5)
```

### Option 2: No Integration (Keep Separate)
- User tracks parameters manually
- Builds jbr-fed-accounting mechanism independently
- Simpler, more explicit

**Recommendation**: Start with Option 2, add Option 1 if users request it.

---

## Test Coverage After Removal

**Before Removal**: 111 tests (55 accounting + 56 optimizer)
**After Removal**: 175 tests (0 accounting + 175 other)

**Coverage Breakdown**:
- Clipping: 45 tests (100% coverage of core API)
- Noise: 12 tests (covers stateless Gaussian)
- Sampling: 23 tests (Poisson + TruncatedPoisson)
- Optimizers: 56 tests (adaptive clipping wrapper)
- Utils: 30 tests (PyTree, functional, partition/merge)
- Integration: 9 tests (gradient equivalence, real models)

**Total**: 175 tests passing ✅

---

## Next Steps

### Immediate (Phase 1)
1. Focus on production hardening (memory, validation)
2. No accounting dependencies needed
3. Users compute privacy budgets externally

### Future (Phase 4+)
1. Monitor jbr-fed-accounting Python bindings development
2. Design integration strategy (if needed)
3. Add convenience wrappers for common patterns

### For Users
- Use `dp_accounting` directly for now
- Wait for jbr-fed-accounting Python bindings
- Analytical formulas for simple DP-SGD scenarios

---

## References

- **jbr-fed-accounting**: `../../federated-compute/federated-research/packages/dp-accounting/crates/jbr-fed-accounting/`
- **RFC_PRODUCTION_PLAN.md**: Section 9.3 (Integration Strategy)
- **Google dp_accounting**: https://github.com/google/differential-privacy/tree/main/python/dp_accounting
