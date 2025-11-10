# Design Decisions

This document explains key technical decisions made during Opaque's development, including rationale and trade-offs.

---

## 1. PyTree Implementation

**Decision**: Use `torch.utils._pytree` (private API)

### Options Considered

**A. Use `torch.utils._pytree`**
- ✅ Full PyTorch compatibility
- ✅ Handles complex nested structures
- ❌ Private API (underscore prefix) may change
- ❌ Couples us to PyTorch internals

**B. Implement minimal PyTree for dict-only**
- ✅ Stable, controlled API
- ✅ Simpler to maintain
- ❌ Less flexible
- ❌ Need to maintain our own implementation

### Decision: Option A

**Rationale**:
- LoRA adapters typically use simple dict structures
- Compatibility with PyTorch ecosystem is valuable
- Can migrate if API changes (add abstraction layer)

**Mitigation**:
- Thin wrapper module: `opaque.core.pytree_utils`
- Document dependency on private API in README
- Add tests for dict structures specifically

---

## 2. Microbatching Strategy

**Decision**: Explicit `microbatch_size` parameter (user control)

### Options Considered

**A. Always sequential processing**
- ✅ Predictable memory
- ✅ Simpler implementation
- ❌ Slower than vmap for small batches

**B. Auto-select based on batch size**
- ✅ Best of both worlds (when heuristic works)
- ❌ Complex to tune
- ❌ Unpredictable behavior

**C. Explicit user control via parameter**
- ✅ Users control trade-offs
- ✅ Explicit and predictable
- ❌ Requires understanding microbatching

### Decision: Option C

**Rationale**:
- Users doing DP training understand memory constraints
- Explicit > implicit for performance-critical code
- Can add auto mode later as convenience

**API**:
```python
clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=None,  # None = full batch at once
)
```

---

## 3. Numerical Tolerance for JAX Validation

**Decision**: `atol=1e-5, rtol=1e-5` + property-based tests

### Challenges

- JAX and PyTorch use different PRNG algorithms
- Reduction order may differ (sum, sqrt)
- Gradient computation internals differ
- Floating point associativity issues

### Options Considered

**A. Strict tolerance** (`atol=1e-6`)
- May fail due to legitimate implementation differences

**B. Moderate tolerance** (`atol=1e-5`)
- Catches real bugs while allowing minor variations

**C. Property-based testing only**
- Test invariants (e.g., clipped_norm ≤ clip_norm)
- More robust to implementation differences

### Decision: Option B + Option C

**Testing strategy**:
```python
# Numerical comparison (JAX validation tests)
torch.allclose(torch_result, jax_result, atol=1e-5, rtol=1e-5)

# Property-based tests (unit tests)
assert compute_norm(clipped_grads) <= clip_norm + 1e-6  # numerical slop
```

**Rationale**:
- Moderate tolerance for regression tests
- Property tests for invariants
- Document known divergence sources

---

## 4. Error Handling Philosophy

**Decision**: Fail-fast by default, opt-in for graceful handling

### Edge Cases

1. NaN/Inf gradients (numerical instability)
2. Empty PyTree (no parameters)
3. Zero gradient norm
4. Invalid clip_norm (negative or NaN)

### Options Considered

**A. Fail fast** (raise exceptions)
- ✅ Catches bugs early
- ✅ Forces explicit handling
- ❌ May be annoying for users

**B. Graceful fallbacks** (warnings + defaults)
- ✅ Easier to use
- ❌ May hide bugs

**C. Explicit opt-in for graceful handling**
- ✅ Best of both worlds
- ✅ Security-critical code surfaces issues
- ❌ More complex API

### Decision: Option C

**API design**:
```python
clip_pytree(
    tree,
    clip_norm,
    nan_safe=False,  # If True, replace NaN/Inf with 0
)

clipped_grad(
    loss_fn,
    l2_clip_norm,
    on_empty_grad="error",  # Options: "error", "warn", "ignore"
)
```

**Rationale**:
- DP training is security-critical - surprises are bad
- Fail fast by default
- Allow graceful handling when explicitly requested

---

## 5. Documentation System

**Decision**: Material for MkDocs ✅ **IMPLEMENTED**

### Options Considered

**A. Sphinx**
- ✅ Standard for Python scientific libraries
- ✅ Excellent autodoc support
- ✅ Better for API reference docs
- ❌ Steeper learning curve

**B. Material for MkDocs**
- ✅ Simpler, modern UI
- ✅ Better for narrative docs
- ✅ Great Jupyter integration
- ✅ Beautiful Material Design theme
- ❌ Less standard in PyTorch ecosystem

### Decision: Option B

**Rationale**:
- Modern, clean UI appeals to users
- Simpler for contributors to write docs
- `mkdocstrings` provides excellent autodoc
- Easier to maintain and customize

**Configuration**: See [`mkdocs.yml`](https://github.com/evgri243/opaque/blob/main/mkdocs.yml)

---

## 6. Testing Against JAX-Privacy

**Decision**: Separate directory + pytest markers, optional dependency

### Structure

```
tests/
├── core/                      # Unit tests (always run)
│   ├── test_pytree_utils.py
│   └── test_clipping.py
├── jax_validation/            # JAX comparison tests (optional)
│   └── test_jax_clipping.py
└── conftest.py
```

### Pytest Markers

```python
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_privacy")

@pytest.mark.jax_validation
def test_clipping_matches_jax():
    # Test using both JAX-Privacy and Opaque
    ...
```

### Running Tests

```bash
# Regular tests (no JAX needed)
uv run pytest

# JAX validation tests (requires jax-validation group)
uv run --group jax-validation pytest -m jax_validation
```

**Rationale**:
- Clear separation of validation vs. unit tests
- JAX-Privacy actively developed - test against live API
- Optional dependency doesn't burden regular contributors

---

## 7. Device Handling

**Decision**: Device-agnostic initially, optimize later

### Strategy

**Phase 1** (Current): Device-agnostic
- All ops preserve tensor device
- Works on CPU/CUDA/MPS/XLA automatically

**Phase 2** (Future): Device-specific optimizations
- Custom CUDA kernels for clipping
- Only after profiling shows bottlenecks

**Requirements**:
- All operations preserve tensor device
- Tests run on CPU in CI
- Parametrize tests for CUDA when available

**Rationale**:
- Correctness before optimization
- Profile first, optimize later
- PyTorch's automatic device handling is good enough initially

---

## 8. Loss Function Signatures

**Decision**: Style 1 initially (`loss_fn(params, data)`), extend later

### Supported Signatures

**Style 1** (Phase 1):
```python
loss_fn(params, data) -> scalar
```

**Style 2** (Future):
```python
loss_fn(params, *, batch) -> scalar
```

**Arbitrary** (Future):
- Via `argnums` parameter (mirrors JAX-Privacy exactly)

**Rationale**:
- Style 1 covers 95% of use cases
- Start simple, add complexity when needed
- Matches JAX-Privacy functional API

---

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2025-11-10 | PyTree implementation | Use `torch.utils._pytree` | Compatibility + simplicity for LoRA |
| 2025-11-10 | Loss signatures | Style 1 initially | Covers 95% of use cases |
| 2025-11-10 | Microbatching | Explicit `microbatch_size` | User control > auto-magic |
| 2025-11-10 | Numeric tolerance | `atol=1e-5` + property tests | Balance strictness with robustness |
| 2025-11-10 | JAX validation | Separate directory + markers | Clear separation, optional |
| 2025-11-10 | Device handling | Device-agnostic initially | Correctness first, optimize later |
| 2025-11-10 | Error handling | Fail fast with opt-in graceful | Security-critical, avoid surprises |
| 2025-11-10 | Documentation | Material for MkDocs | Modern UI, simpler for contributors |
| 2025-11-10 | JAX dependency | Optional `jax-validation` group | Test against live API, optional |

---

## Revisit Triggers

These decisions should be revisited when:

1. **PyTree implementation** - If `torch.utils._pytree` API changes in PyTorch release
2. **Microbatching** - After profiling shows performance issues
3. **Numeric tolerance** - If systematic divergence from JAX discovered
4. **JAX validation** - If JAX-Privacy API changes significantly
5. **Device handling** - After profiling on GPU shows bottlenecks
6. **Error handling** - Based on user feedback about edge cases

---

**Last Updated**: 2025-11-10
