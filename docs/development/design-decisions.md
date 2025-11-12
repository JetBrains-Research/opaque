# Design Decisions

This document records key design decisions made during the development of Opaque, including rationale, alternatives considered, and trade-offs.

---

## Decision #1: PyTree Library Choice

**Date**: 2025-11-11
**Status**: ✅ Decided

### Context

JAX-Privacy uses JAX's built-in `jax.tree_util` for PyTree operations. PyTorch has `torch.utils._pytree`, but it's a private API.

### Decision

Use `optree` library as the primary PyTree implementation, with `torch.utils._pytree` as fallback.

### Rationale

- `optree` is a stable, maintained library specifically for PyTree operations
- More feature-complete than torch's private API
- Better error messages and documentation
- JAX-compatible API surface

### Alternatives Considered

1. **torch.utils._pytree only**: Private API, may break in future PyTorch versions
2. **Custom implementation**: Too much engineering effort, reinventing the wheel
3. **JAX's tree_util**: Would require JAX dependency even for non-validation usage

### Trade-offs

- **Pro**: Stable API, good performance, JAX-compatible
- **Con**: Additional dependency (but lightweight)

### References

- `src/opaque/pytree_utils.py:13-28` - Implementation with optree

---

## Decision #2: Numerical Tolerance for JAX Validation

**Date**: 2025-11-11
**Status**: ✅ Decided

### Context

Need to validate PyTorch implementation against JAX-Privacy reference, but numerical differences are expected due to:
- Different random number generators
- Different dtype promotion rules
- Floating-point accumulation order

### Decision

Use `atol=1e-5` (absolute tolerance) for all JAX validation tests.

### Rationale

- Empirically sufficient for catching correctness bugs
- Tight enough to detect implementation errors
- Loose enough to handle legitimate numerical differences
- Standard tolerance in scientific computing

### Alternatives Considered

1. **Stricter tolerance (1e-7)**: Would fail on legitimate numerical noise
2. **Looser tolerance (1e-3)**: Might miss subtle bugs
3. **Relative tolerance only**: Fails for values near zero

### Trade-offs

- **Pro**: Catches bugs while allowing numerical noise
- **Con**: Might theoretically miss very subtle numerical bugs (unlikely in practice)

### References

- All tests in `tests/jax_validation/` use `atol=1e-5`

---

## Decision #3: Gradient API Design

**Date**: 2025-11-11
**Status**: ✅ Decided

### Context

JAX-Privacy provides two APIs:
1. **Low-level**: `clip_pytree()` for direct PyTree clipping
2. **High-level**: `clipped_grad()` for gradient clipping

### Decision

Implement both APIs with exact parameter parity to JAX-Privacy (excluding distributed parameters).

### Rationale

- Low-level API (`clip_pytree`, `clip_sum`) provides building blocks for advanced users
- High-level API (`clipped_grad`) provides convenience for common use cases
- Maintaining parity ensures easy migration from JAX-Privacy

### Alternatives Considered

1. **High-level only**: Would limit flexibility for advanced users
2. **Different API design**: Would make migration harder, lose JAX-Privacy's proven design

### Trade-offs

- **Pro**: Maximum compatibility, flexibility
- **Con**: More API surface to maintain

### References

- `src/opaque/clipping.py` - Both APIs implemented
- `docs/development/jax-privacy-comparison.md` - Full API mapping

---

## Decision #4: Error Handling Strategy

**Date**: 2025-11-11
**Status**: ✅ Decided

### Context

Differential privacy is security-critical - incorrect implementation can leak private data.

### Decision

Fail-fast by default:
- Validate inputs aggressively (dimension checks, dtype checks)
- Raise exceptions on invalid configurations
- No silent fallbacks or warnings for critical parameters

### Rationale

- Better to crash than to silently compute wrong DP guarantees
- Easier to debug explicit errors than silent failures
- Aligns with security-critical nature of DP

### Alternatives Considered

1. **Permissive with warnings**: Dangerous for DP applications
2. **Runtime type checking with typegrad**: Too heavy-weight, adds complexity

### Trade-offs

- **Pro**: Catches bugs early, prevents silent DP violations
- **Con**: May be less convenient for rapid prototyping

### References

- `src/opaque/clipping.py:489-509` - Argument validation in `clipped_grad()`

---

## Decision #5: Test Organization

**Date**: 2025-11-11
**Status**: ✅ Decided

### Context

Need to test both correctness and equivalence with JAX-Privacy.

### Decision

Two-tier test structure:
1. **Unit tests** (`tests/core/`): Test Opaque in isolation
2. **JAX validation** (`tests/jax_validation/`): Compare against JAX-Privacy

JAX validation tests marked with `@pytest.mark.jax_validation` and in separate optional dependency group.

### Rationale

- Unit tests run fast, don't require JAX dependency
- JAX validation provides numerical correctness guarantee
- Optional JAX dependency keeps installation lightweight

### Alternatives Considered

1. **JAX validation only**: Would require JAX for all testing
2. **Unit tests only**: Wouldn't guarantee numerical equivalence
3. **Inline JAX tests**: Would mix concerns, harder to skip

### Trade-offs

- **Pro**: Fast unit tests, optional comprehensive validation
- **Con**: More test files to maintain

### References

- `pyproject.toml:53-55` - Optional jax-validation dependency group
- `tests/conftest.py` - Test markers configuration

---

## Decision #6: Documentation Strategy

**Date**: 2025-11-11
**Status**: ✅ Decided

### Context

Need to document:
- API reference (auto-generated)
- Conceptual guides (hand-written)
- Tutorials (interactive)

### Decision

Use Material for MkDocs with:
- **API docs**: Auto-generated from docstrings with mkdocstrings
- **Tutorials**: Jupyter notebooks with mkdocs-jupyter
- **Guides**: Hand-written markdown

### Rationale

- MkDocs provides excellent static site generation
- Material theme is modern, readable
- Jupyter integration allows interactive learning
- Google-style docstrings are readable in code and rendered docs

### Alternatives Considered

1. **Sphinx**: More complex, overkill for this project
2. **Plain markdown**: No API docs generation, less features
3. **ReadTheDocs default**: Less polished than Material theme

### Trade-offs

- **Pro**: Beautiful docs, easy to maintain
- **Con**: Requires learning MkDocs configuration

### References

- `mkdocs.yml` - Full configuration
- `docs/tutorials/` - Jupyter notebook tutorials

---

## Decision #7: Microbatching Approach

**Date**: 2025-11-11
**Status**: ✅ Decided (Deferred)

### Context

JAX-Privacy implements sophisticated microbatching with:
- `inmemory_microbatched_fn_general()` wrapper
- `AccumulationType` (SUM, CONCAT) for different outputs
- Sharding-aware reshaping for distributed training

Implementing this correctly requires significant engineering effort and careful handling of PyTorch's tracing behavior.

### Decision

**Defer microbatching support to later stages** (Stage 3 or 4).

For Stage 1, users must batch-process if memory is limited, or use smaller batch sizes.

### Rationale

- Microbatching is memory optimization, not core DP algorithm
- Simple for-loop approach would not match JAX behavior and could trace incorrectly
- Stage 1 focus is correctness, not performance
- Memory optimization can be added later without breaking API

### Alternatives Considered

1. **Simple for-loop**: Wouldn't match JAX behavior, could have tracing issues
2. **Partial implementation**: Would be incomplete and confusing
3. **Implement now**: Would significantly delay Stage 1 completion

### Trade-offs

- **Pro**: Faster Stage 1 completion, simpler codebase
- **Con**: Higher memory usage for large batches
- **Mitigation**: Documented as tech debt, planned for later stage

### References

- `CLAUDE.md:48-55` - Tech debt documentation
- `../jax_privacy/jax_privacy/experimental/microbatching.py` - JAX reference implementation

---

## Decision #8: LoRA Integration Approach

**Date**: 2025-11-11
**Status**: 📋 Planned (Stage 4)

### Context

Opaque is designed for DP fine-tuning of LLMs with LoRA. Need to decide how to integrate with existing LoRA libraries.

### Decision

**Deferred to Stage 4**: Will integrate with `peft` (HuggingFace's Parameter-Efficient Fine-Tuning library).

### Rationale

- `peft` is the most widely used LoRA library for PyTorch
- Provides standardized LoRA parameter management
- Integrates well with `transformers` library
- Stage 1-3 focus on DP primitives, Stage 4 on high-level API

### Alternatives Considered

1. **Custom LoRA implementation**: Reinventing the wheel, poor ecosystem integration
2. **Multiple library support**: Too much maintenance burden
3. **No specific LoRA integration**: Would miss a key use case

### Trade-offs

- **Pro**: Leverage existing ecosystem, standard API
- **Con**: Dependency on external library evolution

### Status

📋 To be implemented in Stage 4.

---

## Decision #9: PyTorch/JAX API Differences

**Date**: 2025-11-11
**Status**: ✅ Decided

### Context

PyTorch's `torch.func` API is inspired by JAX but has critical differences:

1. **value_and_grad behavior**:
   - JAX: `jax.value_and_grad(fun, has_aux=True)(*args)` returns `((value, aux), grad)`
   - PyTorch: `torch.func.grad(fun, has_aux=True)(*args)` returns `(grad, aux)` - NO VALUE!

2. **vmap None handling**:
   - JAX: `vmap` can handle None values in outputs transparently
   - PyTorch: `vmap` with `out_dims != None` fails on None values: `ValueError: must only return Tensors`

### Decision

**Implement compatibility wrappers**:

1. **`_value_and_grad()` helper**: Calls both original function (for value) and `torch.func.grad` (for gradient), returns JAX-compatible format `((value, aux), grad)`

2. **Dict-based aux handling**: Inside `vmap`'d functions, return dicts instead of namedtuples with None values. Convert back to `AuxiliaryOutput` namedtuple after `vmap` completes.

### Rationale

- Maintains API compatibility with JAX-Privacy
- Avoids breaking PyTorch vmap limitations
- Trade-off: Performance cost of calling function twice is acceptable for API compatibility

### Alternatives Considered

1. **Different return format**: Would break JAX-Privacy API parity
2. **Conditional logic inside vmap**: Too complex, error-prone
3. **Always return all aux values**: Would change semantics unnecessarily

### Trade-offs

- **Pro**: Full JAX-Privacy API compatibility, clean user-facing API
- **Con**: Performance overhead (calling function twice), internal complexity
- **Validation**: All workarounds tested against JAX-Privacy, numerical equivalence within atol=1e-5

### Implementation Details

**_value_and_grad helper** (35 lines):
```python
def _value_and_grad(fun: Callable, argnums: int | tuple[int, ...] = 0, has_aux: bool = False):
    """Create a function that returns both value and gradient, mimicking jax.value_and_grad."""
    grad_fn = _torch_grad(fun, argnums=argnums, has_aux=has_aux)

    if has_aux:
        def wrapper(*args, **kwargs):
            value, aux = fun(*args, **kwargs)  # Call original
            gradient, _ = grad_fn(*args, **kwargs)  # Call grad_fn
            return (value, aux), gradient  # JAX format
    else:
        def wrapper(*args, **kwargs):
            value = fun(*args, **kwargs)
            gradient = grad_fn(*args, **kwargs)
            return value, gradient

    return wrapper
```

**Dict-based aux handling**:
```python
# Inside grad_fn (called by vmap):
aux_dict = {}  # Use dict instead of namedtuple
if return_grad_norms:
    aux_dict["grad_norms"] = norm
# ... other optional values

return result, aux_dict  # Return dict, not AuxiliaryOutput

# After clipped_fun returns (outside vmap):
grad, aux_dict = clipped_grad_fn(*args, **kwargs)
aux_output = AuxiliaryOutput(
    values=aux_dict.get("values"),  # None if not present
    grad_norms=aux_dict.get("grad_norms"),
    aux=aux_dict.get("aux"),
)
return grad, aux_output  # Convert back to namedtuple
```

### Testing

- Created `test_vmap_none.py` to verify PyTorch vmap limitation
- All 79 tests pass (34 unit + 45 JAX validation)
- Numerical equivalence with JAX-Privacy within atol=1e-5

### References

- `src/opaque/clipping.py:29-64` - `_value_and_grad()` implementation
- `src/opaque/clipping.py:569-619` - Dict-based aux handling
- `docs/development/jax-privacy-comparison.md` - Detailed API comparison
- `tests/core/test_gradient_clipping.py` - Unit tests
- `tests/jax_validation/test_clipping.py` - Validation tests

---

## Future Decisions

Topics to decide in future stages:

### Stage 2 (Noise Injection)
- [ ] PRNG seeding strategy (reproducibility vs security)
- [ ] Noise distribution library (native PyTorch vs custom)

### Stage 3 (Privacy Accounting)
- [ ] Privacy accounting library choice (Opacus vs custom vs PRV accountant)
- [ ] Privacy budget tracking API design

### Stage 5 (Polish)
- [ ] Performance optimization priorities
- [ ] Distributed training support (if needed)
