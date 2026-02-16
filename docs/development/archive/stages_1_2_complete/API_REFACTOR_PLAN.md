# API Refactor Plan: Simplified Functional Design

**Date**: 2026-02-12
**Status**: Planning
**Goal**: Simplify API by removing `BoundedSensitivityCallable` wrapper, use plain functions with attributes

---

## Motivation

Current design inherited JAX-Privacy's `BoundedSensitivityCallable` wrapper for sensitivity tracking. However:

1. **Sensitivity is a privacy accounting concern**, not a training concern
2. **Neighboring relations** (ADD_OR_REMOVE, REPLACE_ONE) belong in accounting layer
3. **Wrapper adds complexity** without clear benefit for our use cases
4. **Research flexibility** requires composable components, not complex wrappers

**Goal**: Keep full composability (for different clipping/noise mechanisms) while maximizing simplicity.

---

## Design Principles

1. **Plain functions** - No custom wrapper classes
2. **Simple attributes** - Use function attributes for metadata (`.clip_norm`)
3. **Explicit parameters** - User does simple math (`stddev = noise_mult * clip_norm`)
4. **Full composability** - Easy to swap clipping/noise mechanisms

---

## Proposed API

### Clipping

```python
# Returns plain function with .clip_norm attribute
grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_argnums=1,
)

# Access clip norm as simple attribute
print(grad_fn.clip_norm)  # 1.0

# Use it
grads = grad_fn(params, batch_data)
```

**Implementation**: Use function attributes (Python allows `func.attr = value`)

### Noise

```python
# Option 1: User does multiplication explicitly
noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)

# Option 2: Convenience helper (sugar)
noise_fn = gaussian(noise_multiplier=1.1, clip_norm=grad_fn.clip_norm)

# Use it
noisy_grads = noise_fn(grads)
```

### Composition

```python
# Configure once
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)

# Training loop
for batch in dataloader:
    grads = grad_fn(params, batch)
    noisy = noise_fn(grads)
    params = optimizer.step(params, noisy)
```

### Future Research Flexibility

```python
# Swap clipping mechanism
grad_fn = per_layer_clipped_grad(loss_fn, clip_norms={'layer1': 1.0, 'layer2': 0.5})

# Swap noise mechanism
noise_fn = correlated_gaussian(stddev=1.1, rank=10)  # Matrix factorization
noise_fn = clipped_gaussian(stddev=1.1, clip_at=3.0)  # Truncated Gaussian
noise_fn = laplace(scale=1.1)  # Pure DP

# Composition still works
for batch in dataloader:
    grads = grad_fn(params, batch)
    noisy = noise_fn(grads)
    params = optimizer.step(params, noisy)
```

---

## Migration Plan

### Phase 1: Update Core Functions

1. **Remove `BoundedSensitivityCallable`** from `types.py`
2. **Update `clipped_fun()`**:
   - Return plain function
   - Add `.clip_norm` attribute
   - Remove `.sensitivity()` method
   - Remove `.l2_norm_bound` and `.has_aux` attributes
3. **Update `clipped_grad()`**:
   - Same changes as `clipped_fun()`
4. **Update `gaussian.py`**:
   - Implement `gaussian(stddev)` - primary API
   - Implement `gaussian_stateful(stddev, seed)` - for reproducibility
   - Deprecate `add_gaussian_noise()`

### Phase 2: Update Tests

1. Remove all `.sensitivity()` calls
2. Use `.clip_norm` attribute instead
3. Update noise tests for new API
4. JAX validation tests - adapt or remove if not applicable

### Phase 3: Update Examples & Docs

1. Update tutorial notebook
2. Update integration tests
3. Create migration guide (old → new API)

---

## Detailed Changes

### `clipping/types.py`

**Before**:
```python
@dataclass(frozen=True)
class BoundedSensitivityCallable:
    fun: Callable[..., Any]
    l2_norm_bound: float
    has_aux: bool

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def sensitivity(self, neighboring_relation: str) -> float:
        # ...
```

**After**:
```python
# File becomes simpler or removed entirely
AuxiliaryOutput = namedtuple("AuxiliaryOutput", ["values", "grad_norms", "aux"])

__all__ = ["AuxiliaryOutput"]
```

### `clipping/clipped_fun.py`

**Before** (returns `BoundedSensitivityCallable`):
```python
def clipped_fun(...) -> BoundedSensitivityCallable:
    # ... implementation ...
    return BoundedSensitivityCallable(clipped_fn, norm_bound, output_has_aux)
```

**After** (returns plain function with attribute):
```python
def clipped_fun(...) -> Callable:
    # ... implementation ...

    # Add clip_norm as function attribute
    clipped_fn.clip_norm = (1.0 if rescale_to_unit_norm else l2_clip_norm) / normalize_by

    return clipped_fn
```

### `clipping/clipped_grad.py`

**Before**:
```python
def clipped_grad(...) -> BoundedSensitivityCallable:
    clipped_grad_fn = clipped_fun(...)
    # ... wrap to convert aux_dict ...
    return BoundedSensitivityCallable(wrapper, clipped_grad_fn.l2_norm_bound, ...)
```

**After**:
```python
def clipped_grad(...) -> Callable:
    clipped_grad_fn = clipped_fun(...)
    # ... wrap to convert aux_dict ...
    wrapper.clip_norm = clipped_grad_fn.clip_norm
    return wrapper
```

### `noise/gaussian.py`

**New API**:
```python
def gaussian(stddev: float) -> Callable:
    """Create stateless Gaussian noise function.

    Args:
        stddev: Standard deviation of noise (usually noise_multiplier * clip_norm)

    Returns:
        Function that adds N(0, stddev²) noise to gradients
    """
    if stddev == 0:
        return lambda grads: grads

    def noise_fn(grads):
        def add_noise(tensor):
            return tensor + torch.randn_like(tensor) * stddev
        return tree_map(add_noise, grads)

    return noise_fn


def gaussian_stateful(stddev: float, seed: int) -> tuple[Callable, torch.Generator]:
    """Create Gaussian noise function with explicit state.

    Returns:
        (noise_fn, generator) where noise_fn(grads, gen) adds reproducible noise
    """
    generator = torch.Generator().manual_seed(seed)

    def noise_fn(grads, gen):
        def add_noise(tensor):
            return tensor + torch.randn(tensor.shape, generator=gen) * stddev
        return tree_map(add_noise, grads)

    return noise_fn, generator
```

---

## Breaking Changes

1. **`grad_fn.sensitivity()` removed** → Use `grad_fn.clip_norm`
2. **`BoundedSensitivityCallable` removed** → Plain functions
3. **`add_gaussian_noise()` deprecated** → Use `gaussian()` or `gaussian_stateful()`
4. **Noise API changed**:
   - Old: `add_gaussian_noise(grads, stddev=...)`
   - New: `noise_fn = gaussian(stddev=...); noise_fn(grads)`

---

## Benefits

1. ✅ **Simpler mental model** - Plain functions, no wrappers
2. ✅ **Same composability** - Easy to swap mechanisms
3. ✅ **Less code** - Remove `BoundedSensitivityCallable`, simpler logic
4. ✅ **Clearer separation** - Privacy accounting stays in accounting layer
5. ✅ **Research flexibility** - Easy to experiment with new mechanisms

---

## Next Steps

1. Get approval on design
2. Implement Phase 1 (core functions)
3. Update tests (Phase 2)
4. Update docs/examples (Phase 3)
