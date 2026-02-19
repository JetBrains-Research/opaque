# Migration Guide: Old API → New API

**Last Updated**: 2026-02-12
**Opaque Version**: 0.2.0 → 1.0.0

---

## Overview

Opaque 0.2.0 introduces a simplified functional API that removes wrapper classes in favor of plain functions with attributes. This guide helps you migrate from the old PoC API to the new production-ready API.

**TL;DR**: The new API is simpler, more explicit, and fully backward compatible (with deprecation warnings).

---

## Quick Migration Checklist

- [ ] Replace `.sensitivity()` calls with `.clip_norm` attribute
- [ ] Replace `add_gaussian_noise()` with `gaussian()` or `gaussian_noise()`
- [ ] Update noise calibration to use explicit multiplication
- [ ] Remove `BoundedSensitivityCallable` imports (if any)
- [ ] Test your code - all old API calls will emit deprecation warnings

---

## Core Changes

### 1. Clipping API - No More `.sensitivity()` Method

**Old API** (with `BoundedSensitivityCallable` wrapper):

```python
from opaque import clipped_grad

grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)

# Old: Access sensitivity via method call
sensitivity = grad_fn.sensitivity("REPLACE_SPECIAL")  # Returns 1.0
stddev = noise_multiplier * sensitivity
```

**New API** (plain function with attribute):

```python
from opaque import clipped_grad

grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)

# New: Access clip_norm as simple attribute
clip_norm = grad_fn.clip_norm  # 1.0
stddev = noise_multiplier * clip_norm
```

**Why?**
- **Simpler**: `.clip_norm` is a straightforward attribute, not a method
- **Explicit**: No hidden "neighboring relation" logic
- **Clearer**: Sensitivity is a privacy accounting concern, not a training concern

---

### 2. Noise API - Function Factories

**Old API** (direct function call):

```python
from opaque import add_gaussian_noise

# Old: Direct call every time
noisy_grads = add_gaussian_noise(grads, stddev=1.1)
```

**New API** (higher-order function):

```python
from opaque import gaussian

# New: Configure once, use many times
noise_fn = gaussian(stddev=1.1)

# Use in training loop
for batch in dataloader:
    grads = compute_gradients(params, batch)
    noisy_grads = noise_fn(grads)  # Reusable!
```

**Why?**
- **Composable**: Configure once outside the loop
- **Efficient**: No repeated parameter passing
- **Flexible**: Easy to swap noise mechanisms

---

## Migration Examples

### Example 1: Basic DP-SGD Training

**Old API**:

```python
from opaque import clipped_grad, add_gaussian_noise

grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)

for batch in dataloader:
    # Compute gradients
    grads = grad_fn(params, batch_x, batch_y)

    # Add noise (repeated every iteration)
    sensitivity = grad_fn.sensitivity("REPLACE_SPECIAL")
    stddev = 1.1 * sensitivity
    noisy_grads = add_gaussian_noise(grads, stddev=stddev)

    # Update
    params = optimizer.step(params, noisy_grads)
```

**New API**:

```python
from opaque import clipped_grad, gaussian

# Configure once (outside loop)
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)

for batch in dataloader:
    # Compute gradients
    grads = grad_fn(params, batch_x, batch_y)

    # Add noise (simple function call)
    noisy_grads = noise_fn(grads)

    # Update
    params = optimizer.step(params, noisy_grads)
```

**Benefits**: Configure once, cleaner loop, natural composition

---

### Example 2: Reproducible Noise (Testing/Debugging)

**Old API**:

```python
from opaque import add_gaussian_noise
import torch

generator = torch.Generator().manual_seed(42)

for batch in dataloader:
    grads = compute_gradients(params, batch)
    noisy = add_gaussian_noise(grads, stddev=1.1, generator=generator)
    params = update(params, noisy)
```

**New API**:

```python
from opaque.noise import gaussian_noise

# Create function with explicit generator (reproducible)
noise_fn, state = gaussian_noise(stddev=1.1, generator=42)

for batch in dataloader:
    grads = compute_gradients(params, batch)
    noisy, state = noise_fn(grads, state)  # Reproducible
    params = update(params, noisy)
```

**Benefits**: Explicit state management, functional pattern

---

### Example 3: Different Clip Norms

**Old API**:

```python
# Standard clipping
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
sensitivity = grad_fn.sensitivity("REPLACE_SPECIAL")  # 1.0

# Rescale to unit norm
grad_fn_unit = clipped_grad(
    loss_fn,
    l2_clip_norm=5.0,
    rescale_to_unit_norm=True
)
sensitivity_unit = grad_fn_unit.sensitivity("REPLACE_SPECIAL")  # 1.0
```

**New API**:

```python
# Standard clipping
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
clip_norm = grad_fn.clip_norm  # 1.0

# Rescale to unit norm
grad_fn_unit = clipped_grad(
    loss_fn,
    l2_clip_norm=5.0,
    rescale_to_unit_norm=True
)
clip_norm_unit = grad_fn_unit.clip_norm  # 1.0 (rescaled)
```

**Benefits**: Same semantics, simpler attribute access

---

## Backward Compatibility

### Deprecated Functions

The following functions are **deprecated** but still work (with warnings):

1. **`add_gaussian_noise()`** - Use `gaussian()` or `gaussian_noise()`
2. **`.sensitivity()` method** - Use `.clip_norm` attribute

### Deprecation Timeline

- **v0.2.0** (current): Old API deprecated, warnings emitted
- **v0.3.0** - **v0.9.0**: Both APIs work, warnings continue
- **v1.0.0**: Old API removed

### Testing Your Migration

Run your tests with warnings enabled:

```bash
# See all deprecation warnings
pytest -W default::DeprecationWarning

# Or with Python
python -W default your_script.py
```

You should see warnings like:

```
DeprecationWarning: add_gaussian_noise() is deprecated and will be removed in version 1.0.0.
Use `gaussian_noise()` for stateless noise; for reproducible noise pass an explicit `generator` (e.g., `generator=42`) to `gaussian_noise()`.
```

---

## API Reference Changes

### Removed Exports

- `BoundedSensitivityCallable` - No longer exported (internal implementation detail removed)

### New Exports

- `gaussian_noise(stddev, generator=None)` - Stateless (or reproducible with `generator`) noise function factory

### Modified Behavior

- `clipped_grad()` returns plain `Callable` (not `BoundedSensitivityCallable`)
- `clipped_fun()` returns plain `Callable` (not `BoundedSensitivityCallable`)
- Both have `.clip_norm` attribute instead of `.l2_norm_bound` and `.sensitivity()` method

---

## Common Pitfalls

### Pitfall 1: Trying to call `.sensitivity()`

**Error**:
```python
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
sensitivity = grad_fn.sensitivity()  # AttributeError!
```

**Fix**:
```python
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
clip_norm = grad_fn.clip_norm  # Correct
```

### Pitfall 2: Using `add_gaussian_noise()` in a loop

**Old (inefficient)**:
```python
for batch in dataloader:
    grads = grad_fn(params, batch)
    noisy = add_gaussian_noise(grads, stddev=1.1)  # Deprecated + inefficient
```

**New (efficient)**:
```python
noise_fn, _ = gaussian_noise(stddev=1.1)  # Configure once (generator optional)
for batch in dataloader:
    grads = grad_fn(params, batch)
    noisy, _ = noise_fn(grads, _)  # Reuse; pass/receive state when using a generator
```

### Pitfall 3: Importing `BoundedSensitivityCallable`

**Old**:
```python
from opaque.clipping import BoundedSensitivityCallable  # ImportError!
```

**Fix**: Don't import it - you don't need it anymore! Functions are plain callables now.

---

## Benefits of the New API

1. ✅ **Simpler** - ~150 lines of wrapper code removed
2. ✅ **More explicit** - `stddev = noise_mult * clip_norm` is clearer than hidden sensitivity logic
3. ✅ **Same composability** - Easy to swap clipping/noise mechanisms
4. ✅ **Better performance** - Configure once, reuse many times
5. ✅ **Research-friendly** - No abstraction barriers for experimenting

---

## Need Help?

- **Documentation**: See updated [README.md](../README.md) and [examples/](../examples/)
- **Issues**: Report migration problems at https://github.com/anthropics/opaque/issues
- **Examples**: Check [examples/dp_sgd_simple.py](../examples/dp_sgd_simple.py) for complete working code

---

## Summary

**Key changes**:
1. Replace `.sensitivity()` → `.clip_norm`
2. Replace `add_gaussian_noise()` → `gaussian_noise()`
3. Configure noise functions once, reuse in loops

**Migration effort**: Low (mostly find-and-replace)

**Timeline**: You have until v1.0.0 to migrate (old API emits warnings)

**Questions?** Open an issue on GitHub!
