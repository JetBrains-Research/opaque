# Stage 1: Core Clipping Module

**Goal**: Implement per-example gradient clipping without noise (noise comes later in Stage 2)

**Timeline**: 3 weeks

**Status**: 📋 Ready to start

---

## Overview

Stage 1 focuses on porting JAX-Privacy's `experimental/clipping.py` module to PyTorch. This provides the foundation for DP-SGD by implementing per-example gradient clipping with bounded sensitivity.

### Deliverables

1. **`opaque.core.pytree_utils`** (~100 LOC)
   - `global_norm()` - Compute L2 norm across PyTree
   - `tree_leaves()` - Extract tensors from PyTree
   - `tree_map()` - Apply function to PyTree leaves

2. **`opaque.core.clipping`** (~400 LOC)
   - `clip_pytree()` - Clip PyTree to max L2 norm
   - `clipped_grad()` - Main API for per-example clipped gradients

3. **Tests** (~400 LOC)
   - Unit tests for all functions
   - JAX-Privacy numerical validation
   - Property-based tests

---

## Week 1: PyTree Utilities + Basic Clipping

### Days 1-2: PyTree Utilities

**File**: `src/opaque/core/pytree_utils.py`

**Functions to implement**:

#### 1. `global_norm(tree: dict) -> torch.Tensor`
Compute L2 norm of all tensors in tree.

```python
def global_norm(tree):
    """Compute L2 norm of all tensors in tree."""
    squares = []
    for tensor in tree_leaves(tree):
        squares.append((tensor ** 2).sum())
    return torch.sqrt(sum(squares))
```

**Tests to write**:
- Single tensor
- Multiple tensors
- Nested dicts
- Empty trees
- Mixed dtypes/devices

#### 2. `tree_leaves(tree: dict) -> list[torch.Tensor]`
Wrapper around `torch.utils._pytree.tree_leaves`

#### 3. `tree_map(fn: Callable, *trees) -> dict`
Wrapper around `torch.utils._pytree.tree_map`

### Days 3-4: Basic Clipping

**File**: `src/opaque/core/clipping.py` (Part 1)

**Implement**: `clip_pytree()` with all edge cases

**Edge cases to handle**:
1. `clip_norm = 0` → return zero tree
2. `clip_norm = inf` → return original tree
3. `tree_norm = 0` → return original tree (avoid division by zero)
4. NaN safety: `nan_to_num()` if `nan_safe=True`
5. `rescale_to_unit_norm=True` → divide by `clip_norm` after clipping

**Tests**:
- Verify output norm ≤ clip_norm
- Test with different dtypes (float16, float32, float64)
- Test on GPU (if available)

### Day 5: Simple `clipped_grad()` (No Microbatching)

**File**: `src/opaque/core/clipping.py` (Part 2)

**Key challenge**: Understanding `torch.func.vmap` in_dims

Example:
```python
# Loss function signature: loss_fn(param, data)
# param: no batch dim
# data: has batch dim (e.g., shape [B, D])

grad_fn = torch.func.grad(loss_fn, argnums=0)  # w.r.t. param

# To get per-example gradients:
vmapped_grad = torch.func.vmap(
    grad_fn,
    in_dims=(None, 0)  # param shared, data batched
)
per_example_grads = vmapped_grad(param, data)  # shape [B, param_shape]
```

**Tests**:
- Compare against manual loop over examples
- Verify gradient correctness with toy linear model
- Test with `rescale_to_unit_norm=True/False`
- Test with `normalize_by` parameter

---

## Week 2: Numerical Validation

### Goal
Ensure PyTorch implementation matches JAX-Privacy numerically

### Toy Problem: Linear Regression

```python
# JAX version
import jax.numpy as jnp
from jax_privacy.experimental.clipping import clipped_grad

def jax_loss(w, x, y):
    return 0.5 * jnp.mean((x @ w - y) ** 2)

jax_grad_fn = clipped_grad(jax_loss, l2_clip_norm=1.0)

# PyTorch version
import torch
from opaque.core.clipping import clipped_grad

def torch_loss(w, x, y):
    return 0.5 * ((x @ w - y) ** 2).mean()

torch_grad_fn = clipped_grad(torch_loss, l2_clip_norm=1.0)

# Compare on same data
w_init = ...  # same random init
x_batch = ...  # same data
y_batch = ...

jax_result = jax_grad_fn(w_jax, x_jax, y_jax)
torch_result = torch_grad_fn(w_torch, x_torch, y_torch)

assert torch.allclose(torch_result, jax_to_torch(jax_result), atol=1e-5)
```

### Acceptance Criteria
- Numerical difference < 1e-5 for float32
- Works with batches of size 1, 32, 256
- Works with multiple parameters (dict of tensors)

---

## Week 3: Microbatching + Integration

### Days 1-2: Implement Microbatching

**Goal**: Memory optimization by processing batch in chunks

**Implementation**:
- Split batch into chunks
- Process sequentially with `for` loop
- Accumulate clipped gradients

**Verification**:
- Memory profiling to verify benefit
- Results identical to non-microbatched version

### Days 3-4: Integration Test with Real LoRA Layer

**Goal**: Test with realistic use case

**Steps**:
1. Use `transformers` library's `Linear` layer
2. Attach LoRA adapters manually
3. Compute clipped gradients for LoRA params only

### Day 5: Documentation + Examples

**Tasks**:
- Write `examples/01_linear_regression.py`
- Add comprehensive docstrings
- Update CLAUDE.MD with learnings

---

## Success Criteria

1. ✅ `clip_pytree()` handles all edge cases correctly
2. ✅ `clipped_grad()` matches JAX-Privacy output within 1e-5
3. ✅ Works with dictionaries of tensors (PyTrees)
4. ✅ Microbatching reduces memory usage without changing results
5. ✅ Tests pass on CPU and GPU
6. ✅ Code is documented and formatted with Ruff

---

## What Comes Next

After Stage 1 is complete:

- **Stage 2**: Noise Injection (`opaque.core.noise`)
- **Stage 3**: Privacy Accounting (`opaque.accounting`)
- **Stage 4**: High-Level API (`opaque.api`)

See [Project Roadmap](../development/roadmap.md) for full timeline.
