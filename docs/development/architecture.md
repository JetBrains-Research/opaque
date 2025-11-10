# Architecture

Opaque's architecture is designed around **functional composition** and **modularity**, inspired by JAX-Privacy's modern experimental API.

---

## Design Philosophy

### 1. Functional API

Opaque uses a functional API inspired by JAX-Privacy's `experimental/clipping.py` module:

- **Composable primitives**: Clipping, noise, and accounting are separate modules
- **Stateless functions**: Easier testing and debugging
- **Explicit over implicit**: Fail-fast error handling

**Example**:
```python
# Separate concerns
clipped_grads = clipped_grad(loss_fn, l2_clip_norm=1.0)(params, data)
noisy_grads = add_noise(clipped_grads, noise_scale=σ)
```

### 2. LoRA-First Approach

Optimized for parameter-efficient fine-tuning:

- Per-example gradients only for adapter weights (not full model)
- Memory efficiency through microbatching
- LoRA structure exploitation: rank `r << d`

**Memory savings**:
```
Full model:    O(batch_size × d × k)
LoRA only:     O(batch_size × r)     where r << d
```

### 3. Test-Driven Development

Every feature follows the TDD workflow:

1. Discover JAX-Privacy behavior
2. Write JAX reference test (optional)
3. Write failing Opaque test
4. Implement to pass tests
5. Document and create examples

See [TDD Workflow](tdd-workflow.md) for details.

---

## Module Structure

```
opaque/
├── core/                      # Core DP primitives
│   ├── pytree_utils.py       # PyTree operations
│   ├── clipping.py           # Per-example gradient clipping
│   └── noise.py              # Gaussian noise (future)
├── accounting/                # Privacy budget tracking (future)
│   └── calibration.py
├── optim/                     # DP-aware optimizers (future)
│   └── dp_optimizer.py
├── lora/                      # LoRA-specific utilities (future)
│   └── utils.py
└── api.py                     # High-level API (future)
```

---

## PyTorch Functional Equivalents

Opaque leverages `torch.func` for functional transformations:

| JAX | Purpose | PyTorch |
|-----|---------|---------|
| `jax.vmap(fn, in_dims=...)` | Vectorize over batch | `torch.func.vmap(fn, in_dims=...)` |
| `jax.grad(fn, argnums=...)` | Automatic differentiation | `torch.func.grad(fn, argnums=...)` |
| `jax.lax.scan(fn, init, xs)` | Sequential accumulation | Custom loop or checkpointing |
| `jax.tree_util.tree_map(fn, tree)` | Apply to nested structures | `torch.utils._pytree.tree_map(fn, tree)` |
| `jax.random.PRNGKey(seed)` | Reproducible randomness | `torch.Generator().manual_seed(seed)` |
| `optax.GradientTransformation` | Optimizer interface | `torch.optim.Optimizer` |

---

## Core Abstractions

### PyTree

**What it is**: Nested dict structure of tensors (model parameters)

**Example**:
```python
params = {
    "layer1": {
        "weight": torch.randn(10, 5),
        "bias": torch.randn(10),
    },
    "layer2": {
        "weight": torch.randn(5, 3),
    }
}
```

**Operations**:
- `tree_leaves(tree)`: Extract all tensors
- `tree_map(fn, tree)`: Apply function to all tensors
- `global_norm(tree)`: Compute L2 norm across all tensors

### Per-Example Gradients

**Problem**: Standard backprop computes batch-averaged gradients

**Solution**: Use `torch.func.vmap` to vectorize over batch:

```python
# Standard: batch-averaged gradient
loss = loss_fn(params, data_batch)
grads = torch.autograd.grad(loss, params)  # Single gradient

# DP-SGD: per-example gradients
grad_fn = torch.func.grad(loss_fn, argnums=0)
per_example_grad_fn = torch.func.vmap(
    grad_fn,
    in_dims=(None, 0)  # params shared, data batched
)
per_example_grads = per_example_grad_fn(params, data_batch)  # [B, param_shape]
```

### Gradient Clipping

**How it works**:
1. Compute per-example gradients (one gradient per training example)
2. Compute L2 norm of each example's gradient
3. Clip to maximum norm if exceeded: `g_clipped = g * min(1, C / ||g||)`
4. Sum clipped gradients

**Sensitivity**: Maximum influence of single example = clip norm `C`

---

## Key Design Decisions

### 1. PyTree Implementation

**Decision**: Use `torch.utils._pytree` (private API)

**Rationale**: Full PyTorch compatibility, handles nested structures

**Mitigation**: Thin wrapper module for easy migration if API changes

See [Design Decisions](design-decisions.md#1-pytree-implementation) for details.

### 2. Microbatching

**Decision**: Explicit `microbatch_size` parameter (user control)

**Rationale**: Users doing DP understand memory constraints, explicit > implicit

**API**:
```python
clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=32,  # Process batch in chunks of 32
)
```

See [Design Decisions](design-decisions.md#2-microbatching-strategy) for details.

### 3. Error Handling

**Decision**: Fail-fast by default, opt-in for graceful handling

**Rationale**: DP is security-critical, surprises are dangerous

**API**:
```python
clip_pytree(tree, clip_norm, nan_safe=False)  # Fail on NaN by default
```

See [Design Decisions](design-decisions.md#4-error-handling-philosophy) for details.

---

## Comparison with JAX-Privacy

Opaque ports JAX-Privacy's **functional API** (`experimental/clipping.py`), not the older object-oriented API.

### Why?

- **Modular**: Clipping and noise are separate
- **Composable**: Functions chain naturally
- **Testable**: No hidden state
- **Future-proof**: JAX-Privacy moving this direction

See [JAX-Privacy Comparison](jax-privacy-comparison.md) for full analysis.

---

## Related Documentation

- [JAX-Privacy Comparison](jax-privacy-comparison.md) - Which API we're porting and why
- [Design Decisions](design-decisions.md) - Technical decision rationale
- [TDD Workflow](tdd-workflow.md) - Development process
- [Stage 1 Plan](stage1-plan.md) - Implementation details
