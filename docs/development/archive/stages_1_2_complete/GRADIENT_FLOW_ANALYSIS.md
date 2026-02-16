# Per-Example vs Aggregated Gradients: Critical Analysis for DP-SGD

## TL;DR - Your Question Answered

**Q: Does Optax expect summed/averaged gradients, can it process per-sample, does it expect to?**

**A: Optax expects AGGREGATED (summed) gradients, NOT per-example gradients.**

The DP-SGD workflow is:
1. **Clipping layer** → Processes per-example gradients, clips them, sums them → Outputs aggregated gradient
2. **Noise layer** → Adds noise to aggregated gradient → Outputs DP-aggregated gradient
3. **Optimizer layer** → Updates parameters using DP-aggregated gradient

**Key Insight**: Clipping and noise are SEPARATE stages from the optimizer. The optimizer never sees per-example gradients.

---

## JAX-Privacy Architecture

### Complete DP-SGD Pipeline

From `jax_privacy/experimental/execution_plan.py` (lines 81-87):

```python
noise_state = noise_addition_transform.init(...)
for indices in batch_selection_strategy():
    batch = data.select(indices)

    # 1. CLIPPING: per-example → aggregated
    clipped_grad = clipped_aggregation_fn(batch, ...)
    # ^ Returns SUMMED clipped gradients (single PyTree)

    # 2. NOISE: aggregated → DP-aggregated
    dp_grad, noise_state = noise_addition_transform.update(
        clipped_grad, noise_state
    )
    # ^ Takes aggregated, returns DP-aggregated

    # 3. OPTIMIZER: DP-aggregated → parameter updates
    # (This happens after - optimizer never sees per-example grads)
    del indices, batch, clipped_grad  # Security: delete sensitive data
```

### Component Details

#### 1. Clipping Layer: `clipped_fun()` / `clipped_grad()`

**Input**: Batch of examples (per-example data)
**Process**:
- Compute per-example gradients using `jax.vmap`
- Clip each per-example gradient to L2 norm
- **Sum** all clipped gradients
**Output**: Single aggregated PyTree (summed clipped gradients)

```python
# From jax_privacy/clipping.py
def clipped_fun(...) -> BoundedSensitivityCallable:
    """Transforms a function to clip its output and sum across a batch."""

    def clipped_fn(*args, **kwargs):
        # vmap over batch dimension
        per_example_values = jax.vmap(fun)(*batch_args, **kwargs)

        # Clip each example
        clipped_values = jax.vmap(clip_pytree)(per_example_values, clip_norm, ...)

        # SUM across batch (not mean!)
        aggregated = jax.tree.map(lambda x: x.sum(axis=0), clipped_values)

        return aggregated  # Single PyTree, no batch dimension

    return BoundedSensitivityCallable(clipped_fn, l2_norm_bound, has_aux)
```

**Key Property**: `clipped_aggregation_fn.sensitivity()` returns the L2 sensitivity of the AGGREGATED output.

#### 2. Noise Layer: `optax.GradientTransformation`

**Input**: Aggregated gradient (single PyTree)
**Process**: Add Gaussian noise scaled by sensitivity
**Output**: DP-aggregated gradient (single PyTree)

From `jax_privacy/noise_addition.py` (line 237):

```python
def privatize(sum_of_clipped_grads, noise_state, params=None):
    """Add noise to AGGREGATED gradients.

    Args:
        sum_of_clipped_grads: Single PyTree (NOT per-example!)
        noise_state: PRNG key or matrix factorization state
    """
    # Generate noise matching structure of sum_of_clipped_grads
    noise = optax.tree.random_like(
        rng_key=prng_key,
        target_tree=sum_of_clipped_grads,  # Single PyTree
        sampler=jax.random.normal,
        dtype=dtype,
    )

    # Add noise element-wise
    noisy_grads = jax.tree.map(lambda g, n: g + n, sum_of_clipped_grads, noise)

    return noisy_grads, new_noise_state

return optax.GradientTransformation(init, privatize)
```

**Critical**: The signature is `update(updates, state, params=None)` where `updates` is a **single PyTree**, not a batch.

#### 3. Optimizer Layer: Standard Optax

**Input**: DP-aggregated gradient (single PyTree)
**Process**: Standard optimizer update (Adam, SGD, etc.)
**Output**: Parameter updates

```python
# Standard Optax - no knowledge of per-example gradients
optimizer = optax.adamw(lr=0.001)
opt_state = optimizer.init(params)

# dp_grad is AGGREGATED, not per-example
updates, opt_state = optimizer.update(dp_grad, opt_state, params=params)
params = optax.apply_updates(params, updates)
```

---

## PyTorch/TorchOpt Implications

### Current Problem: Confusion of Responsibilities

Our current `adaptive_clipped_grad()` mixes concerns:

```python
# src/opaque/clipping/adaptive.py (WRONG ARCHITECTURE)

clipped_grad_fn = adaptive_clipped_grad(loss_fn, ...)

for batch in data:
    # This does BOTH:
    # 1. Per-example clipping + aggregation
    # 2. Adaptive clip norm updates
    # 3. Returns AGGREGATED gradient (no noise yet!)
    grads = clipped_grad_fn(params, batch_x, batch_y)

    # Then noise is added OUTSIDE
    noisy_grads = add_noise(grads, ...)

    # Then optimizer is applied
    optimizer.step()
```

**Problem**: Clipping layer is stateful and separate from optimizer. Can't compose with other transformations.

### Correct Architecture: Three Separate Transformations

```python
# CORRECT: Functional composition via torchopt.chain()

# 1. CLIPPING TRANSFORMATION
def per_example_clip(
    l2_clip_norm: float,
    loss_fn: Callable,
) -> GradientTransformation:
    """Clips per-example gradients and sums them.

    Input: (params, batch) - batch has shape [B, ...]
    Output: aggregated_grad - no batch dimension
    """

    def init_fn(params):
        return EmptyState()

    def update_fn(updates, state, *, params=None):
        # updates is a BATCH of per-example gradients [B, ...]
        # NOT the standard aggregated gradient!

        # Clip each example
        per_example_norms = compute_per_example_norms(updates)  # [B]
        clip_factors = (l2_clip_norm / per_example_norms).clamp(max=1.0)

        # Apply clipping
        clipped = tree_map(
            lambda u: u * clip_factors.view(-1, *[1]*(u.ndim-1)),
            updates
        )

        # SUM across batch (output is single PyTree)
        aggregated = tree_map(lambda u: u.sum(dim=0), clipped)

        return aggregated, state

    return GradientTransformation(init_fn, update_fn)

# 2. ADAPTIVE CLIPPING TRANSFORMATION
def adaptive_clip_grad_norm(...) -> GradientTransformation:
    """Adapts clipping bound based on clipping rate.

    Input: per-example gradients [B, ...]
    Output: aggregated clipped gradients (single PyTree)
    """

    def update_fn(updates, state, *, params=None):
        # Same as above but updates clip norm adaptively
        per_example_norms = compute_per_example_norms(updates)

        # Adaptive logic
        unclipped_frac = (per_example_norms < state.clip_norm).float().mean()
        new_clip_norm = state.clip_norm * exp(lr * (unclipped_frac - target))

        # Clip and aggregate
        clip_factors = (new_clip_norm / per_example_norms).clamp(max=1.0)
        clipped = tree_map(lambda u: u * clip_factors.view(-1, *[1]*(u.ndim-1)), updates)
        aggregated = tree_map(lambda u: u.sum(dim=0), clipped)

        return aggregated, AdaptiveClipState(clip_norm=new_clip_norm, ...)

    return GradientTransformation(init_fn, update_fn)

# 3. NOISE TRANSFORMATION
def add_gaussian_noise(
    noise_multiplier: float,
    l2_sensitivity: float,
) -> GradientTransformation:
    """Adds Gaussian noise to aggregated gradients.

    Input: aggregated_grad (single PyTree, no batch dim)
    Output: noisy_grad (single PyTree)
    """

    def update_fn(updates, state, *, params=None):
        # updates is AGGREGATED (single PyTree)
        stddev = noise_multiplier * l2_sensitivity
        noise = tree_map(lambda u: torch.randn_like(u) * stddev, updates)
        noisy = tree_map(lambda u, n: u + n, updates, noise)
        return noisy, state

    return GradientTransformation(init_fn, update_fn)

# COMPOSE THEM
dp_sgd_pipeline = torchopt.chain(
    adaptive_clip_grad_norm(target_quantile=0.5, lr=0.2),  # per-example → aggregated
    add_gaussian_noise(noise_multiplier=1.1, l2_sensitivity=1.0),  # aggregated → DP
    torchopt.adamw(lr=0.001),  # DP → parameter updates
)
```

---

## The Critical Question: What is `updates` in `update_fn`?

### Standard Optax/TorchOpt Usage

In standard (non-DP) training:

```python
optimizer = torchopt.adamw(lr=0.001)
state = optimizer.init(params)

for batch in data:
    # Compute AGGREGATED gradient (batch-averaged or batch-summed)
    loss = compute_loss(params, batch)
    grads = torch.autograd.grad(loss, params.values())  # Single PyTree
    grads = dict(zip(params.keys(), grads))

    # Apply optimizer to AGGREGATED gradient
    updates, state = optimizer.update(grads, state, params=params)
    params = torchopt.apply_updates(params, updates)
```

**Key**: `grads` is a **single PyTree** (dict of tensors), NOT a batch of gradients.

### DP-SGD Usage: What Changes?

For DP-SGD, we need per-example gradients for clipping. Two approaches:

#### Approach 1: Compute Per-Example Gradients, Then Use Transformations

```python
# DP-SGD pipeline
clipping_transform = adaptive_clip_grad_norm(...)
noise_transform = add_gaussian_noise(...)
optimizer = torchopt.adamw(lr=0.001)

# Compose
dp_sgd = torchopt.chain(clipping_transform, noise_transform, optimizer)

# Initialize
state = dp_sgd.init(params)

for batch in data:
    # 1. Compute PER-EXAMPLE gradients using vmap
    def loss_per_example(params, x, y):
        return compute_loss(params, x, y)

    grads_per_example = torch.func.vmap(
        lambda x, y: torch.func.grad(loss_per_example)(params, x, y),
        in_dims=(0, 0)
    )(batch_x, batch_y)
    # grads_per_example has batch dimension: [B, ...]

    # 2. Apply DP-SGD pipeline
    # - clipping_transform: [B, ...] → aggregated (single PyTree)
    # - noise_transform: aggregated → DP-aggregated
    # - optimizer: DP-aggregated → updates
    updates, state = dp_sgd.update(grads_per_example, state, params=params)
    params = torchopt.apply_updates(params, updates)
```

**Problem**: The first transformation in the chain expects per-example gradients `[B, ...]`, but standard optimizers expect aggregated gradients!

#### Approach 2: Separate Clipping from Optimizer Chain

```python
# Clipping is SEPARATE (not a GradientTransformation)
clipping_fn = adaptive_clipped_grad(loss_fn, ...)

# Only noise + optimizer are chained
dp_optimizer = torchopt.chain(
    add_gaussian_noise(...),
    torchopt.adamw(lr=0.001),
)

state = dp_optimizer.init(params)

for batch in data:
    # 1. Clipping (per-example → aggregated)
    clipped_grads = clipping_fn(params, batch_x, batch_y)  # Single PyTree

    # 2. Noise + optimizer (aggregated → DP → updates)
    updates, state = dp_optimizer.update(clipped_grads, state, params=params)
    params = torchopt.apply_updates(params, updates)
```

**Better**: Clipping is separate function that returns aggregated gradients. Optimizer chain operates on standard aggregated gradients.

---

## Recommendation: Hybrid Approach

### Design Decision

**Clipping should NOT be a `GradientTransformation`** because:

1. It operates on **per-example gradients** (batch dimension present)
2. Standard `GradientTransformation.update()` expects **aggregated gradients** (no batch dimension)
3. Mixing these in a chain would be confusing and error-prone

**Instead**:
- **Clipping**: Use `BoundedSensitivityCallable` pattern (JAX-Privacy approach)
- **Noise**: Use `GradientTransformation`
- **Optimizer**: Use `GradientTransformation` (standard)

### Proposed Architecture

```python
# src/opaque/clipping/adaptive.py

def adaptive_clipped_grad(
    loss_fn: Callable,
    target_unclipped_quantile: float,
    clipbound_learning_rate: float,
    initial_clip_norm: float = 1.0,
    ...
) -> BoundedSensitivityCallable:
    """Adaptive per-example gradient clipping.

    Returns a callable that:
        - Takes (params, batch_x, batch_y)
        - Computes per-example gradients
        - Clips them adaptively
        - Sums them
        - Returns aggregated gradient (single PyTree)
    """

    state = {'clip_norm': initial_clip_norm, 'step': 0}

    def clipped_grad_fn(params, batch_x, batch_y):
        # Compute per-example gradients
        grads_per_example = torch.func.vmap(
            lambda x, y: torch.func.grad(loss_fn)(params, x, y)
        )(batch_x, batch_y)

        # Adaptive clipping logic
        per_example_norms = compute_norms(grads_per_example)
        clip_factors = (state['clip_norm'] / per_example_norms).clamp(max=1.0)

        # Update clip norm
        unclipped_frac = (clip_factors == 1.0).float().mean()
        state['clip_norm'] *= torch.exp(
            clipbound_learning_rate * (unclipped_frac - target_unclipped_quantile)
        )

        # Clip and aggregate
        clipped = tree_map(
            lambda g: g * clip_factors.view(-1, *[1]*(g.ndim-1)),
            grads_per_example
        )
        aggregated = tree_map(lambda g: g.sum(dim=0), clipped)

        return aggregated  # Single PyTree

    # Calculate sensitivity
    l2_norm_bound = state['clip_norm']  # or 1.0 if rescale

    return BoundedSensitivityCallable(
        fun=clipped_grad_fn,
        l2_norm_bound=l2_norm_bound,
        has_aux=False,
    )

# Usage
clipped_grad_fn = adaptive_clipped_grad(loss_fn, target_quantile=0.5, lr=0.2)

# Noise + optimizer as GradientTransformation
dp_optimizer = torchopt.chain(
    add_gaussian_noise(
        noise_multiplier=1.1,
        l2_sensitivity=clipped_grad_fn.sensitivity(),  # Use sensitivity!
    ),
    torchopt.adamw(lr=0.001),
)

opt_state = dp_optimizer.init(params)

for batch in data:
    # 1. Clipping: (params, batch) → aggregated gradient
    clipped_grad = clipped_grad_fn(params, batch_x, batch_y)

    # 2. Noise + optimizer: aggregated → DP → updates
    updates, opt_state = dp_optimizer.update(clipped_grad, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)
```

### Why This Works

1. **Clipping**: Handles per-example → aggregated transformation
2. **Noise**: Operates on aggregated gradients (standard interface)
3. **Optimizer**: Operates on aggregated gradients (standard interface)
4. **Composability**: Noise + optimizer can be chained with `torchopt.chain()`
5. **Sensitivity**: `.sensitivity()` method provides L2 bound for noise calibration

---

## Summary

**Your Question**: Does Optax expect summed/averaged gradients, can it process per-sample?

**Answer**:
- Optax/TorchOpt `GradientTransformation` expects **AGGREGATED** gradients (single PyTree)
- Per-example gradient processing happens in the **clipping layer** (before the optimizer chain)
- Clipping layer outputs **aggregated** gradients that are compatible with standard optimizers
- The optimizer never sees per-example gradients

**Architecture**:
```
Per-example data → [Clipping Layer] → Aggregated gradient → [Noise Layer] → DP gradient → [Optimizer Layer] → Updates
                    (BoundedSensitivityCallable)           (GradientTransformation chain)
```

**Key Insight**: Don't try to make clipping a `GradientTransformation`. Keep it as a separate `BoundedSensitivityCallable` that outputs aggregated gradients. Only noise + optimizer need to be `GradientTransformation` objects.
