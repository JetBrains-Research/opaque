# Statefulness in JAX-Privacy: Distributed Training Analysis

## TL;DR - Your Question Answered

**Q: JAX-Privacy implements FSDP and other methods. That's where statefulness may fire back. How do they handle it?**

**A: They avoid statefulness in clipping entirely. Clipping is STATELESS (`BoundedSensitivityCallable` is `frozen=True`). State is only in `GradientTransformation` objects (noise, optimizer) which use explicit state-passing (pure functions).**

---

## Key Finding: JAX-Privacy's `BoundedSensitivityCallable` is Immutable

From `jax_privacy/clipping.py`:

```python
@dataclasses.dataclass(frozen=True)  # ← IMMUTABLE!
class BoundedSensitivityCallable:
    """Callable with a sensitivity property."""
    fun: Callable[..., Any]
    l2_norm_bound: float
    has_aux: bool

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def sensitivity(self, neighboring_relation=...):
        """Returns the L2 sensitivity (computed from l2_norm_bound)."""
        # Pure function - no internal state
```

**Critical**: The callable itself has NO mutable state. The `l2_norm_bound` is **constant**.

---

## How Distributed Training Works in JAX-Privacy

### 1. SPMD (Single Program Multiple Data)

JAX uses `jax.vmap(..., spmd_axis_name='data')` for data parallelism:

```python
# From jax_privacy/clipping.py line 360
microbatched_vmap_fun = optax.microbatching.micro_vmap(
    clipped_fun_one_group,
    in_axes=axes,
    microbatch_size=microbatch_size,
    accumulator=(sum_, concat, concat),
    num_real_microbatches=num_real_mb,
    vmap_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name),  # ← SPMD
)
```

**How it works**:
- `spmd_axis_name` tells JAX which axis is sharded across devices
- Each device processes its shard independently (SIMD execution)
- JAX automatically inserts collectives (all-reduce) when needed
- **No communication during clipping** - each device clips its local batch
- Communication happens during **aggregation** (sum across devices)

### 2. Distributed Noise Generation

From `jax_privacy/sharding_utils.py`:

```python
def flatten_with_zero_redundancy(abstract_array) -> jax.ShapeDtypeStruct:
    """Return flattened, padded, ZeRo-sharded abstract version.

    Zero-redundancy sharding: no redundant copies exist anywhere.
    """
    mesh = jax.typeof(abstract_array).sharding.mesh
    return jax.ShapeDtypeStruct(
        shape=(_ceiling_to_multiple(abstract_array.size, mesh.size),),
        dtype=abstract_array.dtype,
        sharding=jax.sharding.NamedSharding(mesh, jax.P(mesh.axis_names)),
    )

def local_reshape_add(x: jax.Array, y: jax.Array) -> jax.Array:
    """Reshapes y[:x.size] into x.shape and adds to x.

    Uses jax.shard_map to avoid inter-device communication.
    """
    per_device_shape = out_sharding.shard_shape(x.shape)
    per_device_size = math.prod(per_device_shape)

    reshape = jax.shard_map(
        lambda v: v[:per_device_size].reshape(per_device_shape),
        mesh=out_sharding.mesh,
        in_specs=_flatten_pspec(out_sharding.spec),
        out_specs=out_sharding.spec,
    )
    return (x + reshape(y)).astype(x.dtype)
```

**Key insight**: Noise is generated locally on each device, avoiding communication overhead.

### 3. Stateful Components Use Explicit State-Passing

From `jax_privacy/noise_addition.py`:

```python
def _dense_matrix_factorization_privatizer(...) -> optax.GradientTransformation:
    """Stateful noise addition via GradientTransformation."""

    def privatize(sum_of_clipped_grads, noise_state, params=None):
        index = noise_state  # ← State is EXPLICIT parameter
        matrix_row = noising_matrix[index] * stddev

        # Generate noise
        noise = optax.tree.random_like(...)
        noisy_grads = jax.tree.map(lambda g, n: g + n, sum_of_clipped_grads, noise)

        return noisy_grads, index + 1  # ← New state RETURNED

    init = lambda _: jnp.array(0)  # ← Pure initialization
    return optax.GradientTransformation(init, privatize)
```

**Pure functional pattern**:
- State is an explicit input parameter
- New state is an explicit output
- No mutation - state is immutable
- Works seamlessly with JAX's `jax.jit` and `jax.pmap`

---

## Comparison: Optax's Adaptive Gradient Clipping

Optax has `adaptive_grad_clip()` (AGC by Brock et al. 2021), which is **NOT** the same as Andrew et al. 2021 adaptive clipping:

```python
# From optax/transforms/_clipping.py
def adaptive_grad_clip(
    clipping: float,  # Maximum ratio of update/param norm
    eps: float = 1e-3,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
) -> base.GradientTransformation:
    """Clips updates to be at most clipping * parameter_norm, unit-wise.

    This is AGC (Adaptive Gradient Clipping) from Brock et al. 2021,
    NOT Andrew et al. 2021!
    """

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("AGC requires params!")

        # Compute unit-wise norms
        g_norm, p_norm = jax.tree.map(
            lambda x: unitwise_norm(x, axis=axis),
            (updates, params)
        )

        # Max norm = clipping * param_norm (per-unit)
        max_norm = jax.tree.map(lambda x: clipping * jnp.maximum(x, eps), p_norm)

        # Clip if grad_norm > clipping * param_norm
        updates = jax.tree.map(unitwise_clip, g_norm, max_norm, updates)

        return updates, state

    return base.GradientTransformation(base.init_empty_state, update_fn)
```

**Key differences from Andrew et al. 2021**:
- **No adaptive clip norm** - `clipping` is fixed
- Clips based on **parameter norm**, not fixed bound
- **Stateless** - `state = EmptyState()`
- Works on **aggregated** gradients, not per-example

**Works in distributed setting because**:
- Stateless (no mutable state)
- Operates on aggregated gradients
- Each device has full copy of params (FSDP replicates params needed for forward/backward)

---

## Optax's Per-Example Clipping

Optax also has `per_example_global_norm_clip()` (for DP-SGD):

```python
def per_example_global_norm_clip(
    grads: base.ArrayTree,  # ← Has batch dimension!
    l2_norm_clip: float,
) -> tuple[base.ArrayTree, jax.Array]:
    """Clips per-example gradients and sums them.

    Args:
        grads: Flattened update with BATCH DIMENSION on 0th axis

    Returns:
        Tuple of (summed_clipped_grads, num_clipped)
    """
    if not _check_arrays_have_batch_dim(grads):
        raise ValueError("Expects batch dimension on 0th axis!")

    # Compute per-example norms
    global_grad_norms = jax.vmap(optax.tree.norm)(grads)

    # Clip factors
    multipliers = jnp.nan_to_num(
        jnp.minimum(l2_norm_clip / global_grad_norms, 1.0), nan=1.0
    )

    num_clipped = jnp.sum(multipliers < 1.0)

    # Clip and sum in one operation
    clipped_sum = jax.tree.map(
        lambda g: jnp.tensordot(multipliers, g, axes=1),  # Weighted sum
        grads
    )

    return clipped_sum, num_clipped
```

**Key properties**:
- **Pure function** - no state
- Input: per-example grads `[B, ...]`
- Output: aggregated grad (single PyTree)
- Works in distributed setting via `jax.vmap` with `spmd_axis_name`

---

## Why JAX-Privacy Doesn't Implement Andrew et al. 2021

**Andrew et al. 2021 adaptive clipping requires**:
1. Tracking `clip_norm` across iterations (mutable state)
2. Geometric updates: `C_{t+1} = C_t * exp(η * sign(ρ_t - γ))`
3. Computing `ρ_t` = fraction of unclipped gradients in batch

**Problem for distributed training**:
- Mutable state doesn't work with `jax.jit` and `jax.pmap`
- Would need to track state across devices
- Communication overhead to compute global `ρ_t`

**JAX-Privacy's solution**: Don't implement it. Use stateless alternatives:
- Fixed clipping with `clipped_grad(l2_clip_norm=...)`
- AGC (Brock et al. 2021) which adapts to parameter norm, not clipping rate

---

## Implications for Our PyTorch Implementation

### Problem: Mutable State Doesn't Work Well with Distributed Training

Our current `adaptive_clipped_grad()` has mutable state:

```python
# src/opaque/clipping/adaptive.py (CURRENT - PROBLEMATIC)

state = {'clip_norm': initial_clip_norm, 'step': 0}  # ← Mutable dict!

def clipped_grad_fn(params, batch_x, batch_y):
    # ... compute gradients ...

    # Update mutable state
    state['clip_norm'] *= torch.exp(...)  # ← Mutation!
    state['step'] += 1

    return aggregated_grad

return AdaptiveSensitivityCallable(
    fun=clipped_grad_fn,
    state=state,  # ← Exposed for introspection
    ...
)
```

**Problems**:
1. **PyTorch DDP**: Each process has its own state copy - they diverge!
2. **PyTorch FSDP**: State isn't sharded - need manual synchronization
3. **torch.compile**: Mutable state breaks tracing
4. **Reproducibility**: State mutations are hard to debug

### Solution 1: Pure Functional State-Passing (Recommended)

Follow JAX-Privacy/Optax pattern - **explicit state parameter**:

```python
# src/opaque/clipping/adaptive.py (FUNCTIONAL - RECOMMENDED)

@dataclass(frozen=True)
class AdaptiveClipState:
    """Immutable state for adaptive clipping."""
    clip_norm: torch.Tensor
    step: int

def adaptive_clipped_grad(
    loss_fn: Callable,
    target_unclipped_quantile: float,
    clipbound_learning_rate: float,
    initial_clip_norm: float = 1.0,
) -> tuple[Callable, AdaptiveClipState]:
    """Returns (clipped_grad_fn, initial_state).

    clipped_grad_fn signature:
        clipped_grad_fn(params, batch, state) -> (grad, new_state)
    """

    def clipped_grad_fn(params, batch_x, batch_y, state: AdaptiveClipState):
        # Compute per-example gradients
        grads_per_example = torch.func.vmap(
            lambda x, y: torch.func.grad(loss_fn)(params, x, y)
        )(batch_x, batch_y)

        # Clip with current bound
        per_example_norms = compute_norms(grads_per_example)
        clip_factors = (state.clip_norm / per_example_norms).clamp(max=1.0)

        # Compute new clip norm
        unclipped_frac = (clip_factors == 1.0).float().mean()
        new_clip_norm = state.clip_norm * torch.exp(
            clipbound_learning_rate * (unclipped_frac - target_unclipped_quantile)
        )

        # Clip and aggregate
        clipped = tree_map(
            lambda g: g * clip_factors.view(-1, *[1]*(g.ndim-1)),
            grads_per_example
        )
        aggregated = tree_map(lambda g: g.sum(dim=0), clipped)

        # Return gradient AND new state
        new_state = AdaptiveClipState(clip_norm=new_clip_norm, step=state.step + 1)
        return aggregated, new_state

    initial_state = AdaptiveClipState(
        clip_norm=torch.tensor(initial_clip_norm),
        step=0,
    )

    return clipped_grad_fn, initial_state

# Usage
clipped_grad_fn, clip_state = adaptive_clipped_grad(loss_fn, ...)

for batch in data:
    # State is EXPLICIT parameter and return value
    grad, clip_state = clipped_grad_fn(params, batch_x, batch_y, clip_state)

    # Add noise
    noisy_grad = add_noise(grad, noise_multiplier * clip_state.clip_norm)

    # Update optimizer
    updates, opt_state = optimizer.update(noisy_grad, opt_state, params=params)
    params = apply_updates(params, updates)
```

**Benefits**:
- ✅ Works with PyTorch DDP/FSDP (state is passed explicitly)
- ✅ Works with `torch.compile` (state is traced)
- ✅ Reproducible (pure function)
- ✅ Matches JAX-Privacy pattern

**Distributed training**:
```python
# With DDP
for batch in data:
    grad, clip_state = clipped_grad_fn(params, batch_x, batch_y, clip_state)

    # Synchronize clip_norm across devices (all-reduce)
    clip_state = AdaptiveClipState(
        clip_norm=dist.all_reduce(clip_state.clip_norm, op=dist.ReduceOp.AVG),
        step=clip_state.step,
    )

    # Continue with noise + optimizer...
```

### Solution 2: Use TorchOpt GradientTransformation (More Complex)

Integrate clipping into the GradientTransformation pattern:

```python
# This works but requires special handling of per-example gradients

def adaptive_clip_per_example() -> GradientTransformation:
    """IMPORTANT: Expects per-example gradients [B, ...] not aggregated!"""

    def init_fn(params):
        return AdaptiveClipState(clip_norm=torch.tensor(1.0), step=0)

    def update_fn(updates, state, *, params=None):
        # updates has BATCH dimension (special case!)
        # ... clipping logic ...
        # Return AGGREGATED gradient + new state
        return aggregated_grad, new_state

    return GradientTransformation(init_fn, update_fn)

# Usage - requires custom gradient computation
for batch in data:
    # 1. Compute per-example gradients (NOT standard!)
    grads_per_example = compute_per_example_grads(params, batch)

    # 2. Apply clipping (per-example → aggregated)
    aggregated_grad, clip_state = clip_transform.update(
        grads_per_example, clip_state, params=params
    )

    # 3. Apply noise + optimizer (aggregated → DP → updates)
    updates, opt_state = optimizer.update(aggregated_grad, opt_state, params=params)
    params = apply_updates(params, updates)
```

**Problems**:
- Breaks `GradientTransformation` convention (expects aggregated grads)
- Confusing for users
- Can't easily compose with standard transformations

---

## Recommendation

### For Single-Device Training

**Use pure functional state-passing** (Solution 1):

```python
clipped_grad_fn, clip_state = adaptive_clipped_grad(...)

for batch in data:
    grad, clip_state = clipped_grad_fn(params, batch, clip_state)  # Explicit state
    # ... noise + optimizer ...
```

**Benefits**:
- Simple
- Pure functional
- Matches JAX-Privacy pattern
- Easy to reason about

### For Distributed Training

**Either**:

1. **Manual synchronization** with `dist.all_reduce()` on `clip_state.clip_norm`
2. **Don't use adaptive clipping** - use fixed clipping instead (JAX-Privacy approach)

**Why JAX-Privacy doesn't implement Andrew et al. 2021**:
- Requires global state synchronization
- Communication overhead
- Complexity doesn't justify benefits
- Fixed clipping with well-tuned bounds works well in practice

---

## Summary

**Your Question**: How does JAX-Privacy handle statefulness with FSDP and distributed training?

**Answer**:

1. **Clipping is stateless**: `BoundedSensitivityCallable` is `frozen=True` (immutable)
2. **Stateful components use explicit state-passing**: `GradientTransformation` returns new state
3. **No mutable state anywhere**: Everything is pure functional
4. **Distributed works because**: JAX's `jax.vmap` with `spmd_axis_name` handles sharding automatically
5. **They DON'T implement Andrew et al. 2021**: Too complex for distributed setting

**Our recommendation**:
- Use **pure functional state-passing** for adaptive clipping
- State is explicit parameter and return value
- No mutation - return new state each iteration
- Synchronize state across devices with `dist.all_reduce()` if using DDP/FSDP
- Consider using **fixed clipping** for distributed training (simpler)
