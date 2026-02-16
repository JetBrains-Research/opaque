# Microbatching & Memory Management: JAX-Privacy vs Opacus

## TL;DR - Your Question Answered

**Q: Does JAX-Privacy have some sort of microbatching, memory manager like Opacus?**

**A: Yes, but fundamentally different approach:**

| Feature | JAX-Privacy | Opacus |
|---------|-------------|--------|
| **Microbatching** | ✅ Via Optax `micro_vmap` | ✅ Via accumulated gradients |
| **Memory Manager** | ❌ No explicit manager | ✅ `GradSampleModule` + hooks |
| **Approach** | Functional (reshape + loop) | OOP (gradient hooks) |
| **Per-example grads** | Computed with `vmap` | Computed with hooks |
| **Memory strategy** | Sequential processing | Accumulation + checkpointing |

---

## JAX-Privacy: Optax `micro_vmap` Approach

### Architecture

JAX-Privacy uses **Optax's `micro_vmap`** - a generalized `vmap` that supports microbatching:

```python
# From jax_privacy/clipping.py line 354
microbatched_vmap_fun = optax.microbatching.micro_vmap(
    clipped_fun_one_group,
    in_axes=axes,
    microbatch_size=microbatch_size,  # e.g., 64
    accumulator=(sum_, concat, concat),  # How to combine results
    num_real_microbatches=num_real_mb,
    vmap_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name),
)
```

### How `micro_vmap` Works

From `optax/microbatching/_microbatching.py`:

```python
def micro_vmap(
    fun: Function,
    in_axes: int | Sequence[int] = 0,
    microbatch_size: int | None = None,
    accumulator: AccumulationType = AccumulationType.CONCAT,
    ...
) -> Function:
    """A generalized vmap that supports microbatching.

    Conceptually does:
        def microbatched_fun(full_batch):
            microbatches = split_batch(full_batch, microbatch_size)
            accumulator = init()
            for microbatch in microbatches:
                result = vmap(fun)(microbatch)
                accumulator = update(accumulator, result)
            return finalize(accumulator)
    """
```

**Key mechanism**:

1. **Reshape batch axis**: `[B, ...] → [num_microbatches, microbatch_size, ...]`
   ```python
   def reshape_batch_axis(tree, microbatch_size, axis=0):
       """Reshape using Fortran order to preserve sharding."""
       new_shape = x.shape[:axis] + (-1, microbatch_size) + x.shape[axis+1:]
       return x.reshape(new_shape, order='F')  # Column-major!
   ```

2. **Loop over microbatches**: Process sequentially
   ```python
   for microbatch_idx in range(num_microbatches):
       microbatch_result = jax.vmap(fun)(microbatch)
       accumulator = update(accumulator, microbatch_result, microbatch_idx)
   ```

3. **Accumulate results**: Using `Accumulator` objects
   ```python
   class AccumulationType(enum.Enum):
       SUM = auto()      # Sum microbatch outputs
       MEAN = auto()     # Average microbatch outputs
       CONCAT = auto()   # Concatenate along axis 0
       RUNNING_MEAN = auto()  # Running average
   ```

### Example: Clipping with Microbatching

```python
# Full batch: [1024, ...]
clipped_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=64,  # Process 64 examples at a time
)

# Under the hood:
# 1. Reshape: [1024, ...] → [16, 64, ...]
# 2. Loop 16 times:
#    - vmap over 64 examples (compute per-example grads)
#    - Clip each gradient
#    - Sum clipped gradients (partial sum)
# 3. Final sum across all microbatches
# Output: Single aggregated gradient

grad = clipped_grad_fn(params, batch_x, batch_y)
```

### Memory Benefits

**Without microbatching**:
```python
# vmap over full batch
per_example_grads = jax.vmap(compute_grad)(batch)  # [1024, num_params]
# Memory: O(batch_size * num_params)
```

**With microbatching**:
```python
# vmap over microbatch, loop over num_microbatches
for microbatch in split(batch, microbatch_size=64):
    per_example_grads = jax.vmap(compute_grad)(microbatch)  # [64, num_params]
    # Memory: O(microbatch_size * num_params)
```

**Savings**: `batch_size / microbatch_size` reduction in memory for per-example gradients.

### No Explicit Memory Manager

JAX-Privacy does NOT have a memory manager like Opacus. Instead:
- Users manually set `microbatch_size` based on available memory
- JAX's memory management is automatic (XLA compiler optimizes)
- No profiling or automatic batch size selection

---

## Opacus: GradSampleModule + Hooks Approach

### Architecture

Opacus uses **gradient hooks** to compute per-example gradients:

```python
# From opacus/privacy_engine.py
module = GradSampleModule(
    model,
    batch_first=True,
    loss_reduction="mean",
)

# Hooks are registered on each parameter
# During backward pass, hooks compute per-sample gradients
```

### How It Works

1. **Hook registration**: Each parameter gets a hook
   ```python
   # Simplified from opacus/grad_sample/grad_sample_module.py
   for name, param in model.named_parameters():
       param.register_hook(compute_grad_sample)
   ```

2. **Per-example gradient computation**: During `.backward()`
   ```python
   def compute_grad_sample(grad):
       # grad is aggregated gradient from loss.backward()
       # Reconstruct per-example gradients from activations
       per_sample_grads = reconstruct_per_example(grad, activations)
       param.grad_sample = per_sample_grads  # Attach to parameter
   ```

3. **Clipping in optimizer**: Access `param.grad_sample`
   ```python
   # From opacus/optimizers/optimizer.py
   for p in parameters:
       per_sample_norms = p.grad_sample.norm(2, dim=1)  # [B]
       clip_factor = (self.max_grad_norm / per_sample_norms).clamp(max=1.0)
       # Clip and sum
   ```

### Memory Management

Opacus **does NOT have automatic microbatching**. Users must:

1. **Manually accumulate gradients**:
   ```python
   optimizer.zero_grad()

   # Split batch manually
   for microbatch in split_batch(full_batch, microbatch_size):
       loss = model(microbatch)
       loss.backward()  # Accumulates grad_sample

   optimizer.step()  # Clip and apply
   ```

2. **Use `virtual_step()`** for accumulation:
   ```python
   # Opacus 1.0+ virtual batching
   for physical_batch in dataloader:
       optimizer.zero_grad()
       loss = model(physical_batch)
       loss.backward()

       if (step + 1) % virtual_batch_size == 0:
           optimizer.step()  # Clip and update
           optimizer.zero_grad()
       else:
           optimizer.virtual_step()  # Accumulate without updating
   ```

### Memory Issues in Opacus

**Problem**: `grad_sample` is stored for EVERY parameter
- **Memory**: `O(batch_size * num_params)` extra storage
- For large models (GPT-2: 124M params), this is prohibitive
- **Example**: Batch size 128, GPT-2 → ~60GB extra memory

**Partial solutions**:
- Smaller physical batch sizes (manual microbatching)
- Gradient checkpointing (recompute activations)
- Ghost clipping (approximate per-example norms, doesn't store `grad_sample`)

---

## Key Differences

### 1. Per-Example Gradient Computation

| Aspect | JAX-Privacy | Opacus |
|--------|-------------|--------|
| **Method** | `jax.vmap` (functional) | Gradient hooks (imperative) |
| **When** | Explicit call to `vmap` | Automatic during `.backward()` |
| **Storage** | Transient (in loop) | Persistent (`param.grad_sample`) |
| **Memory** | Microbatch-controlled | Full batch (unless manual) |

### 2. Microbatching

| Aspect | JAX-Privacy | Opacus |
|--------|-------------|--------|
| **Built-in** | ✅ `micro_vmap` | ❌ Manual accumulation |
| **API** | `microbatch_size` parameter | User loops |
| **Accumulation** | Automatic (`Accumulator`) | Manual (`virtual_step()`) |

### 3. Memory Management

| Aspect | JAX-Privacy | Opacus |
|--------|-------------|--------|
| **Automatic** | ❌ No profiling | ❌ No profiling |
| **Strategy** | Microbatching via `micro_vmap` | Manual batch splitting |
| **Memory overhead** | Microbatch-sized | Full batch-sized |

### 4. Flexibility

| Aspect | JAX-Privacy | Opacus |
|--------|-------------|--------|
| **Composability** | ✅ Functional, chains with Optax | ⚠️ Stateful optimizer |
| **Distributed** | ✅ SPMD via JAX | ✅ DDP support |
| **Custom layers** | ✅ Works with any JAX function | ⚠️ Needs hook implementation |

---

## What We Should Implement

### Option A: JAX-Privacy Style (Functional Microbatching)

**Pros**:
- Clean functional API
- Built-in memory efficiency
- Composable

**Implementation**:
```python
def clipped_grad(
    loss_fn: Callable,
    l2_clip_norm: float,
    microbatch_size: int | None = None,  # Memory control
) -> Callable:
    """Fixed clipping with optional microbatching."""

    def clipped_grad_fn(params, batch_x, batch_y):
        if microbatch_size is None:
            # No microbatching - process full batch
            return _clip_full_batch(params, batch_x, batch_y)
        else:
            # Microbatching - loop over chunks
            return _clip_microbatched(params, batch_x, batch_y, microbatch_size)

    def _clip_full_batch(params, batch_x, batch_y):
        # Single vmap call
        grads_per_example = torch.func.vmap(
            lambda x, y: torch.func.grad(loss_fn)(params, x, y)
        )(batch_x, batch_y)

        # Clip and sum
        return clip_and_sum(grads_per_example, l2_clip_norm)

    def _clip_microbatched(params, batch_x, batch_y, microbatch_size):
        # Split batch
        num_microbatches = batch_x.shape[0] // microbatch_size
        batch_x_split = batch_x.reshape(num_microbatches, microbatch_size, *batch_x.shape[1:])
        batch_y_split = batch_y.reshape(num_microbatches, microbatch_size, *batch_y.shape[1:])

        # Accumulator for summed gradients
        total_grad = None

        for i in range(num_microbatches):
            # vmap over microbatch
            grads_per_example = torch.func.vmap(
                lambda x, y: torch.func.grad(loss_fn)(params, x, y)
            )(batch_x_split[i], batch_y_split[i])

            # Clip and sum microbatch
            microbatch_grad = clip_and_sum(grads_per_example, l2_clip_norm)

            # Accumulate
            if total_grad is None:
                total_grad = microbatch_grad
            else:
                total_grad = tree_map(lambda a, b: a + b, total_grad, microbatch_grad)

        return total_grad

    return clipped_grad_fn
```

**Usage**:
```python
# Memory-efficient with microbatching
clipped_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0, microbatch_size=64)

for batch in data:  # batch_size = 1024
    # Internally: 1024 / 64 = 16 microbatches processed sequentially
    # Memory: O(64 * num_params) instead of O(1024 * num_params)
    grad = clipped_grad_fn(params, batch_x, batch_y)

    # Add noise and optimize
    noisy_grad = add_noise(grad, ...)
    params = optimizer_step(params, noisy_grad)
```

### Option B: Opacus Style (Manual Accumulation)

**Cons**:
- More boilerplate for users
- Easy to get wrong
- Doesn't match our functional API

**Not recommended** - violates our design principles.

---

## Recommendation

### Implement JAX-Privacy style microbatching:

1. **Add `microbatch_size` parameter to `clipped_grad()`**
   - `None` → full batch (default)
   - `int` → split into microbatches

2. **Implement sequential accumulation**:
   - Loop over microbatches
   - `vmap` over each microbatch
   - Accumulate sums

3. **For adaptive clipping**:
   - Accumulate `unclipped_count` and `total_count` across microbatches
   - Update `clip_norm` once per full batch (not per microbatch!)

4. **No memory manager needed** (for now):
   - Users set `microbatch_size` manually
   - Document memory vs batch size tradeoffs
   - Future: Add `estimate_microbatch_size()` helper

### Why this approach?

- ✅ Matches JAX-Privacy functional design
- ✅ Built-in memory efficiency
- ✅ No hooks or stateful magic
- ✅ Works naturally with `torch.func.vmap`
- ✅ Simple to understand and debug

---

## Summary

**Your Question**: Does JAX-Privacy have microbatching/memory manager like Opacus?

**Answer**:
- **Microbatching**: ✅ Yes - via Optax `micro_vmap` (functional, built-in)
- **Memory Manager**: ❌ No - users set `microbatch_size` manually

**Key differences**:
| JAX-Privacy | Opacus |
|-------------|--------|
| Functional `micro_vmap` | Manual loop + `virtual_step()` |
| Automatic accumulation | Manual accumulation |
| No persistent storage | Stores `param.grad_sample` |
| Cleaner API | More boilerplate |

**Our recommendation**: Implement JAX-Privacy style with `microbatch_size` parameter in `clipped_grad()`.
