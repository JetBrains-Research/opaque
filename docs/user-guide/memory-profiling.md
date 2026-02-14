# Memory Profiling

Opaque provides comprehensive memory profiling tools to help you understand and optimize memory usage during differentially private training. The `MemoryProfiler` context manager tracks memory throughout your training loop with detailed breakdowns.

## Why Profile Memory?

DP training with per-example gradients requires significantly more memory than standard training:

- **Per-example gradients**: Computing gradients for each sample individually
- **Gradient accumulation**: Storing gradients before clipping
- **Intermediate activations**: Forward pass results needed for backward pass

Understanding where memory is allocated helps you:

- Choose optimal `microbatch_size` values
- Identify memory bottlenecks
- Detect memory leaks or inefficiencies
- Debug out-of-memory errors

## Basic Usage

### Simple Memory Timeline

Track memory at key points in your training loop:

```python
from opaque.profiling import MemoryProfiler
from opaque import clipped_grad

# Setup your model and data
grad_fn, state = clipped_grad(loss_fn, l2_clip_norm=1.0, microbatch_size=16)

# Profile a training step
profiler = MemoryProfiler()
with profiler:
    # Compute gradients
    grads, aux = grad_fn(params, batch, targets, state=state)
    profiler.mark("after_grad")

    # Add noise
    noisy_grads = noise_fn(grads)
    profiler.mark("after_noise")

    # Update parameters
    params = optimizer.step(params, noisy_grads)
    profiler.mark("after_optimizer")

# View results
print(profiler.report())
```

**Output:**
```
============================================================
Memory Profile Report (CUDA)
============================================================
Peak Memory:          2.45 GB
Total Available:     23.50 GB
Peak Utilization:    10.4%

Timeline:
------------------------------------------------------------
Label                         Memory (GB)      Delta (GB)
------------------------------------------------------------
start                                0.15            +0.00
after_grad                           1.92            +1.77
after_noise                          1.93            +0.01
after_optimizer                      1.45            -0.48
end                                  1.45            +0.00
============================================================
```

### Key Metrics

- **Peak Memory**: Maximum memory used during profiling
- **Total Available**: Total device memory
- **Peak Utilization**: Percentage of memory used at peak
- **Delta**: Memory change since previous checkpoint

## Advanced Features

### Detailed CUDA Statistics

On CUDA devices, get additional insights about memory allocation:

```python
profiler = MemoryProfiler(device="cuda")
with profiler:
    grads, aux = grad_fn(params, batch, targets, state=state)
    profiler.mark("after_grad")

# Enable detailed report
print(profiler.report(detailed=True))
```

**Output:**
```
===============================================================================================
Memory Profile Report (CUDA)
===============================================================================================
Peak Memory:          2.45 GB
Total Available:     23.50 GB
Peak Utilization:    10.4%

CUDA Memory Stats:
  Reserved:             2.60 GB (memory cached by PyTorch)
  Alloc Retries:           0 (fragmentation indicator)
  OOM Count:               0 (out-of-memory errors)

Timeline:
-----------------------------------------------------------------------------------------------
Label                   Allocated      Delta     Reserved Efficiency
-----------------------------------------------------------------------------------------------
start                      0.15 GB +    0.00 GB       0.20 GB      75.0%
after_grad                 1.92 GB +    1.77 GB       2.10 GB      91.4%
after_optimizer            1.45 GB -    0.47 GB       2.10 GB      69.0%
===============================================================================================
```

**Understanding CUDA Metrics:**

- **Reserved**: Memory allocated by PyTorch's caching allocator (always ≥ Allocated)
- **Efficiency**: Allocated/Reserved ratio (100% is perfect, lower indicates fragmentation)
- **Alloc Retries**: Failed allocations that needed retry (indicates fragmentation)
- **OOM Count**: Number of out-of-memory errors encountered

Low efficiency (<80%) suggests memory fragmentation. Consider:
- Calling `torch.cuda.empty_cache()` periodically
- Using smaller `microbatch_size`
- Reducing model size or batch size

### Component-Level Tracking

Understand exactly where memory is allocated in **standard PyTorch code**:

```python
# Component tracking works for regular PyTorch operations
model = YourModel()
profiler = MemoryProfiler()

with profiler:
    # Track components at key points
    profiler.mark("before_forward", track_components=True)

    output = model(data)
    profiler.mark("after_forward", track_components=True)

    loss = criterion(output, targets)
    loss.backward()
    profiler.mark("after_backward", track_components=True)

print(profiler.report())
```

**Output:**
```
============================================================
Memory Profile Report (CUDA)
============================================================
Peak Memory:          3.20 GB
Total Available:     23.50 GB
Peak Utilization:    13.6%

Timeline:
------------------------------------------------------------
Label                         Memory (GB)      Delta (GB)
------------------------------------------------------------
start                                0.15            +0.00
before_forward                       0.15            +0.00
after_forward                        1.80            +1.65
after_backward                       3.20            +1.40
end                                  3.20            +0.00
============================================================

Component Breakdown (GB):
------------------------------------------------------------
Label                    Params      Grads  Activations      Other
------------------------------------------------------------
start                                              (not tracked)
before_forward            0.120      0.000        0.000      0.030
after_forward             0.120      0.000        1.500      0.180
after_backward            0.120      0.120        1.800      1.160
end                                                (not tracked)
============================================================
```

**Component Categories:**

- **Parameters**: Model weights (should remain constant)
- **Gradients**: Per-parameter gradients (appear after backward pass)
- **Activations**: Intermediate tensors from forward pass
- **Other**: Optimizer state, buffers, temporary allocations

**Important Limitations:**

1. **Component tracking does NOT work inside `clipped_grad()`**: The per-example gradient computation happens inside `torch.func.vmap`, which creates and destroys tensors that never become visible to Python's garbage collector. You'll only see memory state before and after `clipped_grad` executes.

2. **Use peak memory tracking instead**: For profiling DP training with `clipped_grad`, rely on:
   - Peak memory metrics (captured automatically)
   - Detailed CUDA stats (`detailed=True`)
   - Timeline profiling (memory before/after operations)

3. **When component tracking IS useful**:
   - Regular PyTorch training loops (non-DP)
   - Model loading and initialization
   - Optimizer state analysis
   - Debugging memory leaks outside of `clipped_grad`

**Performance Warning**: Component tracking uses `gc.get_objects()` which is slow (100-500ms). Use sparingly, only at critical checkpoints where you need detailed analysis.

### Profiling DP Training (What Actually Works)

For DP training with `clipped_grad`, use peak memory and detailed CUDA stats:

```python
from opaque.profiling import MemoryProfiler
from opaque import clipped_grad

# Setup
grad_fn, state = clipped_grad(loss_fn, l2_clip_norm=1.0, microbatch_size=16)

# Profile DP training step
profiler = MemoryProfiler(device="cuda")
with profiler:
    # Peak memory is captured during this call
    grads, aux = grad_fn(params, data, targets, state=state)
    profiler.mark("after_clipped_grad")

    # Noise and optimizer
    noisy_grads = noise_fn(grads)
    profiler.mark("after_noise")

    params = optimizer.step(params, noisy_grads)
    profiler.mark("after_optimizer")

# Use detailed report to see CUDA stats
print(profiler.report(detailed=True))
```

**Output shows what matters:**
```
Peak Memory:          2.45 GB  ← Maximum during clipped_grad
Reserved:             2.60 GB  ← PyTorch cache overhead
Efficiency:          94.7%     ← Good (>90%)
Alloc Retries:           0     ← No fragmentation
```

**Key insights:**
- Peak memory tells you if you'll hit OOM
- Reserved vs Allocated shows caching overhead
- Low efficiency (<80%) indicates fragmentation → reduce `microbatch_size`
- Alloc retries > 0 means memory fragmentation → call `torch.cuda.empty_cache()`

## Automated Profiling

### One-Shot Profiling

Profile a single training step without manual instrumentation:

```python
from opaque.profiling import profile_memory

profile = profile_memory(
    model=model,
    sample_batch=(data, targets),
    loss_fn=loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=16,
)

print(profile)
```

**Output:**
```
Memory Profile (batch_size=32, microbatch_size=16)
  Device:           cuda
  Peak Memory:       1.85 GB
  Available:        23.50 GB
  Utilization:       7.9%
  Status:           ✓ OK
```

### Auto-Tuning Microbatch Size

Automatically find the largest `microbatch_size` that fits in memory:

```python
from opaque.profiling import find_max_microbatch_size

optimal_size = find_max_microbatch_size(
    model=model,
    sample_batch=(data, targets),
    batch_size=128,
    loss_fn=loss_fn,
    l2_clip_norm=1.0,
    safety_margin=0.9,  # Keep 10% memory free
)

print(f"Optimal microbatch_size: {optimal_size}")

# Use in training
grad_fn, state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=optimal_size,
    batch_argnums=(1, 2),
)
```

**Parameters:**

- `batch_size`: Your target training batch size
- `safety_margin`: Fraction of memory to use (0.9 = use 90%, keep 10% free)
- `min_size`: Minimum microbatch size to try (default: 1)

## Best Practices

### 1. Profile Early in Development

Run profiling on a representative workload before scaling up:

```python
# Profile with small batch first
profiler = MemoryProfiler()
with profiler:
    grads, _ = grad_fn(params, small_batch, targets, state=state)
    profiler.mark("after_grad")

print(profiler.report(detailed=True))

# Identify issues before training on full dataset
if profiler.get_peak_memory() / profiler.get_total_memory() > 0.8:
    print("WARNING: Using >80% memory, consider reducing batch size")
```

### 2. Use Microbatching

If you hit OOM errors, enable microbatching:

```python
# Start with automatic tuning
optimal_size = find_max_microbatch_size(
    model, sample_batch, batch_size=128, loss_fn, l2_clip_norm=1.0
)

# Then use in training
grad_fn, state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=optimal_size,
    batch_argnums=(1, 2),
)
```

See [Known Limitations](../limitations.md) for memory optimization strategies.

### 3. Profile Periodically During Training

Memory usage can change over training (e.g., optimizer state grows):

```python
# Profile every N epochs
if epoch % 10 == 0:
    profiler = MemoryProfiler()
    with profiler:
        # Run training step
        grads, _ = grad_fn(params, batch, targets, state=state)
        profiler.mark("after_grad")

    report = profiler.report(detailed=True)
    # Log or save report
    with open(f"memory_profile_epoch_{epoch}.txt", "w") as f:
        f.write(report)
```

### 4. Focus on Peak Memory and CUDA Stats

For DP training, peak memory and CUDA efficiency are most actionable:

```python
profiler = MemoryProfiler(device="cuda")
with profiler:
    grads, _ = grad_fn(params, batch, targets, state=state)
    profiler.mark("after_grad")

# Check detailed stats
print(profiler.report(detailed=True))

# Look for:
# - Peak memory (will you OOM?)
# - Efficiency < 80% (fragmentation issue)
# - Alloc retries > 0 (need empty_cache)
```

**Note:** Component tracking (`track_components=True`) is only useful for debugging non-DP code paths like model initialization, optimizer state, or data loading.

## Device Support

| Device | Basic Profiling | Detailed Stats | Component Tracking |
|--------|----------------|----------------|-------------------|
| CUDA   | ✅ Full support | ✅ Full support | ✅ Supported |
| MPS (Apple Silicon) | ✅ Full support | ⚠️ Limited | ✅ Supported |
| CPU    | ⚠️ Limited | ❌ Not available | ✅ Supported |

**Notes:**
- **CUDA**: Full feature support with accurate measurements
- **MPS**: Memory tracking supported, but no `max_memory_allocated` API (uses current)
- **CPU**: Basic tracking only, measurements may be approximate

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. **Check current memory usage:**
   ```python
   profile = profile_memory(model, sample_batch, loss_fn, l2_clip_norm=1.0)
   print(f"Current usage: {profile.utilization() * 100:.1f}%")
   ```

2. **Enable microbatching:**
   ```python
   optimal_mb = find_max_microbatch_size(
       model, sample_batch, batch_size, loss_fn, l2_clip_norm=1.0
   )
   ```

3. **Profile with detailed stats:**
   ```python
   profiler = MemoryProfiler()
   with profiler:
       # ... training step ...
   print(profiler.report(detailed=True))
   # Check for low efficiency or high alloc retries
   ```

### Memory Leaks

If memory grows over time:

1. **Profile across iterations:**
   ```python
   for i in range(10):
       profiler = MemoryProfiler()
       with profiler:
           grads, _ = grad_fn(params, batch, targets, state=state)
           profiler.mark(f"iteration_{i}")
       print(f"Iteration {i}: {profiler.get_peak_memory():.2f} GB")
   ```

2. **Use component tracking:**
   ```python
   # Check which component is growing
   profiler.mark("iter_5", track_components=True)
   # Look for unexpected growth in "activations" or "other"
   ```

3. **Clear caches:**
   ```python
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

### Slow Profiling

Component tracking is slow due to `gc.get_objects()`. Solutions:

- Only use `track_components=True` at 1-2 key points
- Use basic profiling for most checkpoints
- Profile on smaller batch sizes during development

## API Reference

See [API Documentation](../api/profiling.md) for complete API details.

### Key Functions

- `MemoryProfiler()`: Context manager for tracking memory
- `profile_memory()`: One-shot profiling of a training step
- `find_max_microbatch_size()`: Auto-tune microbatch size
- `MemoryTracker`: Low-level device memory tracking

## See Also

- [Known Limitations](../limitations.md) - Memory optimization strategies
- [Sampling & Microbatching](sampling.md) - Microbatching configuration
- [Clipping Guide](clipping.md) - Gradient clipping basics
