# Memory Profiling

DP training with per-example gradients requires significantly more memory
than standard training. `vmap` materializes one gradient per example, so
peak memory scales as `microbatch_size * model_parameters`. Opaque provides
profiling tools to measure memory usage and automatically tune the
microbatch size.

## MemoryProfiler

`MemoryProfiler` is a context manager that tracks memory at key points in
your training loop.

```python
from opaque.profiling import MemoryProfiler
from opaque import clipped_grad

grad_fn, state = clipped_grad(loss_fn, l2_clip_norm=1.0, microbatch_size=16)

profiler = MemoryProfiler()
with profiler:
    grads, state = grad_fn(params, batch, targets, state=state)
    profiler.mark("after_grad")

    noisy_grads, noise_state = noise_fn(grads, noise_state)
    profiler.mark("after_noise")

print(profiler.report())
```

Output:

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
end                                  1.45            -0.48
============================================================
```

### Detailed CUDA statistics

Pass `detailed=True` for additional CUDA memory allocator information:

```python
profiler = MemoryProfiler(device="cuda")
with profiler:
    grads, state = grad_fn(params, batch, targets, state=state)
    profiler.mark("after_grad")

print(profiler.report(detailed=True))
```

The detailed report includes:

- **Reserved** -- memory cached by PyTorch's allocator (always >= allocated).
- **Efficiency** -- allocated / reserved ratio. Below 80% suggests
  fragmentation.
- **Alloc Retries** -- failed allocations that needed retry, indicating
  fragmentation.
- **OOM Count** -- out-of-memory errors encountered.

### Component tracking

Pass `track_components=True` to `mark()` for a breakdown into parameters,
gradients, activations, and other allocations:

```python
profiler.mark("after_forward", track_components=True)
```

Component tracking uses `gc.get_objects()` and is slow (100-500ms). Use it
sparingly at critical checkpoints. It does not work inside `clipped_grad`
because vmap creates and destroys tensors that are not visible to Python's
garbage collector.

## One-shot profiling

`profile_memory` profiles a single training step without manual
instrumentation:

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

## Auto-tuning microbatch size

`find_max_microbatch_size` finds the largest microbatch that fits in GPU
memory via binary search:

```python
from opaque.profiling import find_max_microbatch_size

optimal = find_max_microbatch_size(
    model=model,
    sample_batch=(data, targets),
    batch_size=128,
    loss_fn=loss_fn,
    l2_clip_norm=1.0,
    safety_margin=0.9,  # use 90% of available memory
)

grad_fn, state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_argnums=(1, 2),
    microbatch_size=optimal,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | The model to profile |
| `sample_batch` | `tuple` | Representative input batch |
| `batch_size` | `int` | Target training batch size |
| `loss_fn` | `Callable` | Per-example loss function |
| `l2_clip_norm` | `float` | Clip norm for gradient clipping |
| `safety_margin` | `float` | Fraction of memory to use (default: 0.9) |

## Device support

| Device | Basic profiling | Detailed stats | Component tracking |
|--------|----------------|----------------|-------------------|
| CUDA | Full | Full | Supported |
| MPS | Full | Limited | Supported |
| CPU | Limited | Not available | Supported |

## Troubleshooting

**Out of memory:** Use `find_max_microbatch_size` to automatically select
the largest feasible microbatch size. If the model itself does not fit, use
LoRA or another parameter-efficient method to reduce the trainable
parameter count.

**Low efficiency (<80%):** Memory fragmentation. Call
`torch.cuda.empty_cache()` between steps, or reduce `microbatch_size`.

**Memory grows over time:** Profile across iterations to identify whether
peak memory is increasing. Check for tensors that are accumulating outside
the training loop (e.g., appending to a list without detaching).

## API reference

See the `opaque.profiling` module for complete function signatures.
