# Memory Profiling

DP training with per-example gradients requires significantly more memory
than standard training. `vmap` materializes one gradient per example, so
peak memory scales as `microbatch_size * model_parameters`. Opaque provides
profiling tools to measure memory usage and automatically tune the
microbatch size.

## Understanding vmap memory usage

In standard training, a forward-backward pass produces one gradient tensor
per parameter — the batch dimension is implicit. With `vmap`, each example
in the batch gets its own gradient copy, so memory scales as:

$$\text{peak} \approx P + M \cdot P + A$$

where $P$ is model parameters, $M$ is the microbatch size (or full batch
size if no microbatching), and $A$ is activation memory. The $M \cdot P$
term dominates for large models.

**Microbatching** reduces $M$ by processing the batch in chunks. With
`microbatch_size=16` and `batch_size=256`, vmap runs 16 forward-backward
passes of 16 examples each, accumulating the clipped gradients. Memory
drops from $256 \cdot P$ to $16 \cdot P$ for the gradient term, at the
cost of 16x more sequential computation.

| Model size | Full batch (256) | Microbatch 16 | Microbatch 1 |
|------------|-----------------|---------------|--------------|
| 125M (GPT-2) | ~32 GB | ~2 GB | ~125 MB |
| 7B (LLaMA) | infeasible | ~112 GB | ~7 GB |
| 7B + LoRA r=8 | ~2.5 GB | ~160 MB | ~10 MB |

LoRA dramatically reduces the gradient memory because only the adapter
parameters (~0.1% of model) require per-example gradients.

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

## Microbatch size vs throughput

Smaller microbatch sizes use less memory but require more sequential passes
through the model:

| Microbatch size | Memory | Passes (batch=256) | Relative speed |
|-----------------|--------|--------------------|----------------|
| 256 (no microbatch) | Highest | 1 | Fastest |
| 64 | 4x less | 4 | ~3.5x slower |
| 16 | 16x less | 16 | ~10x slower |
| 1 | Minimum | 256 | ~100x slower |

The relationship is not purely linear because GPU utilization drops for very
small microbatches. In practice, `microbatch_size >= 4` maintains reasonable
GPU utilization. Below that, the overhead of launching kernels dominates.

The `safety_margin` parameter in `find_max_microbatch_size` (default 0.9)
reserves 10% of GPU memory for PyTorch's CUDA allocator overhead,
fragmentation, and temporary tensors. Reduce it if you see OOM errors
during training that did not occur during profiling.

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

## Distributed memory considerations

In DDP training, each rank holds the full model and computes per-example
gradients for its local batch. `AllReduce` temporarily doubles the gradient
memory while summing across ranks. Profile on a single GPU first to
establish the memory baseline, then account for the AllReduce overhead when
scaling.

If memory is tight, reduce `microbatch_size` to leave headroom for AllReduce.
A typical pattern:

```python
single_gpu_max = find_max_microbatch_size(model, sample_batch, ...)
# Leave 20% headroom for AllReduce
distributed_microbatch = int(single_gpu_max * 0.8)
```

## Common profiling patterns

**Profile → auto-tune → train:** Profile the model once during setup, use
the result to configure training:

```python
from opaque.profiling import find_max_microbatch_size

optimal = find_max_microbatch_size(model, sample_batch, batch_size, loss_fn, l2_clip_norm=1.0)
grad_fn, state = clipped_grad(loss_fn, l2_clip_norm=1.0, microbatch_size=optimal)
```

**Compare configurations:** Profile multiple LoRA ranks or model sizes to
find the best memory-accuracy trade-off:

```python
for rank in [4, 8, 16]:
    model = get_peft_model(base_model, LoraConfig(r=rank))
    max_mb = find_max_microbatch_size(model, sample_batch, batch_size, loss_fn)
    print(f"LoRA r={rank}: max microbatch = {max_mb}")
```

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
