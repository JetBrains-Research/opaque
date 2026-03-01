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

## TrainingProfiler

Use `TrainingProfiler` to track checkpoints and step-level metrics in your
training loop.

```python
from opaque.profiling import TrainingProfiler

profiler = TrainingProfiler(device)
profiler.mark("start")

for batch in dataloader:
    with profiler.step(batch_size=len(batch["input_ids"])):
        train_step(batch)

    # Current step + memory metrics for logging
    metrics = profiler.current_metrics()
    # e.g., metrics["step_time_sec"], metrics["memory_peak_gb"]

profiler.mark("end")
print(profiler.final_summary())
```

For one-off checkpoints without a profiler, use `print_memory(device, label)`
or `get_memory_stats(device)`.

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

## Microbatch tuning workflow

Opaque currently does not expose an automatic microbatch search helper.
Use a short manual sweep with `TrainingProfiler`:

```python
from opaque import clipped_grad
from opaque.profiling import TrainingProfiler, reset_peak_memory

def try_microbatch(candidate_mb: int) -> float:
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        l2_clip_norm=1.0,
        batch_argnums=(1, 2),
        microbatch_size=candidate_mb,
    )

    reset_peak_memory(device)
    profiler = TrainingProfiler(device)
    with profiler.step(batch_size=len(batch_x)):
        _grads, _aux = grad_fn(params, batch_x, batch_y, state=clip_state)

    return profiler.current_metrics()["memory_peak_gb"]
```

Recommended process:

1. Start with `microbatch_size = batch_size`.
2. Halve until OOM stops.
3. Run a 20-50 step smoke loop.
4. Select the largest stable value.

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
Start from your single-device stable value and reduce by 10-20% for DDP.

## Common profiling patterns

**Profile → tune → train:** Profile your training step, choose a stable
`microbatch_size`, then train with that value:

```python
profiler = TrainingProfiler(device)
for _ in range(20):
    with profiler.step(batch_size=len(batch_x)):
        train_step()
print(profiler.final_summary())
```

**Compare configurations:** Profile multiple LoRA ranks or model sizes to
find the best memory-accuracy trade-off:

```python
for rank in [4, 8, 16]:
    model = get_peft_model(base_model, LoraConfig(r=rank))
    # Run short profiling loop and pick largest stable microbatch manually
    print(f"LoRA r={rank}: compare peak memory with TrainingProfiler")
```

## Troubleshooting

**Out of memory:** Reduce `microbatch_size` and re-profile with
`TrainingProfiler`. If the model itself does not fit, use
LoRA or another parameter-efficient method to reduce the trainable
parameter count.

**Low efficiency (<80%):** Memory fragmentation. Call
`torch.cuda.empty_cache()` between steps, or reduce `microbatch_size`.

**Memory grows over time:** Profile across iterations to identify whether
peak memory is increasing. Check for tensors that are accumulating outside
the training loop (e.g., appending to a list without detaching).

## API reference

See the `opaque.profiling` module for complete function signatures.
