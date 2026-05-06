# Memory Optimizations

DP-SGD training with `vmap(grad())` is memory-intensive: per-example gradients
require materializing one gradient copy per sample. This page covers all
available techniques for reducing memory usage.

## Understanding vmap memory

In standard training, a forward-backward pass produces one gradient tensor
per parameter — the batch dimension is implicit. With `vmap`, each example
in the batch gets its own gradient copy, so memory scales as:

$$\text{peak} \approx P + M \cdot P + A$$

where $P$ is model parameters, $M$ is the microbatch size (or full batch
size if no microbatching), and $A$ is activation memory. The $M \cdot P$
term dominates for large models.

| Model size | Full batch (256) | Microbatch 16 | Microbatch 1 |
|------------|-----------------|---------------|--------------|
| 125M (GPT-2) | ~32 GB | ~2 GB | ~125 MB |
| 7B (LLaMA) | infeasible | ~112 GB | ~7 GB |
| 7B + LoRA r=8 | ~2.5 GB | ~160 MB | ~10 MB |

LoRA dramatically reduces gradient memory because only the adapter
parameters (~0.1% of model) require per-example gradients. See
[HuggingFace Compatibility — LoRA](huggingface.md#using-lora-with-dp-sgd)
for details.

## Microbatching

Microbatching reduces $M$ by processing the batch in chunks. With
`microbatch_size=16` and `batch_size=256`, vmap runs 16 forward-backward
passes of 16 examples each, accumulating the clipped gradients. Memory
drops from $256 \cdot P$ to $16 \cdot P$ for the gradient term, at the
cost of 16x more sequential computation.

```python
grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    batch_argnums=(1, 2),
    microbatch_size=16,  # process 16 examples at a time
)
```

### Microbatch size vs throughput

| Microbatch size | Memory | Passes (batch=256) | Relative speed |
|-----------------|--------|--------------------|----------------|
| 256 (no microbatch) | Highest | 1 | Fastest |
| 64 | 4x less | 4 | ~3.5x slower |
| 16 | 16x less | 16 | ~10x slower |
| 1 | Minimum | 256 | ~100x slower |

The relationship is not purely linear because GPU utilization drops for very
small microbatches. In practice, `microbatch_size >= 4` maintains reasonable
GPU utilization. Below that, the overhead of launching kernels dominates.

### Tuning workflow

Use a short manual sweep with `TrainingProfiler`:

```python
from opaque.clipping import clipped_grad
from opaque.profiling import reset_peak_memory
from opaque.profiling import StepTimer, TrainingProfiler

def try_microbatch(candidate_mb: int) -> float:
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        clipping_norm=1.0,
        batch_argnums=(1, 2),
        microbatch_size=candidate_mb,
    )

    reset_peak_memory(device)
    profiler = TrainingProfiler(device)
    timer = StepTimer(device, batch_size=len(batch_x))
    with timer:
        _grads, _aux = grad_fn(params, batch_x, batch_y, state=clip_state)
    profiler = profiler.add_step(timer)

    return profiler.current_metrics()["memory_peak_gb"]
```

1. Start with `microbatch_size = batch_size`.
2. Halve until OOM stops.
3. Run a 20-50 step smoke loop.
4. Select the largest stable value.

## Gradient checkpointing

PyTorch's `torch.utils.checkpoint.checkpoint` is supported under
`vmap(grad(...))`. Enable the runtime patch once with
`opaque.patches.apply_runtime_patches()`.

**With PyTorch directly** (non-reentrant checkpoint only):

```python
from opaque.patches import apply_runtime_patches
from torch.utils.checkpoint import checkpoint

apply_runtime_patches()

def my_model(x):
    h = checkpoint(block1, x, use_reentrant=False)
    h = checkpoint(block2, h, use_reentrant=False)
    return h.sum()

grads = vmap(grad(my_model))(batch_x)
```

**With HuggingFace models:**

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model.gradient_checkpointing_enable()
# Then proceed with make_functional, clipped_grad, etc.
```

Opaque automatically forces `use_reentrant=False` (the only path compatible
with functorch). No special kwargs needed.

**Memory comparison:**

| Technique | Memory | Compute | Notes |
|-----------|--------|---------|-------|
| No optimization | O(batch_size) | 1x | |
| Gradient checkpointing | O(sqrt(layers)) | ~2x | ~81% savings on Mellum-4b |
| Microbatching (size m) | O(m) | 1x | |

**Limitations:**

- Requires `use_reentrant=False` (the non-reentrant checkpoint path).
  The legacy reentrant path is not supported.
- Only safe for first-order differentiation (grad, vjp, jacrev). Not
  compatible with higher-order transforms (hessian, jacrev(jacrev)).
  Opaque only uses first-order differentiation, so this is not an issue
  in practice.
- Skip patches with: `OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES=all`

### CPU offloading of saved tensors

`torch.autograd.graph.save_on_cpu` moves tensors saved for backward to
pinned CPU memory during forward and reloads them during backward. When
combined with gradient checkpointing, it offloads the checkpoint inputs
(inter-layer hidden states); checkpoint handles intermediates separately.

```python
with torch.autograd.graph.save_on_cpu(pin_memory=True):
    grads, aux = grad_fn(params, batch)
```

## Fused Triton kernels

Opaque includes fused Triton kernels that replace standard PyTorch operations
in supported models, reducing memory and improving throughput without changing
training semantics. These are enabled by `apply_model_patches(model)` after
runtime patching has been set up.

The kernels reduce memory by eliminating intermediate tensors (fused forward
passes) and recomputing activations in backward instead of saving them. Each
kernel also implements native vmap support, so `vmap(grad())` works without
fallbacks.

See [HuggingFace Compatibility — Patched operations](huggingface.md#patched-operations)
for per-operation details and per-model support.

### Kernel benchmarks

Measured at Mellum-4b scale (batch=4, seq=1024, vocab=128256, LoRA r=16).
Values > 1.0 mean the kernel is faster (speed) or uses less memory (memory)
than the PyTorch baseline.

| Kernel | Forward | Backward | Memory | vmap(grad) speed | vmap(grad) memory |
|--------|---------|----------|--------|------------------|-------------------|
| SwiGLU | 0.69x | 0.83x | 1.20x | 1.19x | 2.10x |
| GeGLU Exact | 0.76x | 0.78x | 1.38x | 0.84x | 1.43x |
| GeGLU Approx | 0.81x | 0.72x | 1.38x | 0.77x | 1.43x |
| RoPE | 2.01x | 1.13x | 1.46x | 0.98x | 1.70x |
| CE (V=32K) | 1.56x | 1.33x | 1.67x | 2.63x | 2.00x |
| CE (V=128K) | 2.20x | 2.24x | 1.67x | 3.68x | 2.00x |

SwiGLU/GeGLU forward is slower than native PyTorch because
`autograd.Function.apply()` dispatch overhead dominates the trivially fast
element-wise operation. The real value is in the fused backward and vmap
memory savings.

### Fused linear cross-entropy

The most impactful optimization. Standard cross-entropy requires materializing
`logits = hidden_states @ lm_head.T` — for Mellum-4b with 128K vocab, this is
~2 GB per forward pass. Fused linear cross-entropy computes the loss directly
from hidden states, never materializing the full logits tensor.

| Metric | V=32K | V=128K |
|--------|-------|--------|
| Forward speedup | 8.73x | 9.46x |
| Forward memory | 2.85x | 3.19x |
| Backward speedup | 2.63x | 2.76x |
| Backward memory | 3.35x | 3.80x |
| vmap forward speedup | 8.86x | 8.88x |
| vmap forward memory | 12.10x | 22.67x |
| vmap(grad) speedup | 2.65x | 2.70x |
| vmap(grad) memory | 6.06x | 8.05x |

## Profiling

### TrainingProfiler

Use `TrainingProfiler` to track checkpoints and step-level metrics in your
training loop.

```python
from opaque.profiling import StepTimer, TrainingProfiler

profiler = TrainingProfiler(device)
profiler, _ = profiler.mark("start")

for batch in dataloader:
    timer = StepTimer(device, batch_size=len(batch["input_ids"]))
    with timer:
        train_step(batch)
    profiler = profiler.add_step(timer)

    metrics = profiler.current_metrics()
    # e.g., metrics["step_time_sec"], metrics["memory_peak_gb"]

profiler, _ = profiler.mark("end")
print(profiler.final_summary())
```

For one-off checkpoints without a profiler, use `print_memory(device, label)`
or `get_memory_stats(device)`.

### Device support

| Device | Basic profiling | Detailed stats | Component tracking |
|--------|----------------|----------------|-------------------|
| CUDA | Full | Full | Supported |
| MPS | Full | Limited | Supported |
| CPU | Limited | Not available | Supported |

### Distributed memory considerations

In DDP training, each rank holds the full model and computes per-example
gradients for its local batch. `AllReduce` temporarily doubles the gradient
memory while summing across ranks. Profile on a single GPU first to
establish the memory baseline, then account for the AllReduce overhead when
scaling.

If memory is tight, reduce `microbatch_size` to leave headroom for AllReduce.
Start from your single-device stable value and reduce by 10-20% for DDP.

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

**OOM with fused linear CE disabled:** Without fused linear CE, the full
`(batch*seq, vocab)` logits tensor is materialized. For 128K vocab models,
this uses ~2 GB per sample. Re-enable fused CE or reduce batch size.

## API reference

See the `opaque.profiling` module for complete function signatures.
