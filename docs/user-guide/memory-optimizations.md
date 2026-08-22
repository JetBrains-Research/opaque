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
parameters (~0.1% of the model) require per-example gradients. Use
`make_functional(model, partition_trainable=True)` to expose only the
trainable subset to `vmap(grad(...))` — see [Utilities reference —
`make_functional`](../reference/utilities.md#trainable-frozen-partition-for-peft-and-lora).

## Microbatching

Microbatching reduces $M$ by processing the batch in chunks. With
`microbatch_size=16` and `batch_size=256`, vmap runs 16 forward-backward
passes of 16 examples each, accumulating the clipped gradients. Memory
drops from $256 \cdot P$ to $16 \cdot P$ for the gradient term, at the
cost of 16× more sequential computation.

```python
grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    batch_argnums=(1, 2),
    microbatch_size=16,  # process 16 examples at a time
)
```

### Microbatch size vs throughput

Smaller microbatches use less memory but require more passes. Measure the
trade-off on your workload.

### Tuning workflow

Use a short manual sweep with `step_perf`:

```python
from opaque.dpsgd.clipping import clipped_grad
from opaque.profiling import reset_peak_memory, step_perf

def try_microbatch(candidate_mb: int) -> float:
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        clipping_norm=1.0,
        batch_argnums=(1, 2),
        microbatch_size=candidate_mb,
    )

    reset_peak_memory(device)
    with step_perf(device, batch_size=len(batch_x)) as perf:
        _grads, _aux = grad_fn(params, batch_x, batch_y, state=clip_state)

    return perf.result.memory_peak_gb
```

1. Start with `microbatch_size = batch_size`.
2. Halve until OOM stops.
3. Run a 20-50 step smoke loop.
4. Select the largest stable value.

## Gradient checkpointing

`opaque.execution.checkpoint(fn)` wraps a callable in backend-native
activation checkpointing. The wrapper is lazy: it resolves to the backend
of its first call and caches one transformed callable per backend.

```python
from opaque.execution import checkpoint
from opaque.autodiff import grad_and_value, vmap

checkpointed_block = checkpoint(block)
grads, values = vmap(grad_and_value(checkpointed_block))(batch_x)
```

The portable first version of each transform accepts only `fn`;
backend-specific compile flags, checkpoint policies, buffer donation, or
static-argument lists are intentionally not part of the portable contract.

**Torch** maps to `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`,
which uses saved-tensor hooks that `torch.func` rejects on its own. Call
`opaque.torch.checkpoint.apply_checkpoint_patch()` once before composing
`checkpoint` or `optimize_saved_activations` under `grad_and_value` / `vmap` /
`clipped_grad`; without it the first call raises `RuntimeError:
torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks`.
It ships with `opaque-torch`, so this needs no `opaque-transformers` install;
`opaque.torch.apply_runtime_patches()` selects it as the provider's one runtime
concern, and `opaque.transformers.patches.apply_runtime_patches()` forwards
there before applying the Hugging Face patches, so either call covers it.
`DPTrainer` applies the runtime patches during construction, so trainer-driven
flows are already covered. Plain eager
checkpointing, outside any functional transform, needs no patch.

**With Hugging Face models:**

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model.gradient_checkpointing_enable()
# Then proceed with make_functional, clipped_grad, etc.
```

`model.gradient_checkpointing_enable()` remains a Torch integration
convenience; Opaque still forces the non-reentrant path under the hood.

**Limitations:**

- Only safe for first-order differentiation (`grad`, `vjp`, `jacrev`).
  Higher-order transforms (`hessian`, `jacrev(jacrev)`) are not supported.
  Opaque only uses first-order differentiation.
- `compile` around a checkpointed transform runs and matches the eager result,
  but Dynamo warns while tracing the checkpoint boundary. Measure before
  assuming a speedup.
- Build a direct `vmap(grad_and_value(...))` whose result you only read with
  `grad_and_value(..., values_only=True)`: it tells the provider not to keep
  the result differentiable, which is the portable spelling of a
  `torch.no_grad()` region. `clipped_grad` already passes it.
- Opt out at the API layer (there are no env-var kill switches): pass
  `vmap_checkpointing=False` to `apply_runtime_patches(...)` or
  `apply_model_patches(...)`, or
  `performance_kernels_config={"vmap_checkpointing": False}` to
  `TrainingArguments`.

**Memory comparison:**

| Technique | Memory | Compute | Notes |
|---|---|---|---|
| No optimization | O(batch_size) | 1x | |
| Gradient checkpointing | Workload-dependent | Recomputation overhead | Best for long chains; measure on target model |
| Microbatching (size m) | O(m) gradient term | More sequential passes | Measure on target device |

### Saved-activation optimization

`opaque.execution.optimize_saved_activations(fn)` reduces separate
accelerator-memory pressure by offloading activations saved for backward.

```python
from opaque.execution import optimize_saved_activations

optimized_block = optimize_saved_activations(block)
```

**Torch** enters `torch.autograd.graph.save_on_cpu(pin_memory=True)` for
each call, moving tensors saved for backward to pinned CPU memory during
forward and reloading them during backward. Combined with gradient
checkpointing, this offloads checkpoint inputs while checkpoint handles
intermediates separately.

### Recommended transform order

Apply execution transforms in this order:

```
checkpoint / optimize_saved_activations → grad / vmap / clip → compile
```

1. **Select a region** and wrap it with `checkpoint` or
   `optimize_saved_activations`.
2. **Build autodiff and clipping** around that region with `grad`,
   `vmap`, and `clipped_grad`.
3. **Compile the outermost stable transform** with `compile` so the
   compiler sees fixed shapes and dtypes.

Install the checkpoint patch first. The engine binds an execution transform
lazily, so the provider only reaches its own implementation on the transform's
first invocation — by then a `torch.func` interpreter is already on the stack
and has made its saved-tensor-hook check:

```python
from opaque.torch.checkpoint import apply_checkpoint_patch
from opaque.execution import checkpoint, compile
from opaque.dpsgd.clipping import clipped_grad

apply_checkpoint_patch()

core = checkpoint(selected_region)
grad_fn, _ = clipped_grad(
    lambda p, x, y: loss_fn(core(p, x), y),
    clipping_norm=1.0,
    argnums=0,
    batch_argnums=(1, 2),
)
compiled_step = compile(grad_fn)
```

Compilation is the outermost layer because it specializes on the stable
shapes and dtypes produced by the already-differentiated transform.

## Fused Triton kernels

Opaque includes fused Triton kernels that replace standard PyTorch operations
in supported models, reducing memory and improving throughput without changing
training semantics. These are enabled by `apply_model_patches(model)` after
runtime patching has been set up.

The kernels reduce memory by eliminating intermediate tensors (fused forward
passes) and recomputing activations in backward instead of saving them. Each
kernel also implements native vmap support, so `vmap(grad())` works without
fallbacks.

See [Model Patches — Triton kernels](huggingface/model-patches.md#triton-kernels)
for per-operation details and per-model support.

### Kernel benchmarks

Kernel performance is workload-dependent; profile patched and unpatched
paths on your workload.

### Fused linear cross-entropy

Computes the loss directly from hidden states and the `lm_head` weight matrix,
never materializing the full `(batch*seq, vocab)` logits tensor. Enable it by
passing `fused_linear_cross_entropy=True` to
`apply_model_patches(model, ...)`.

The fused path returns `logits=None` from `XForCausalLM.forward`, which is
incompatible with callers that read `outputs.logits` — `compute_metrics`,
`preprocess_logits_for_metrics`, and generation eval. Enable the patch when
loss is the only consumer of the forward output;
`examples/train_dpsgd.py` and `examples/train_dpftrl.py` do.

## Profiling

### step_perf + perf_state

Use `step_perf` to measure individual training steps and `perf_state` to
accumulate throughput statistics across a run. The state is threaded like
every other Opaque state object: `add` returns a new one.

```python
from opaque.profiling import perf_state, print_memory, step_perf

print_memory(device, "start")
state = perf_state(device)

for batch in dataloader:
    with step_perf(device, batch_size=len(batch["input_ids"])) as perf:
        train_step(batch)
        perf.mark("clip")

    state = state.add(perf.perf)
    # e.g., perf.perf.step_time_sec, perf.perf.memory_peak_gb

print_memory(device, "end")
print(state.to_dict(prefix="train/"))
```

For several named stages (`train` / `eval` / …), `perf_tracker(device)`
accumulates each one for you.

For one-off memory snapshots, use `print_memory(device, label)`
or `get_memory_stats(device)`.

### Device support

| Device | Basic profiling | Detailed stats | Component tracking |
|--------|----------------|----------------|-------------------|
| CUDA | Full | Full | Supported |
| MPS | Full | Partial | Supported |
| CPU | Limited | Not available | Supported |

On MPS, `get_memory_stats` reports allocated, reserved, and total memory
(via `torch.mps.current_allocated_memory`, `driver_allocated_memory`, and
`recommended_max_memory`). PyTorch's MPS backend exposes no allocated-peak
counter, so **`peak_gb` is the driver's reserved high-water mark** — a
monotonic measurement that *captures transients* (a tensor freed before the
read still counts), which is the useful number for "how much did this step
need". This is a precise measurement, so `MemoryStats.exact_peak` is `True`;
it differs from CUDA's `peak_gb` only in *quantity* — reserved high-water (it
equals `reserved_gb` and upper-bounds the allocated peak) rather than the
allocated peak — not in precision.

Because there is no cheap per-step peak reset on MPS, `step_perf` does not
reset it each step (the only reset is `torch.mps.empty_cache`, too costly to
run every step). Instead, MPS `peak_gb` accumulates as the run's reserved
high-water and `max_peak_memory_gb` gives the run peak. For a clean
per-config measurement (e.g. benchmarking kernels), call
`reset_peak_memory(device)` before the measured region.

Sub-step `.mark()` calls are device-synchronized, so their timings reflect
real GPU execution — without the sync, an accelerator mark would record only
async kernel-launch time (microseconds for hundreds of ms of work).

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
`step_perf`. If the model itself does not fit, use LoRA or another
parameter-efficient method to reduce the trainable parameter count.

**Low efficiency (<80%):** Memory fragmentation. Call
`torch.cuda.empty_cache()` between steps, or reduce `microbatch_size`.

**Memory grows over time:** Profile across iterations to identify whether
peak memory is increasing. Check for tensors that are accumulating outside
the training loop (e.g., appending to a list without detaching).

**OOM with fused linear CE not enabled:** Fused linear CE is opt-in
(`apply_model_patches(model, fused_linear_cross_entropy=True)`). Without
it, the full `(batch*seq, vocab)` logits tensor is materialized — for
128K vocab models, that is ~2 GB per sample. Enable the flag (only if
nothing else reads logits) or reduce batch size.

## API reference

See the `opaque.profiling` module for complete function signatures.
