# Known Limitations

## Gradient checkpointing incompatibility

Gradient checkpointing (`torch.utils.checkpoint.checkpoint`) is
incompatible with `torch.func.vmap`, which Opaque uses for per-example
gradient computation.

**Error:**

```
RuntimeError: You tried to vmap over _NoopSaveInputs, but it does not have
vmap support.
```

**When this happens:**

- Explicit use of `torch.utils.checkpoint.checkpoint` in a model's
  `forward` method.
- Calling `model.gradient_checkpointing_enable()` on a HuggingFace
  Transformers model.
- Third-party models that enable checkpointing by default.

**Solution:** Use microbatching instead. The `microbatch_size` parameter on
`clipped_grad` achieves similar memory savings by processing the batch in
chunks:

```python
from opaque import clipped_grad

grad_fn, state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_argnums=(1, 2),
    microbatch_size=16,
)
```

Use `TrainingProfiler` from `opaque.profiling` to test a few microbatch
values and keep the largest stable one:

```python
from opaque.profiling import TrainingProfiler, reset_peak_memory

profiler = TrainingProfiler(device)
for candidate in [32, 16, 8, 4, 2, 1]:
  grad_fn, state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_argnums=(1, 2),
    microbatch_size=candidate,
  )
  reset_peak_memory(device)
  with profiler.step(batch_size=batch_size):
    grads, aux = grad_fn(params, x, y, state=state)

  print(candidate, profiler.current_metrics()["memory_peak_gb"])
```

**Memory comparison:**

| Technique | Memory | Compute | Opaque compatible |
|-----------|--------|---------|-------------------|
| No optimization | O(batch_size) | 1x | Yes |
| Gradient checkpointing | O(sqrt(batch_size)) | ~2x | No |
| Microbatching (size m) | O(m) | 1x | Yes |

**Root cause:** PyTorch's checkpoint uses `autograd.Function` internally.
`torch.func.vmap` requires functions to implement vmap rules, and the
checkpoint `autograd.Function` does not. This is tracked in
[PyTorch #165880](https://github.com/pytorch/pytorch/issues/165880). When
PyTorch resolves this, Opaque will automatically support checkpointing.

## Flash Attention 2 incompatibility

Flash Attention 2 uses `torch.nonzero` internally for sequence unpadding,
which produces dynamic output shapes incompatible with vmap.

**Solution:** Use SDPA or eager attention when loading HuggingFace models:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="sdpa",
)
```

See [HuggingFace Compatibility](user-guide/huggingface.md#attention-implementation-support)
for supported attention implementations.

## DDP only

Opaque supports `torch.nn.parallel.DistributedDataParallel` (DDP). FSDP,
Tensor Parallel, and Pipeline Parallel are not supported. Multi-node DDP
should work but is not extensively tested. The NCCL backend is recommended;
Gloo and MPI are not tested.

## Kernel patching lives in `opaque.compat`

Kernel optimization and patching for HuggingFace models is part of
`opaque.compat.transformers` and is CUDA+Triton only.

Low-level Triton-backed `Opaque_*` autograd classes (for example,
`Opaque_SwiGLU`, `Opaque_RoPE_QK`, `Opaque_LinearCrossEntropyLoss`) are
internal implementation details and should not be imported directly in user
code.

On CPU/MPS (or without Triton), Opaque falls back to non-kernel compatibility
paths. To control patching behavior, use the `OPAQUE_SKIP_COMPAT_PATCHES` and
`OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES` environment variables. See
[Kernel Optimizations](user-guide/kernel-optimizations.md#configuration).

Advanced users can still call kernel wrappers directly via
`opaque.compat.kernels`.

## In-place operations under vmap

`torch.func.vmap` does not support in-place tensor operations. Models that
use in-place operations (e.g., `x.add_()`, `x[:, 0] = 0`) in their
forward pass will fail. Replace them with out-of-place equivalents
(`x = x + y`, `x = torch.cat([zeros, x[:, 1:]], dim=1)`).

## Data-dependent control flow under vmap

vmap requires the same operations to execute for every example in the
batch. Conditional branches that depend on tensor values
(`if x.sum() > 0`) will fail. This is a fundamental limitation of
vectorized execution.
