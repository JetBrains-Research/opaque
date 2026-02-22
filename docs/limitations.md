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

Use `find_max_microbatch_size` from `opaque.profiling` to automatically
find the largest microbatch that fits in memory:

```python
from opaque.profiling import find_max_microbatch_size

optimal = find_max_microbatch_size(
    model=model,
    sample_batch=(sample_x, sample_y),
    batch_size=batch_size,
    loss_fn=loss_fn,
    l2_clip_norm=1.0,
    safety_margin=0.9,
)
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
