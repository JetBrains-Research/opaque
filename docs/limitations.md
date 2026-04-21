# Known Limitations

## Matrix factorization (MF) workloads

Correlated-noise mechanisms are analyzed for a **specific linear map** (the strategy
matrix) and optional subsampling model. **DP correctness** requires that accounting,
noise, and sampling match that map. **Utility** can depend on how closely the
encoded workload matches your optimizer (for example, BandMF/BLT `lr_schedule` as a
Toeplitz surrogate when \(\eta_t\) varies). JME adds a second MF stream; new strategy
types must extend `_derive_second_strategy` explicitly.

See [Matrix factorization (MF)](user-guide/matrix-factorization.md).

## Gradient checkpointing

Supported under `vmap(grad(...))` via automatic patches. See
[Memory Optimizations](user-guide/memory-optimizations.md#gradient-checkpointing)
for usage and limitations.

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

## Kernel patching lives in `opaque.huggingface.patches`

Kernel optimization and patching for HuggingFace models is part of
`opaque.huggingface.patches` and is CUDA+Triton only.

Low-level Triton-backed `Opaque_*` autograd classes (for example,
`Opaque_SwiGLU`, `Opaque_RoPE_QK`, `Opaque_LinearCrossEntropyLoss`) are
internal implementation details and should not be imported directly in user
code.

On CPU/MPS (or without Triton), Opaque falls back to non-kernel compatibility
paths. To control patching behavior, use the `OPAQUE_SKIP_COMPAT_PATCHES` and
`OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES` environment variables. See
[HuggingFace Compatibility](user-guide/huggingface.md#configuration).

Advanced users can still call kernel wrappers directly via
`opaque.performance.kernels`.

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
