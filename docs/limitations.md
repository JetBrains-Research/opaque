# Known Limitations

## Matrix factorization (MF) workloads

Correlated-noise mechanisms are analyzed for a **specific linear map** (the strategy
matrix) and optional subsampling model. **DP correctness** requires that accounting,
noise, and sampling match that map. **Utility** can depend on how closely the
encoded workload matches your optimizer (for example, BandMF/BLT `lr_schedule` as a
Toeplitz surrogate when \(\eta_t\) varies). Private second moments add a second MF
stream; pass its strategy explicitly.

See [Matrix factorization (MF)](user-guide/dp-ftrl.md).

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

See [Model Patches — Attention implementations](user-guide/huggingface/model-patches.md#attention-implementations)
for supported attention implementations.

## DDP only

Opaque supports `torch.nn.parallel.DistributedDataParallel` (DDP). FSDP,
Tensor Parallel, and Pipeline Parallel are not supported. Multi-node DDP
should work but is not extensively tested. First-class distributed backends are
NCCL, Gloo, and MPI. Vendor/runtime-specific backends require external stacks
and are not covered by default CI.

## Kernel patching lives in `opaque.patches`

Kernel optimization and patching for HuggingFace models is part of
`opaque.patches` and is CUDA+Triton only.

Low-level Triton-backed autograd primitives are internal
implementation details and should not be imported directly in user
code.

On CPU/MPS (or without Triton), the kernel group is auto-disabled —
the router forces `kernels=False` when CUDA + Triton can't be
imported, so `performance=True` keeps the pure-Python `kv_cache`
patch on those hosts.  Configure the patch surface via the explicit
flags (see
[Model Patches — Configuration](user-guide/huggingface/model-patches.md#configuration-via-trainingarguments));
opaque-patches has no environment-variable kill switches.

Public standalone kernels (`opaque_swiglu`, `opaque_cross_entropy_loss`,
`opaque_lora_w`, `opaque_lora_qkv`, `opaque_lora_mlp`) are importable
from `opaque.patches.kernels`.

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
