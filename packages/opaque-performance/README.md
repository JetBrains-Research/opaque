# opaque-performance

Generic fused Triton kernels and PyTorch optimizations for vmap-style per-example training.

- `opaque.compat.kernels` — SwiGLU, GeGLU, RoPE, cross-entropy, linear-CE, LoRA
- `opaque.compat.pytorch` — gradient-checkpointing patches for vmap

Independent of DP — usable as a plain Triton kernel library.
