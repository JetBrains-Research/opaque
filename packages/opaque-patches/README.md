# opaque-patches

Unified package for Opaque patches, providing:
1. PyTorch checkpoint shim (`opaque.patches.torch.checkpoint`)
2. Hugging Face compatibility patches (`opaque.patches.transformers.compat`)
3. Hugging Face explicit Triton kernel patches (`opaque.patches.transformers.kernels`)

This package is a mandatory dependency for `opaque`, `opaque-dpsgd`, and `opaque-dpftrl`.
