# opaque-patches

Unified package for Opaque patches, providing:

1. Hugging Face runtime compatibility patches (`opaque.patches.transformers.runtime`)
2. Hugging Face model/component patching (`opaque.patches.transformers.models`, `opaque.patches.transformers.components`)
3. Explicit PEFT/LoRA model patching (`opaque.patches.peft`)
4. Fused Triton kernels with PyTorch fallbacks (`opaque.patches.kernels`)

The Torch-core compatibility shims belong to the provider —
`opaque.torch.apply_runtime_patches` and `opaque.torch.checkpoint`.
`apply_runtime_patches` here forwards to them, so one call covers both layers.

This package is pulled in by `opaque` and should be consumed through the root
`opaque` install surface.
