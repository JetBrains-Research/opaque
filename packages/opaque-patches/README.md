# opaque-patches

Unified package for Opaque patches, providing:
1. PyTorch checkpoint shim (`opaque.patches.torch.runtime`)
2. Hugging Face runtime compatibility patches (`opaque.patches.transformers.runtime`)
3. Hugging Face model/component patching (`opaque.patches.transformers.models`, `opaque.patches.transformers.components`)
4. Explicit PEFT/LoRA model patching (`opaque.patches.peft`)

This package is pulled in by `opaque` and should be consumed through the root
`opaque` install surface.
