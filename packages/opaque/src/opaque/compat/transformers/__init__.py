# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap compatibility and memory optimization patches for HuggingFace Transformers models.

## vmap Compatibility Patches

Patches are applied automatically at `import opaque` time.
No user action required - just import opaque and use clipped_grad with any
supported HuggingFace model.

Disable auto-patching with: OPAQUE_NO_PATCH=1

Supported models:
- GPT-2
- LLaMA (and LLaMA-based: Mistral, DeepSeek, etc.)
- Qwen2
- Phi, Phi-3
- OLMo
- Gemma, Gemma2

Attention implementations:
- eager: ✅ Fully supported (explicitly patched, tested on CPU and CUDA)
- sdpa: ✅ Fully supported (uses patched repeat_kv, default in transformers, tested on CPU and CUDA)
- flash_attention_2: ❌ Not compatible (uses torch.nonzero for unpadding, which outputs dynamic shapes incompatible with vmap)
  * Cannot be patched without rewriting the entire kernel (defeats performance purpose)
- flex_attention: ❌ Not compatible (tensor metadata issues with vmap, known upstream PyTorch limitation)
  * May be fixable in future PyTorch versions as flex_attention matures

Training features:
- Mixed precision (fp16/bfloat16): ✅ Fully supported
- Gradient checkpointing: ❌ Not compatible (autograd.Function incompatible with vmap)
- PEFT/LoRA: ✅ Fully supported (LoRA, IA3, Prefix tuning, P-tuning, Prompt tuning tested)
- torch.compile: ✅ Fully supported
- CUDA: ✅ Fully supported

## Memory Optimization Patches

Memory optimizations are also applied automatically at `import opaque` time.
For models with large vocabularies (e.g., Mellum with 98K vocab), the lm_head
is automatically replaced with ChunkedLinear on first forward pass when:
1. lm_head weights are frozen (requires_grad=False, standard for LoRA)
2. Vocabulary size exceeds 50K

This provides ~20% memory reduction on lm_head backward pass by:
- Processing backward pass in chunks (reduces peak memory)
- Skipping weight gradients (lm_head frozen in DP-SGD with LoRA)
- Being vmap-compatible via generate_vmap_rule = True

No user action required - just `import opaque` and train with LoRA.
To check if optimization is active: `is_model_memory_optimized(model)`

## Testing

- tests/compat/ - Patch-specific compatibility tests (18 tests covering attention, PEFT, architectures)
- tests/validation/ - End-to-end DP training validation
- tests/kernels/ - ChunkedLinear unit tests

Note: SDPA is the default attention implementation in recent transformers versions.
It works with our patches but may show performance warnings due to missing batching
rules for scaled_dot_product_attention. This is expected and does not affect correctness.
"""

from opaque.compat.transformers._global_patches import (
    apply_global_patches,
    is_globally_patched,
)
from opaque.compat.transformers._memory_optimizations import (
    apply_memory_patches,
    is_memory_patched,
    is_model_memory_optimized,
)

__all__ = [
    "apply_global_patches",
    "apply_memory_patches",
    "is_globally_patched",
    "is_memory_patched",
    "is_model_memory_optimized",
]
