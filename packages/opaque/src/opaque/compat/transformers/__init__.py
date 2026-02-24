# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap compatibility patches for HuggingFace Transformers models.

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

Testing:
- tests/compat/ - Patch-specific compatibility tests (18 tests covering attention, PEFT, architectures)
- tests/validation/ - End-to-end DP training validation

Note: SDPA is the default attention implementation in recent transformers versions.
It works with our patches but may show performance warnings due to missing batching
rules for scaled_dot_product_attention. This is expected and does not affect correctness.
"""

from opaque.compat.transformers._global_patches import (
    apply_global_patches,
    is_globally_patched,
)

__all__ = [
    "apply_global_patches",
    "is_globally_patched",
]
