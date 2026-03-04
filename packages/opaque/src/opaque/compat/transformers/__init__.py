# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap compatibility and kernel optimization patches for HuggingFace Transformers models.

Patches are applied automatically at `import opaque` time.
No user action required - just import opaque and use clipped_grad with any
supported HuggingFace model.

Control with environment variables:
- OPAQUE_SKIP_TRANSFORMERS_PATCHES: "all" or "vmap,kernels"
- OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES: "all" or "shared,standard,gemma2,phi3"
- OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES: "all" or "swiglu,rope,ce,fused_ce,lora"

Supported models:
- GPT-2 (works without patches)
- LLaMA, Llama 3 (and LLaMA-based: DeepSeek, Mistral, etc.)
- Qwen2, Qwen3
- Phi-3
- Gemma, Gemma2
- Granite
- Cohere, Cohere2

Attention implementations:
- sdpa: Recommended. Fused CUDA kernels (flash/efficient/cuDNN), up to 3.6x memory savings over eager.
- eager: Supported. Materializes full attention matrix — O(N²) memory.
- flash_attention_2: Not compatible (uses torch.nonzero for unpadding, dynamic shapes incompatible with vmap)
- flex_attention: Not compatible (tensor metadata issues with vmap, known upstream PyTorch limitation)

Training features:
- Mixed precision (fp16/bfloat16): Fully supported
- Gradient checkpointing: Supported (enable with model.gradient_checkpointing_enable())
- PEFT/LoRA: Fully supported (LoRA, IA3, Prefix tuning, P-tuning, Prompt tuning tested)
- torch.compile: Fully supported
- CUDA: Fully supported
"""

from opaque._env import parse_skip_env
from opaque.compat.transformers._kernel_patches import (
    apply_kernel_patches,
    is_kernel_patched,
    patch_lora_model,
)
from opaque.compat.transformers._shared import apply_batchify_patches
from opaque.compat.transformers._vmap_patches import (
    apply_vmap_patches,
    is_vmap_patched,
)

_is_transformers_patched = False


def apply_transformers_patches() -> None:
    """Apply all HuggingFace Transformers patches.

    Controlled by environment variables:
    - OPAQUE_SKIP_TRANSFORMERS_PATCHES: "all" or "vmap,kernels"
    - OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES: "all" or "shared,standard,gemma2,phi3"
    - OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES: "all" or "swiglu,rope,ce,fused_ce,lora"
    """
    global _is_transformers_patched

    if _is_transformers_patched:
        return

    skip = parse_skip_env("OPAQUE_SKIP_TRANSFORMERS_PATCHES")
    if "all" in skip:
        _is_transformers_patched = True
        return

    if "vmap" not in skip:
        apply_vmap_patches()

    if "kernels" not in skip:
        apply_kernel_patches()

    # Batchify must run AFTER kernel patches: kernel patches may replace
    # model forward methods (e.g. fused CE), and batchify must wrap the
    # final version.  Also patches PEFT model classes.
    if "vmap" not in skip:
        apply_batchify_patches()

    _is_transformers_patched = True


def is_transformers_patched() -> bool:
    """Check if Transformers patches have been applied."""
    return _is_transformers_patched


__all__ = [
    "apply_transformers_patches",
    "apply_kernel_patches",
    "apply_vmap_patches",
    "is_transformers_patched",
    "is_kernel_patched",
    "is_vmap_patched",
    "patch_lora_model",
]
