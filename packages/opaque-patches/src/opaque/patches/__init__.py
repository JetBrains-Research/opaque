"""Unified patching for opaque.

This package provides:
- Model-level patches: `apply_model_patches()`
- Runtime-level patches: `apply_runtime_patches()`
"""

import logging

import torch.nn as nn

from opaque.patches.transformers._router import apply_transformers_model_patches
from opaque.patches.peft import apply_peft_model_patches

logger = logging.getLogger(__name__)

_runtime_patches_applied: bool = False

def apply_model_patches(
    model: nn.Module,
    *,
    performance: bool = True,
    compat: bool = True,
    peft: bool = True,
    **kwargs
) -> None:
    """Apply global and instance-level patches for a specific model.
    
    This is a convenience orchestrator that handles both Transformers and PEFT natively.
    Users with purely custom non-Transformers architectures can import and invoke 
    `apply_peft_model_patches` directly from the root namespace to apply LoRA kernels.
    """
    global _runtime_patches_applied
    if not _runtime_patches_applied:
        apply_runtime_patches()

    allowed_kwargs = {
        "fuse_swiglu", "fuse_rms_norm", "fuse_add_rms_norm", "fuse_rope", 
        "fuse_cross_entropy", "fuse_lora", "wrap_eager_attention", 
        "wrap_batchify", "disable_kv_cache"
    }
    for k in kwargs:
        if k not in allowed_kwargs:
            logger.warning(f"opaque: Unknown patch kwarg '{k}' passed to apply_model_patches. It will be ignored.")
    
    try:
        from opaque.patches.transformers._router import apply_transformers_model_patches
        apply_transformers_model_patches(
            model,
            performance=performance,
            compat=compat,
            **kwargs
        )
    except ImportError:
        logger.debug("opaque: Hugging Face kernel patches not available.")

    fuse_lora = kwargs.get("fuse_lora", peft and performance)
    if fuse_lora:
        try:
            from opaque.patches.peft import apply_peft_model_patches
            apply_peft_model_patches(model)
        except ImportError:
            pass

def apply_runtime_patches(
    *,
    performance: bool = True,
    compat: bool = True,
    **kwargs
) -> None:
    """Apply global runtime patches."""
    global _runtime_patches_applied
    _runtime_patches_applied = True

    allowed_kwargs = {
        "enable_vmap_checkpointing", "use_fused_loss", 
        "enable_vmap_masking", "allow_empty_batches"
    }
    for k in kwargs:
        if k not in allowed_kwargs:
            logger.warning(f"opaque: Unknown patch kwarg '{k}' passed to apply_runtime_patches. It will be ignored.")

    enable_vmap_masking = kwargs.get("enable_vmap_masking", compat)
    allow_empty_batches = kwargs.get("allow_empty_batches", compat)
    enable_vmap_checkpointing = kwargs.get("enable_vmap_checkpointing", compat)
    use_fused_loss = kwargs.get("use_fused_loss", performance)

    if enable_vmap_masking:
        try:
            from opaque.patches.transformers.runtime.masking import apply_masking_patches
            apply_masking_patches(enable_vmap_masking=enable_vmap_masking)
        except ImportError:
            pass

    if allow_empty_batches:
        try:
            from opaque.patches.transformers.runtime.collator import apply_collator_patches
            apply_collator_patches(allow_empty_batches=allow_empty_batches)
        except ImportError:
            pass

    if enable_vmap_checkpointing:
        try:
            from opaque.patches.torch.runtime import apply_checkpoint_patch
            apply_checkpoint_patch(enable_vmap_checkpointing=enable_vmap_checkpointing)
        except ImportError:
            pass
    
    if use_fused_loss:
        try:
            from opaque.patches.transformers.runtime.loss_mapping import apply_loss_mapping_patch
            apply_loss_mapping_patch(use_fused_loss=use_fused_loss)
        except ImportError:
            logger.debug("opaque: Hugging Face kernel patches not available. Skipping loss mapping.")



__all__ = [
    "apply_model_patches",
    "apply_runtime_patches",
    "apply_transformers_model_patches",
    "apply_peft_model_patches",
]

