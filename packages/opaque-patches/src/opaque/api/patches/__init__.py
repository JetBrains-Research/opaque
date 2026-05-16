"""Unified patching for opaque.

This package provides:
- Model-level patches: `apply_model_patches()`
- Runtime-level patches: `apply_runtime_patches()`
"""

import logging

import torch.nn as nn

from opaque.api.patches.transformers._router import apply_transformers_model_patches
from opaque.api.patches.peft import apply_peft_model_patches

logger = logging.getLogger(__name__)

_runtime_patches_applied: bool = False


def apply_model_patches(
    model: nn.Module,
    *,
    performance: bool = True,
    compat: bool = True,
    peft: bool = True,
    **kwargs,
) -> None:
    """Apply global and instance-level patches for a specific model.

    This is a convenience orchestrator that handles both Transformers and
    PEFT natively.  Users with purely custom non-Transformers
    architectures can import and invoke ``apply_peft_model_patches``
    directly from the root namespace to apply LoRA kernels.

    Per-concern flags pass through ``**kwargs``. Performance bucket
    (kernel / memory-efficiency, default-on with ``performance=True``):
    ``rope``, ``rms_norm``, ``activation``, ``cross_entropy``,
    ``fused_linear_cross_entropy``, ``kv_cache``. Compat bucket (vmap
    safety, default-on with ``compat=True``): ``eager_attention``,
    ``batchify``.

    ``cross_entropy`` swaps in the non-fused CE kernel via
    ``loss_function`` (logits still materialized — safe for callers that
    read them). ``fused_linear_cross_entropy`` replaces the forward to
    skip ``lm_head`` and compute loss directly from hidden states; on the
    fused path it returns ``logits=None``. The latter cascades from the
    former (so the historical ``cross_entropy=False`` opt-out still turns
    both off); pass ``fused_linear_cross_entropy=False`` alone when the
    trainer needs logits (e.g. SFTTrainer with ``compute_metrics`` /
    ``preprocess_logits_for_metrics``) while keeping the non-fused
    kernel on materialized logits.
    """
    global _runtime_patches_applied
    if not _runtime_patches_applied:
        apply_runtime_patches(performance=performance, compat=compat, **kwargs)

    try:
        from opaque.api.patches.transformers._router import (
            apply_transformers_model_patches,
        )

        apply_transformers_model_patches(
            model, performance=performance, compat=compat, **kwargs
        )
    except ImportError:
        logger.debug("opaque: Hugging Face kernel patches not available.")

    if peft:
        try:
            from opaque.api.patches.peft import apply_peft_model_patches

            apply_peft_model_patches(
                model, performance=performance, compat=compat, **kwargs
            )
        except ImportError:
            pass


def apply_runtime_patches(
    *, performance: bool = True, compat: bool = True, **kwargs
) -> None:
    """Apply global runtime patches.

    Liger-style flag names: ``vmap_masking``, ``empty_batches``,
    ``vmap_checkpointing``.  ``use_fused_loss`` was dropped — fused-CE
    is now applied per-model via :func:`apply_model_patches` when
    ``cross_entropy=True``.
    """
    global _runtime_patches_applied
    _runtime_patches_applied = True

    vmap_masking = kwargs.get("vmap_masking", compat)
    empty_batches = kwargs.get("empty_batches", compat)
    vmap_checkpointing = kwargs.get("vmap_checkpointing", compat)

    if vmap_masking:
        try:
            from opaque.api.patches.transformers.runtime.masking import (
                apply_masking_patches,
            )

            apply_masking_patches(vmap_masking=vmap_masking)
        except ImportError:
            pass

    if empty_batches:
        try:
            from opaque.api.patches.transformers.runtime.collator import (
                apply_collator_patches,
            )

            apply_collator_patches(empty_batches=empty_batches)
        except ImportError:
            pass

    if vmap_checkpointing:
        try:
            from opaque.api.patches.torch.runtime import apply_checkpoint_patch

            apply_checkpoint_patch(vmap_checkpointing=vmap_checkpointing)
        except ImportError:
            pass


__all__ = [
    "apply_model_patches",
    "apply_runtime_patches",
    "apply_transformers_model_patches",
    "apply_peft_model_patches",
]
