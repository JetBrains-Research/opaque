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
    fused_linear_cross_entropy: bool = False,
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
    ``kv_cache``. Compat bucket (vmap safety, default-on with
    ``compat=True``): ``eager_attention``, ``batchify``.

    ``cross_entropy`` swaps in the non-fused CE kernel via
    ``loss_function`` — logits stay materialized, so trainers that
    inspect them (compute_metrics, preprocess_logits_for_metrics, eval
    that reads ``outputs.logits``) keep working.

    ``fused_linear_cross_entropy`` is the one per-concern flag promoted
    to an explicit kwarg because it is the only patch whose default is
    ``False`` regardless of ``performance``. When enabled, the forward
    replacement skips ``lm_head`` and computes loss directly from hidden
    states; the fused path returns ``logits=None``, so enable it only
    when loss is the only consumer of the forward output (the bundled
    training examples do exactly that).
    """
    global _runtime_patches_applied
    if not _runtime_patches_applied:
        apply_runtime_patches(performance=performance, compat=compat, **kwargs)

    try:
        from opaque.api.patches.transformers._router import (
            apply_transformers_model_patches,
        )

        apply_transformers_model_patches(
            model,
            performance=performance,
            compat=compat,
            fused_linear_cross_entropy=fused_linear_cross_entropy,
            **kwargs,
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

    Per-concern compat flags (default-on with ``compat=True``):
    ``vmap_masking`` (vmap-safe causal-mask builders),
    ``empty_batches`` (collator handling for Poisson-sampled empty
    batches), ``vmap_checkpointing`` (gradient-checkpointing shim).
    Per-model CE patches live on :func:`apply_model_patches`.
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
