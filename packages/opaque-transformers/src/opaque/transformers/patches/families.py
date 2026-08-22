"""HF Transformers compat patches — vmap-safe attention, KV cache, custom-family registry.

The callable contracts a custom family is written against —
``FamilyPatchFn``, ``ForwardFactory``, ``ForwardFn``, ``ModelPatchFn``,
``ModulePatcher`` — live in :mod:`opaque.transformers.patches.types`.
"""

from opaque.api.transformers.patches.families import (
    apply_transformers_model_patches,
    family_name,
    make_apply_family_patches,
    make_apply_model_patches,
    register_activation_kind,
    register_family,
    register_fused_add_rms_kind,
    register_rms_norm_kind,
    supported_families,
)

__all__ = [
    "apply_transformers_model_patches",
    "family_name",
    "make_apply_family_patches",
    "make_apply_model_patches",
    "register_activation_kind",
    "register_family",
    "register_fused_add_rms_kind",
    "register_rms_norm_kind",
    "supported_families",
]
