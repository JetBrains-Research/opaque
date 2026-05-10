"""Patches transformers façade — re-exports from ``opaque.api.patches.transformers``."""

from opaque.api.patches.transformers import (
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
