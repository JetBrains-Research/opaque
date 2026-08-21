# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the ministral family — built via the patch factories.

The factories close over architecture-specific knobs (MLP kind, RMSNorm
casting, whether the family supports fused-add RMS) at construction
time, so the dispatch is bug-by-construction: e.g. Gemma's
``activation_kind="geglu_exact"`` cannot accidentally route to SwiGLU.

Registration: this module calls ``register_family`` at import time —
the same mechanism downstream users follow to add their own families.
"""

from __future__ import annotations

from opaque.api.transformers.patches._factory import make_apply_model_patches
from opaque.api.transformers.patches._family import make_apply_family_patches
from opaque.api.transformers.patches._registry import register_family

_MODULE_PATH = "transformers.models.ministral.modeling_ministral"


apply_ministral_family_patches = make_apply_family_patches(
    family="ministral",
    module_path=_MODULE_PATH,
)


apply_ministral_patches = make_apply_model_patches(
    family="ministral",
    family_apply=apply_ministral_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "MinistralMLP",
        "rms_norm": "MinistralRMSNorm",
        "decoder_layer": "MinistralDecoderLayer",
        "causal_lm": "MinistralForCausalLM",
    },
    activation_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind="llama",
)


register_family("ministral", apply_ministral_patches)


__all__ = ["apply_ministral_family_patches", "apply_ministral_patches"]
