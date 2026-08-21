# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for the mistral family — built via the patch factories.

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

_MODULE_PATH = "transformers.models.mistral.modeling_mistral"


apply_mistral_family_patches = make_apply_family_patches(
    family="mistral",
    module_path=_MODULE_PATH,
)


apply_mistral_patches = make_apply_model_patches(
    family="mistral",
    family_apply=apply_mistral_family_patches,
    module_path=_MODULE_PATH,
    classes={
        "mlp": "MistralMLP",
        "rms_norm": "MistralRMSNorm",
        "decoder_layer": "MistralDecoderLayer",
        "causal_lm": "MistralForCausalLM",
    },
    activation_kind="swiglu",
    rms_norm_kind="llama",
    fused_add_rms_kind="llama",
)


register_family("mistral", apply_mistral_patches)


__all__ = ["apply_mistral_family_patches", "apply_mistral_patches"]
