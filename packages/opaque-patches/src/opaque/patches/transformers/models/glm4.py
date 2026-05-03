# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from opaque.patches.transformers.components.batchify import apply_batchify_patch
from opaque.patches.transformers.components.kv_cache import apply_kv_cache_patch


import logging
import torch.nn as nn
from opaque.patches.transformers._router import _patch_forward
from opaque.patches.transformers.components.cross_entropy import _make_fused_ce_causal_lm_forward
from opaque.patches.transformers.components.rms_norm import _rmsnorm_fac_glm4
from opaque.patches.transformers.components.swiglu import _make_phi3_mlp_forward
from opaque.patches.transformers.components.attention import vmap_repeat_kv, vmap_eager_attention_forward
from opaque.patches.transformers.components.masking import apply_module_masking_patch


logger = logging.getLogger(__name__)

def apply_glm4_patches(
    model: nn.Module | None = None,
    *,
    performance: bool = True,
    compat: bool = True,
    **kwargs
) -> None:
    fuse_swiglu = kwargs.get('fuse_swiglu', performance)
    fuse_rms_norm = kwargs.get('fuse_rms_norm', performance)
    fuse_add_rms_norm = kwargs.get('fuse_add_rms_norm', performance)
    fuse_rope = kwargs.get('fuse_rope', performance)
    fuse_cross_entropy = kwargs.get('fuse_cross_entropy', performance)
    wrap_eager_attention = kwargs.get('wrap_eager_attention', compat)
    wrap_batchify = kwargs.get('wrap_batchify', compat)
    disable_kv_cache = kwargs.get('disable_kv_cache', compat)
    """Apply Triton kernel patches for glm4 model."""
    try:
        import transformers.models.glm4.modeling_glm4 as mod
    except ImportError:
        return

    if wrap_eager_attention:
        apply_module_masking_patch(mod)
        if hasattr(mod, 'repeat_kv'):
            mod.repeat_kv = vmap_repeat_kv
        if hasattr(mod, 'eager_attention_forward'):
            mod.eager_attention_forward = vmap_eager_attention_forward

    if fuse_swiglu:
        _patch_forward(getattr(mod, 'Glm4MLP', None), _make_phi3_mlp_forward, model)
    if fuse_rms_norm:
        _patch_forward(getattr(mod, 'Glm4RMSNorm', None), _rmsnorm_fac_glm4, model)
    if fuse_cross_entropy:
        _patch_forward(getattr(mod, 'Glm4ForCausalLM', None), _make_fused_ce_causal_lm_forward, model)

    causal_lm_cls = getattr(mod, 'Glm4ForCausalLM', None)
    if wrap_batchify:
        apply_batchify_patch(causal_lm_cls, model)
    if disable_kv_cache:
        apply_kv_cache_patch(causal_lm_cls, model)
