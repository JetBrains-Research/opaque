# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from opaque.patches.transformers.components.batchify import apply_batchify_patch
from opaque.patches.transformers.components.kv_cache import apply_kv_cache_patch


import logging
import torch.nn as nn
from opaque.patches.transformers._router import _patch_forward
from opaque.patches.transformers.components.cross_entropy import _make_fused_ce_causal_lm_forward
from opaque.patches.transformers.components.fused_add_rms_norm import _fused_add_rms_fac_llama
from opaque.patches.transformers.components.rms_norm import _rmsnorm_fac_llama
from opaque.patches.transformers.components.rope import _opaque_apply_rotary_pos_emb
from opaque.patches.transformers.components.swiglu import _make_swiglu_mlp_forward
from opaque.patches.transformers.components.attention import vmap_repeat_kv, vmap_eager_attention_forward, apply_module_masking_patch


logger = logging.getLogger(__name__)

def apply_mistral_patches(
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
    """Apply Triton kernel patches for mistral model."""
    try:
        import transformers.models.mistral.modeling_mistral as mod
    except ImportError:
        return

    if wrap_eager_attention:
        apply_module_masking_patch(mod)
        if hasattr(mod, 'repeat_kv'):
            mod.repeat_kv = vmap_repeat_kv
        if hasattr(mod, 'eager_attention_forward'):
            mod.eager_attention_forward = vmap_eager_attention_forward

    if fuse_swiglu:
        _patch_forward(getattr(mod, 'MistralMLP', None), _make_swiglu_mlp_forward, model)
    if fuse_rms_norm:
        _patch_forward(getattr(mod, 'MistralRMSNorm', None), _rmsnorm_fac_llama, model)
    if fuse_add_rms_norm:
        _patch_forward(getattr(mod, 'MistralDecoderLayer', None), _fused_add_rms_fac_llama, model)
    if fuse_rope:
        if hasattr(mod, 'apply_rotary_pos_emb') and mod.apply_rotary_pos_emb is not _opaque_apply_rotary_pos_emb:
            mod.apply_rotary_pos_emb = _opaque_apply_rotary_pos_emb
    if fuse_cross_entropy:
        _patch_forward(getattr(mod, 'MistralForCausalLM', None), _make_fused_ce_causal_lm_forward, model)

    causal_lm_cls = getattr(mod, 'MistralForCausalLM', None)
    if wrap_batchify:
        apply_batchify_patch(causal_lm_cls, model)
    if disable_kv_cache:
        apply_kv_cache_patch(causal_lm_cls, model)
