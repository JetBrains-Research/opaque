# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from opaque.patches.transformers.components.batchify import apply_batchify_patch
from opaque.patches.transformers.components.kv_cache import apply_kv_cache_patch


import logging
import torch.nn as nn
from opaque.patches.transformers._router import _patch_forward
from opaque.patches.transformers.components.cross_entropy import (
    _make_fused_ce_causal_lm_forward,
)
from opaque.patches.transformers.components.fused_add_rms_norm import (
    _fused_add_rms_fac_llama,
)
from opaque.patches.transformers.components.rms_norm import _rmsnorm_fac_llama
from opaque.patches.transformers.components.rope import _opaque_apply_rotary_pos_emb
from opaque.patches.transformers.components.swiglu import _make_swiglu_mlp_forward
from opaque.patches.transformers.components.attention import (
    vmap_repeat_kv,
    vmap_eager_attention_forward,
)
from opaque.patches.transformers.components.masking import apply_module_masking_patch


logger = logging.getLogger(__name__)


def apply_llama_patches(
    model: nn.Module | None = None,
    *,
    performance: bool = True,
    compat: bool = True,
    **kwargs,
) -> None:
    """Apply Triton kernel patches for llama model.

    Liger-aligned flag names: ``rope``, ``rms_norm`` (unifies the standalone
    RMSNorm patch and the fused-add variant; the latter only applies on
    architectures that support it — Gemma2/3 — and is a no-op on Llama),
    ``swiglu``, ``cross_entropy``. Opaque-specific flags (vmap-safety,
    DP-SGD specific): ``eager_attention``, ``batchify``, ``kv_cache``.

    Polarity note: ``kv_cache=True`` *applies* the vmap-safe KV-cache
    patch, which has the side effect of disabling HF's stateful cache
    (the cache is incompatible with vmap).
    """
    rope = kwargs.get("rope", performance)
    rms_norm = kwargs.get("rms_norm", performance)
    swiglu = kwargs.get("swiglu", performance)
    cross_entropy = kwargs.get("cross_entropy", performance)
    eager_attention = kwargs.get("eager_attention", compat)
    batchify = kwargs.get("batchify", compat)
    kv_cache = kwargs.get("kv_cache", compat)

    try:
        import transformers.models.llama.modeling_llama as mod
    except ImportError:
        return

    if eager_attention:
        apply_module_masking_patch(mod)
        if hasattr(mod, "repeat_kv"):
            mod.repeat_kv = vmap_repeat_kv
        if hasattr(mod, "eager_attention_forward"):
            mod.eager_attention_forward = vmap_eager_attention_forward

    if swiglu:
        _patch_forward(getattr(mod, "LlamaMLP", None), _make_swiglu_mlp_forward, model)
    if rms_norm:
        # Unified rms_norm: standard RMSNorm kernel + fused-add variant
        # where the architecture supports it.
        _patch_forward(getattr(mod, "LlamaRMSNorm", None), _rmsnorm_fac_llama, model)
        _patch_forward(
            getattr(mod, "LlamaDecoderLayer", None), _fused_add_rms_fac_llama, model
        )
    if rope:
        if (
            hasattr(mod, "apply_rotary_pos_emb")
            and mod.apply_rotary_pos_emb is not _opaque_apply_rotary_pos_emb
        ):
            mod.apply_rotary_pos_emb = _opaque_apply_rotary_pos_emb
    if cross_entropy:
        _patch_forward(
            getattr(mod, "LlamaForCausalLM", None),
            _make_fused_ce_causal_lm_forward,
            model,
        )

    causal_lm_cls = getattr(mod, "LlamaForCausalLM", None)
    if batchify:
        apply_batchify_patch(causal_lm_cls, model)
    if kv_cache:
        apply_kv_cache_patch(causal_lm_cls, model)
