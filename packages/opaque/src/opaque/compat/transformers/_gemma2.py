# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Gemma2-specific vmap compatibility patches.

Gemma2 uses softcap attention which requires special handling.
"""

import importlib

import torch

from opaque.compat.transformers._shared import vmap_repeat_kv

_GEMMA2_MODULE = "transformers.models.gemma2.modeling_gemma2"


def vmap_eager_attention_forward_gemma2(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vmap-compatible eager_attention_forward for Gemma2.

    Gemma2-specific variant that supports softcap attention logit capping.
    Uses negative indexing for transposes to handle arbitrary batch dimensions.

    Original (4D): query shape (batch, num_heads, seq_len, head_dim)
                   returns (batch, seq_len, num_heads, head_dim)
    Under vmap (3D): query shape (num_heads, seq_len, head_dim)
                     returns (seq_len, num_heads, head_dim)
    """
    key_states = vmap_repeat_kv(key, module.num_key_value_groups)
    value_states = vmap_repeat_kv(value, module.num_key_value_groups)

    # query shape: (..., num_heads, seq_len, head_dim)
    # key_states shape: (..., num_heads, seq_len, head_dim)
    attn_weights = torch.matmul(query, key_states.transpose(-2, -1))

    # Apply scaling if provided
    if scaling is not None:
        attn_weights = attn_weights * scaling

    # Apply softcap (Gemma2-specific)
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        # Slice mask to match query and key lengths
        # attention_mask shape: (..., 1, full_q_len, full_kv_len) or (..., num_heads, q_len, kv_len)
        # attn_weights shape: (..., num_heads, q_len, kv_len)
        q_len = query.shape[-2]
        kv_len = key_states.shape[-2]

        # Handle both mask formats: (..., 1, q, kv) or (..., h, q, kv)
        # Slice the last two dimensions to match actual sequence lengths
        causal_mask = attention_mask[..., :q_len, :kv_len]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    attn_output = torch.matmul(attn_weights, value_states)
    # Transpose to move seq_len before num_heads: (..., num_heads, seq_len, head_dim) -> (..., seq_len, num_heads, head_dim)
    attn_output = attn_output.transpose(-3, -2).contiguous()

    return attn_output, attn_weights


# =============================================================================
# Patch application
# =============================================================================


def apply_gemma2_patches() -> None:
    """Apply Gemma2-specific vmap patches.

    Patches Gemma2 with softcap-aware attention implementation.

    Note: Requires apply_shared_patches() from _shared to be called first,
    as Gemma2 depends on vmap_create_causal_mask from masking_utils.
    """
    try:
        module = importlib.import_module(_GEMMA2_MODULE)

        # Patch repeat_kv with base implementation
        if hasattr(module, "repeat_kv"):
            module.repeat_kv = vmap_repeat_kv

        # Patch eager_attention_forward with Gemma2-specific implementation
        if hasattr(module, "eager_attention_forward"):
            module.eager_attention_forward = vmap_eager_attention_forward_gemma2

    except ImportError:
        # Gemma2 not available in this transformers version
        pass
