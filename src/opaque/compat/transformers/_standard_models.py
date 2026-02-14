# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Standard model patches for HuggingFace transformers.

These patches work for models that use standard eager attention:
LLaMA, Mistral, Qwen2, Phi, Phi3, OLMo, Gemma.
"""

import importlib

import torch

from opaque.compat.transformers._shared import vmap_repeat_kv

# Standard models that use these patches
_STANDARD_MODEL_MODULES = [
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.phi.modeling_phi",
    "transformers.models.phi3.modeling_phi3",
    "transformers.models.olmo.modeling_olmo",
    "transformers.models.gemma.modeling_gemma",
]


def vmap_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vmap-compatible eager_attention_forward.

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
    attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling

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


def apply_standard_model_patches() -> None:
    """Apply patches to standard models.

    Patches models that use standard eager attention:
    - LLaMA, Mistral, Qwen2, Phi, Phi3, OLMo, Gemma

    Note: Requires apply_shared_patches() to be called first.
    """
    for module_path in _STANDARD_MODEL_MODULES:
        try:
            module = importlib.import_module(module_path)

            if hasattr(module, "repeat_kv"):
                module.repeat_kv = vmap_repeat_kv

            if hasattr(module, "eager_attention_forward"):
                module.eager_attention_forward = vmap_eager_attention_forward

        except ImportError:
            # Model not available in this transformers version
            pass
