# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Model-level compatibility patching functions for vmap."""

import torch

def vmap_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """vmap-compatible repeat_kv for expanding key/value heads to match query heads.

    Uses negative indexing to handle arbitrary batch dimensions from vmap.

    Original (4D): (batch, num_kv_heads, slen, head_dim) -> (batch, num_heads, slen, head_dim)
    Under vmap (3D): (num_kv_heads, slen, head_dim) -> (num_heads, slen, head_dim)
    """
    if n_rep == 1:
        return hidden_states

    # Use negative indexing: works for both 4D (batch, heads, seq, dim) and 3D (heads, seq, dim)
    leading_dims = hidden_states.shape[:-3]  # Empty for 3D, (batch,) for 4D
    num_kv_heads = hidden_states.shape[-3]
    slen = hidden_states.shape[-2]
    head_dim = hidden_states.shape[-1]

    # Reshape to add repeat dimension, then flatten back
    # 4D: (batch, num_kv_heads, slen, head_dim) -> (batch, num_kv_heads, 1, slen, head_dim)
    # 3D: (num_kv_heads, slen, head_dim) -> (num_kv_heads, 1, slen, head_dim)
    hidden_states = hidden_states.unsqueeze(-3)

    # Expand along the new dimension
    # 4D: (batch, num_kv_heads, 1, slen, head_dim) -> (batch, num_kv_heads, n_rep, slen, head_dim)
    # 3D: (num_kv_heads, 1, slen, head_dim) -> (num_kv_heads, n_rep, slen, head_dim)
    expand_shape = list(hidden_states.shape)
    expand_shape[-3] = n_rep
    hidden_states = hidden_states.expand(*expand_shape)

    # Reshape to merge kv_heads and n_rep
    # 4D: (batch, num_kv_heads, n_rep, slen, head_dim) -> (batch, num_heads, slen, head_dim)
    # 3D: (num_kv_heads, n_rep, slen, head_dim) -> (num_heads, slen, head_dim)
    new_shape = (*leading_dims, num_kv_heads * n_rep, slen, head_dim)
    return hidden_states.reshape(*new_shape)

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

def _make_vmap_compatible_init(original_init):
    """Create a vmap-compatible init for DynamicCache."""
    def vmap_compatible_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if not hasattr(self, "get_usable_length"):
            def get_usable_length(
                kv_seq_len: int | None = None,
                layer_idx: int | None = None,
            ) -> int:
                if layer_idx is None:
                    return 0
                if (
                    hasattr(self, "key_cache")
                    and len(self.key_cache) > layer_idx
                ):
                    kc = self.key_cache[layer_idx]
                    if kc is not None:
                        return kc.shape[-2]
                if hasattr(self, "seen_tokens"):
                    return self.seen_tokens
                return 0

            self.get_usable_length = get_usable_length

    return vmap_compatible_init
