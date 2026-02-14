# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared vmap-compatible utilities for HuggingFace transformers.

These patches are used by all models (standard and custom).
"""

from typing import Optional

import torch


def vmap_create_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values,
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function=None,
    and_mask_function=None,
) -> Optional[torch.Tensor]:
    """vmap-compatible create_causal_mask.

    Handles arbitrary batch dimensions. Under vmap, batch dimension is removed.

    Original: input_embeds (batch, seq, hidden) -> mask (batch, 1, seq, seq)
    Under vmap: input_embeds (seq, hidden) -> mask (1, seq, seq)
    """
    # Under vmap, input_embeds has shape (seq_len, hidden_dim)
    # Without vmap, input_embeds has shape (batch_size, seq_len, hidden_dim)
    if input_embeds.ndim == 2:
        # Under vmap: add batch dimension
        batch_size = 1
        seq_len = input_embeds.shape[0]
    else:
        # Normal execution
        batch_size = input_embeds.shape[0]
        seq_len = input_embeds.shape[1]

    # Determine target_length (total sequence length including cache)
    past_seen_tokens = 0
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            past_seen_tokens = past_key_values.get_seq_length()
        elif hasattr(past_key_values, "seen_tokens"):
            past_seen_tokens = past_key_values.seen_tokens
    target_length = past_seen_tokens + seq_len

    # Create causal mask
    causal_mask = torch.full(
        (batch_size, 1, seq_len, target_length),
        torch.finfo(input_embeds.dtype).min,
        dtype=input_embeds.dtype,
        device=input_embeds.device,
    )

    # Fill the causal part (allow attention to past and current tokens)
    if seq_len > 1:
        # For each query position, allow attention to keys up to and including that position
        mask_cond = cache_position.view(seq_len, 1) >= cache_position.view(1, target_length)
        causal_mask[..., :seq_len, :target_length] = torch.where(
            mask_cond,
            torch.tensor(0.0, dtype=input_embeds.dtype, device=input_embeds.device),
            causal_mask[..., :seq_len, :target_length],
        )
    else:
        # Single token: can attend to all cached tokens
        causal_mask[..., :, :target_length] = 0.0

    # Apply attention_mask if provided (padding mask)
    if attention_mask is not None:
        # attention_mask: (batch, seq) with 1 for valid, 0 for padding
        if attention_mask.ndim == 1:
            # Under vmap: (seq,) -> (1, seq)
            attention_mask = attention_mask.unsqueeze(0)

        # Expand to 4D: (batch, 1, 1, target_length)
        attention_mask = attention_mask[:, None, None, :]

        # Combine: set padding positions to -inf
        causal_mask = causal_mask.masked_fill(attention_mask == 0, torch.finfo(input_embeds.dtype).min)

    return causal_mask


def vmap_repeat_kv(
    hidden_states: torch.Tensor, n_rep: int
) -> torch.Tensor:
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


# =============================================================================
# Patch application
# =============================================================================


def apply_shared_patches() -> None:
    """Apply patches to shared utilities used by all models.

    Patches:
    - transformers.masking_utils.create_causal_mask
    - transformers.integrations.sdpa_attention.repeat_kv

    These are required by all models (standard models, Gemma2, etc.).
    """
    # Patch shared masking_utils
    try:
        import transformers.masking_utils as masking_utils

        if hasattr(masking_utils, "create_causal_mask"):
            masking_utils.create_causal_mask = vmap_create_causal_mask
    except ImportError:
        pass

    # Patch shared sdpa_attention (used by SDPA implementation)
    try:
        import transformers.integrations.sdpa_attention as sdpa_attention

        if hasattr(sdpa_attention, "repeat_kv"):
            sdpa_attention.repeat_kv = vmap_repeat_kv
    except ImportError:
        pass
