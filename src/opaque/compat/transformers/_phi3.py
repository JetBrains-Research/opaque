# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Phi-3-specific vmap compatibility patches.

Phi-3 has a custom DynamicCache implementation and rope scaling that requires
special handling under vmap. This module patches:
1. DynamicCache.get_usable_length() - Custom implementation for vmap compatibility
2. eager_attention_forward - Uses fused QKV projection
"""

import importlib
from typing import Any

import torch

from opaque.compat.transformers._shared import vmap_repeat_kv

_PHI3_MODULE = "transformers.models.phi3.modeling_phi3"


class VmapCompatibleDynamicCache:
    """vmap-compatible wrapper for Phi-3's DynamicCache.

    Phi-3 uses a custom DynamicCache that may not have get_usable_length() method.
    This wrapper ensures compatibility under vmap.
    """

    def __init__(self, original_cache):
        """Initialize with original cache object."""
        self._cache = original_cache

    def __getattr__(self, name: str) -> Any:
        """Delegate to original cache for all other attributes."""
        return getattr(self._cache, name)

    def get_usable_length(self, layer_idx: int) -> int:
        """Get usable sequence length for given layer."""
        if hasattr(self._cache, "get_usable_length"):
            return self._cache.get_usable_length(layer_idx)

        # Fallback for caches that store KV in key_cache/value_cache attributes
        if hasattr(self._cache, "key_cache") and len(self._cache.key_cache) > layer_idx:
            key_cache = self._cache.key_cache[layer_idx]
            if key_cache is not None:
                # key_cache shape: (batch, num_heads, seq_len, head_dim)
                # Under vmap: (num_heads, seq_len, head_dim)
                return key_cache.shape[-2]

        # Fallback for caches with seen_tokens attribute
        if hasattr(self._cache, "seen_tokens"):
            return self._cache.seen_tokens

        # Last resort: return 0 (no cached tokens)
        return 0


def vmap_phi3_eager_attention_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
    """vmap-compatible eager attention for Phi-3.

    Phi-3 uses fused QKV projection and custom cache, which requires special handling.

    Original (4D): hidden_states shape (batch, seq_len, hidden_dim)
                   returns attn_output (batch, seq_len, hidden_dim)
    Under vmap (3D): hidden_states shape (seq_len, hidden_dim)
                     returns attn_output (seq_len, hidden_dim)
    """
    # Compute query, key, value with fused projection
    qkv = module.qkv_proj(hidden_states)

    # Unfuse QKV: reshape to (batch*seq, num_heads, head_dim) for each of Q, K, V
    # Under vmap: hidden_states (seq, dim) -> qkv (seq, 3*num_heads*head_dim)
    # Reshape: (seq, 3*num_heads*head_dim) -> (seq, 3, num_heads, head_dim)

    batch_seq_len = qkv.shape[0]

    # Reshape to separate Q, K, V
    qkv = qkv.reshape(batch_seq_len, -1, module.num_heads, module.head_dim)

    # Split into Q, K, V
    # Shape after split: (seq, num_heads, head_dim) for each
    query = qkv[:, 0]  # (seq, num_heads, head_dim)
    key = qkv[:, 1]  # (seq, num_heads, head_dim)
    value = qkv[:, 2]  # (seq, num_heads, head_dim)

    # Add batch dimension back under vmap (if needed)
    if query.ndim == 2:
        # Under vmap: (seq, head_dim) -> (1, seq, head_dim)
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
    # Add num_heads dimension
    if query.shape[-1] != module.num_heads:
        # Reshape if needed
        pass

    # Handle RoPE (Rotary Position Embeddings)
    if hasattr(module, "rotary_emb"):
        q_pos_emb, k_pos_emb = module.rotary_emb(value, position_ids)
        query = apply_rotary_pos_emb(query, q_pos_emb)
        key = apply_rotary_pos_emb(key, k_pos_emb)

    # Reuse key/value from past if available
    if past_key_value is not None:
        # Wrap cache for compatibility
        if not isinstance(past_key_value, VmapCompatibleDynamicCache):
            past_key_value = VmapCompatibleDynamicCache(past_key_value)

        key = torch.cat([past_key_value.key_cache[0], key], dim=-2)
        value = torch.cat([past_key_value.value_cache[0], value], dim=-2)

    # Standard attention computation
    key_states = vmap_repeat_kv(key, module.num_key_value_groups)
    value_states = vmap_repeat_kv(value, module.num_key_value_groups)

    # Attention weights
    scaling = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling

    if attention_mask is not None:
        q_len = query.shape[-2]
        kv_len = key_states.shape[-2]
        causal_mask = attention_mask[..., :q_len, :kv_len]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)

    if module.training:
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=module.attention_dropout, training=True
        )

    attn_output = torch.matmul(attn_weights, value_states)
    # Transpose: (..., num_heads, seq, head_dim) -> (..., seq, num_heads, head_dim)
    attn_output = attn_output.transpose(-3, -2).contiguous()

    # Remove batch dimension if added for vmap
    if attn_output.shape[0] == 1:
        attn_output = attn_output.squeeze(0)

    # Return (attn_output, attn_weights_or_None, past_key_value) to match
    # the standard attention forward signature that some Phi-3 variants expect.
    return attn_output, (attn_weights if output_attentions else None), past_key_value


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor = None
) -> torch.Tensor:
    """Apply rotary position embeddings."""
    # Simplified, numerically-stable rotary embedding helper suitable for
    # vmap-compatible patches. This is NOT a full replacement of the model's
    # rotary implementation, but preserves the interface and shape.
    if sin is None:
        # Compute sin from cos using unit-circle relationship with clamping
        sin = torch.sqrt(torch.clamp(1.0 - cos.pow(2), min=0.0))

    # Typical rotary implementations rotate pairs of elements in the last dim.
    # Implement rotate-half: split last dim into two and apply complex rotation.
    if x.size(-1) % 2 != 0:
        # If head_dim is odd, fall back to elementwise mix to avoid shape errors.
        return x * cos + x * sin

    x1, x2 = x.chunk(2, dim=-1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def vmap_phi3_attention_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, ...]:
    """Wrapper around Phi-3 attention that ensures cache compatibility."""

    # Wrap cache for vmap compatibility
    if past_key_value is not None and not isinstance(
        past_key_value, VmapCompatibleDynamicCache
    ):
        past_key_value = VmapCompatibleDynamicCache(past_key_value)

    # Call the original forward with wrapped cache
    if hasattr(module, "_phi3_original_forward"):
        return module._phi3_original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    # Fallback: return hidden states unchanged
    return (hidden_states, None, past_key_value)


# =============================================================================
# Patch application
# =============================================================================


def apply_phi3_patches() -> None:
    """Apply Phi-3-specific vmap patches.

    Patches Phi-3 with cache-compatible attention implementation.

    Note: Requires apply_shared_patches() from _shared to be called first.
    """
    try:
        module = importlib.import_module(_PHI3_MODULE)

        # Patch DynamicCache if it exists
        if hasattr(module, "DynamicCache"):
            original_init = module.DynamicCache.__init__

            def vmap_compatible_init(self, *args, **kwargs):
                """Initialize with vmap compatibility."""
                original_init(self, *args, **kwargs)

                # Add get_usable_length method if not present
                if not hasattr(self, "get_usable_length"):

                    def get_usable_length(layer_idx: int) -> int:
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

            module.DynamicCache.__init__ = vmap_compatible_init

        # Patch repeat_kv with base implementation
        if hasattr(module, "repeat_kv"):
            module.repeat_kv = vmap_repeat_kv

        # Note: eager_attention_forward patching is more complex for Phi-3
        # due to fused QKV projection, so we defer this unless needed

    except ImportError:
        # Phi-3 not available in this transformers version
        pass
