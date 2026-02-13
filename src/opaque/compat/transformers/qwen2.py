# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Qwen2 model patcher for vmap compatibility.

Qwen2 has additional issues beyond repeat_kv:
- Attention forward uses hardcoded shape calculations
"""

from typing import Optional
import torch

from opaque.compat.transformers.base import BasePatcher, vmap_repeat_kv
from opaque.compat.transformers.registry import register_patcher


def _make_vmap_compatible_qwen2_attention_forward(original_forward):
    """Create a vmap-compatible attention forward that wraps the original.

    Qwen2Attention.forward uses hardcoded shape calculations that break under vmap.
    This wrapper handles the dynamic batch dimensions.
    """

    def vmap_qwen2_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Get shape dynamically - works with any number of leading dims
        *leading_dims, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Get head dimensions from config
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.head_dim

        # Reshape with dynamic leading dims
        query_states = query_states.view(*leading_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
        key_states = key_states.view(*leading_dims, seq_len, num_kv_heads, head_dim).transpose(-3, -2)
        value_states = value_states.view(*leading_dims, seq_len, num_kv_heads, head_dim).transpose(-3, -2)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Handle KV cache if present
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat KV for GQA (vmap-compatible)
        num_key_value_groups = num_heads // num_kv_heads
        key_states = vmap_repeat_kv(key_states, num_key_value_groups)
        value_states = vmap_repeat_kv(value_states, num_key_value_groups)

        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back - transpose and merge heads
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*leading_dims, seq_len, num_heads * head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return vmap_qwen2_attention_forward


@register_patcher("qwen2")
class Qwen2Patcher(BasePatcher):
    """Patcher for Qwen2 models."""

    architecture_name = "qwen2"
    transformers_module_path = "transformers.models.qwen2.modeling_qwen2"

    def _patch_module(self, module) -> None:
        """Apply Qwen2-specific patches."""
        # Patch repeat_kv
        self._store_and_patch(module, "repeat_kv", vmap_repeat_kv)

        # Patch Qwen2Attention.forward - it has hardcoded shape calculations
        if hasattr(module, "Qwen2Attention"):
            attention_cls = module.Qwen2Attention
            if hasattr(attention_cls, "forward"):
                original_forward = attention_cls.forward
                key = f"{self.architecture_name}_Qwen2Attention_forward"
                if key not in self._original_implementations:
                    self._original_implementations[key] = original_forward
                    attention_cls.forward = _make_vmap_compatible_qwen2_attention_forward(
                        original_forward
                    )

    def _unpatch_module(self, module) -> None:
        """Restore original implementations."""
        self._restore(module, "repeat_kv")

        # Restore Qwen2Attention.forward
        key = f"{self.architecture_name}_Qwen2Attention_forward"
        if key in self._original_implementations:
            if hasattr(module, "Qwen2Attention"):
                module.Qwen2Attention.forward = self._original_implementations[key]
            del self._original_implementations[key]
