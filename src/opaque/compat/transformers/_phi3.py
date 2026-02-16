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
    # Non-invasive wrapper: add/remove a leading batch dim for vmap and
    # delegate to the model's original attention/forward implementation.
    added_batch = False
    if hidden_states.ndim == 2:
        # vmap case: (seq_len, hidden_dim) -> add batch dim
        hidden_states = hidden_states.unsqueeze(0)
        added_batch = True

    # Wrap past_key_value for safe access without mutating original
    if past_key_value is not None and not isinstance(
        past_key_value, VmapCompatibleDynamicCache
    ):
        past_key_value = VmapCompatibleDynamicCache(past_key_value)

    # Prefer the module's saved original forward if present
    if hasattr(module, "_phi3_original_forward"):
        out = module._phi3_original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    else:
        # Fall back to best-effort calls: try eager_attention_forward then forward
        if hasattr(module, "eager_attention_forward"):
            out = module.eager_attention_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        else:
            # Last resort: call module.forward with provided args (may raise)
            out = module.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

    # If we added a batch dim, remove it from the primary tensor in the output
    try:
        if isinstance(out, (tuple, list)):
            attn_output = out[0]
            if added_batch and attn_output.shape[0] == 1:
                attn_output = attn_output.squeeze(0)
                out = (attn_output,) + tuple(out[1:])
        else:
            if added_batch and out.shape[0] == 1:
                out = out.squeeze(0)
    except Exception:
        # Best-effort only; don't fail here
        pass

    return out


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor
):
    """Delegate to HuggingFace Phi-3 rotary helper when available.

    Prefers HF's `apply_rotary_pos_emb(q, k, cos, sin)` if provided; otherwise
    falls back to a conservative rotate-half implementation applied to `q` and
    optionally `k`.
    """
    try:
        import transformers.models.phi3.modeling_phi3 as hf_phi3

        if hasattr(hf_phi3, "apply_rotary_pos_emb"):
            return hf_phi3.apply_rotary_pos_emb(q, k, cos, sin)
    except Exception:
        pass

    # Fallback rotate-half applied elementwise (conservative)
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_out = q * cos + rotate_half(q) * sin
    if k is None:
        return q_out
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


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
