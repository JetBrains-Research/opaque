# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Base classes and common vmap-compatible implementations.

This module contains:
1. BasePatcher - abstract base class for model-specific patchers
2. Common vmap-compatible function implementations shared across models
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch


class BasePatcher(ABC):
    """Abstract base class for model-specific patchers.

    Each supported model architecture should have a patcher that inherits from this class.
    The patcher is responsible for:
    1. Storing original implementations
    2. Applying vmap-compatible patches
    3. Restoring original implementations on unpatch
    """

    # Set by subclasses
    architecture_name: str = ""
    transformers_module_path: str = ""  # e.g., "transformers.models.llama.modeling_llama"

    def __init__(self):
        self._original_implementations: dict[str, Any] = {}
        self._is_patched: bool = False

    @property
    def is_patched(self) -> bool:
        return self._is_patched

    def patch(self) -> None:
        """Apply all patches for this architecture."""
        if self._is_patched:
            return

        module = self._get_module()
        if module is None:
            return

        self._patch_module(module)
        self._is_patched = True

    def unpatch(self) -> None:
        """Restore all original implementations."""
        if not self._is_patched:
            return

        module = self._get_module()
        if module is None:
            return

        self._unpatch_module(module)
        self._original_implementations.clear()
        self._is_patched = False

    def _get_module(self):
        """Import and return the transformers module for this architecture."""
        try:
            import importlib
            return importlib.import_module(self.transformers_module_path)
        except ImportError:
            return None

    @abstractmethod
    def _patch_module(self, module) -> None:
        """Apply patches to the module. Implemented by subclasses."""
        pass

    @abstractmethod
    def _unpatch_module(self, module) -> None:
        """Restore original implementations. Implemented by subclasses."""
        pass

    def _store_and_patch(self, module, attr_name: str, new_impl) -> None:
        """Helper to store original and apply patch."""
        if hasattr(module, attr_name):
            key = f"{self.architecture_name}_{attr_name}"
            if key not in self._original_implementations:
                self._original_implementations[key] = getattr(module, attr_name)
                setattr(module, attr_name, new_impl)

    def _restore(self, module, attr_name: str) -> None:
        """Helper to restore original implementation."""
        key = f"{self.architecture_name}_{attr_name}"
        if key in self._original_implementations:
            setattr(module, attr_name, self._original_implementations[key])
            del self._original_implementations[key]


# =============================================================================
# Common vmap-compatible implementations
# =============================================================================


def vmap_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """vmap-compatible version of repeat_kv.

    The original uses hardcoded 4D unpacking which fails under vmap.
    This version uses dynamic shape handling.
    """
    if n_rep == 1:
        return hidden_states

    # Dynamic shape: (..., num_kv_heads, seq_len, head_dim)
    *leading_dims, num_kv_heads, slen, head_dim = hidden_states.shape

    hidden_states = hidden_states.unsqueeze(-3)
    expand_shape = (*leading_dims, num_kv_heads, n_rep, slen, head_dim)
    hidden_states = hidden_states.expand(*expand_shape)

    new_shape = (*leading_dims, num_kv_heads * n_rep, slen, head_dim)
    return hidden_states.reshape(*new_shape)


def vmap_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vmap-compatible version of eager_attention_forward.

    Uses negative indexing for transposes to handle arbitrary batch dimensions.
    """
    key_states = vmap_repeat_kv(key, module.num_key_value_groups)
    value_states = vmap_repeat_kv(value, module.num_key_value_groups)

    # Negative indexing works with any number of leading dimensions
    attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(-3, -2).contiguous()

    return attn_output, attn_weights


def vmap_create_causal_mask(
    config,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values=None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """vmap-compatible version of create_causal_mask.

    The original uses expand() with hardcoded 4D shapes. This version
    uses dynamic shapes that work with vmap's additional batch dimension.
    """
    dtype = input_tensor.dtype
    device = input_tensor.device
    q_length = input_tensor.shape[-2]  # seq is second-to-last dim
    kv_length = q_length

    if past_key_values is not None:
        kv_length = past_key_values.get_seq_length() + q_length

    # Create base causal mask (q_length, kv_length)
    causal_mask = torch.tril(
        torch.ones(q_length, kv_length, dtype=torch.bool, device=device),
        diagonal=kv_length - q_length,
    )

    # Handle 2D padding mask
    if attention_mask is not None and attention_mask.dim() == 2:
        if attention_mask.shape[-1] < kv_length:
            pad_len = kv_length - attention_mask.shape[-1]
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=1)

        padding_mask = attention_mask[..., :kv_length].bool()
        padding_mask = padding_mask.unsqueeze(-2).unsqueeze(-2)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        combined_mask = causal_mask & padding_mask
    else:
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Convert to float mask: True -> 0.0, False -> -inf
    float_mask = torch.where(
        combined_mask,
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(float("-inf"), device=device, dtype=dtype),
    )

    return float_mask
