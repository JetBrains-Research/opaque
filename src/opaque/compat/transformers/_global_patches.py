# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Global patches for shared transformers modules.

These patches are applied once at import time and affect all model architectures
that use these shared utilities.
"""

from typing import Any, Optional
import torch

# Store originals for unpatch
_original_implementations: dict[str, Any] = {}
_is_patched = False


def vmap_create_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values=None,
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function=None,
    and_mask_function=None,
) -> Optional[torch.Tensor]:
    """vmap-compatible version of create_causal_mask.

    The original has data-dependent control flow that breaks under vmap.
    This version creates masks without conditional branches on tensor values.

    Signature matches transformers.masking_utils.create_causal_mask
    """
    dtype = input_embeds.dtype
    device = input_embeds.device

    # Get sequence length from input - use negative indexing for vmap compatibility
    # input_embeds shape: (..., seq_len, hidden_size)
    q_length = input_embeds.shape[-2]
    kv_length = q_length

    if past_key_values is not None:
        kv_length = past_key_values.get_seq_length() + q_length

    # Create base causal mask (q_length, kv_length)
    causal_mask = torch.tril(
        torch.ones(q_length, kv_length, dtype=torch.bool, device=device),
        diagonal=kv_length - q_length,
    )

    # Handle 2D padding mask from attention_mask
    if attention_mask is not None and attention_mask.dim() == 2:
        # Pad if needed
        if attention_mask.shape[-1] < kv_length:
            pad_len = kv_length - attention_mask.shape[-1]
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=1)

        # Combine padding mask with causal mask
        padding_mask = attention_mask[..., :kv_length].bool()
        padding_mask = padding_mask.unsqueeze(-2).unsqueeze(-2)  # (..., 1, 1, kv_length)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, q_length, kv_length)
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


def apply_global_patches() -> None:
    """Apply global patches to shared transformers modules."""
    global _is_patched

    if _is_patched:
        return

    try:
        import transformers.masking_utils as masking_utils

        # Patch create_causal_mask
        if hasattr(masking_utils, "create_causal_mask"):
            _original_implementations["masking_utils_create_causal_mask"] = (
                masking_utils.create_causal_mask
            )
            masking_utils.create_causal_mask = vmap_create_causal_mask

        _is_patched = True
    except ImportError:
        # transformers not installed
        pass


def remove_global_patches() -> None:
    """Remove global patches from shared transformers modules."""
    global _is_patched

    if not _is_patched:
        return

    try:
        import transformers.masking_utils as masking_utils

        if "masking_utils_create_causal_mask" in _original_implementations:
            masking_utils.create_causal_mask = _original_implementations[
                "masking_utils_create_causal_mask"
            ]
            del _original_implementations["masking_utils_create_causal_mask"]

        _is_patched = False
    except ImportError:
        pass


def is_globally_patched() -> bool:
    """Check if global patches are applied."""
    return _is_patched
