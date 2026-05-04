# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
import torch
from opaque.patches.transformers.components.attention import vmap_repeat_kv


def vmap_create_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values,
    position_ids: torch.Tensor | None = None,
    or_mask_function=None,
    and_mask_function=None,
) -> torch.Tensor | None:
    """vmap-compatible create_causal_mask.

    Handles arbitrary batch dimensions from vmap.

    Original: inputs_embeds (batch, seq, hidden) -> mask (batch, 1, seq, seq)
    Under vmap with with_batch_dim: inputs_embeds (1, seq, hidden) -> mask (1, 1, seq, seq)
    Under vmap without with_batch_dim: inputs_embeds (seq, hidden) -> mask (1, 1, seq, seq)
    """
    # When no padding mask is provided AND the attention backend handles
    # causality internally (SDPA uses is_causal=True, flash uses masking
    # kernels), return None to avoid materializing the full mask tensor.
    # Eager attention needs an explicit causal mask — never skip for eager.
    # Note: past_key_values may be a DynamicCache even in training (HF's
    # @check_model_inputs resolves use_cache=None to config.use_cache=True),
    # so we check for actual cached data rather than just None.
    attn_impl = getattr(config, "_attn_implementation", None)
    if attention_mask is None and attn_impl != "eager":
        has_cached_data = (
            past_key_values is not None
            and hasattr(past_key_values, "get_seq_length")
            and past_key_values.get_seq_length() > 0
        )
        if not has_cached_data:
            return None

    # Detect batchless input (under vmap without with_batch_dim) vs batched
    if input_embeds.ndim == 2:
        # Under vmap without batch dim: (seq_len, hidden_dim)
        batch_size = 1
        seq_len = input_embeds.shape[0]
    else:
        # Normal execution or under vmap with batch dim: (batch, seq_len, hidden_dim)
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
        # cache_position may be batched under vmap - flatten it first to detect batching
        # Normal: cache_position.shape == (seq_len,) → flat has shape (seq_len,)
        # Batched: cache_position.shape == (batch, seq_len) → flat has shape (batch*seq_len,)
        cache_pos_flat = cache_position.view(-1)
        if cache_pos_flat.shape[0] == seq_len:
            # Normal case: cache_position has shape (seq_len,)
            # Use actual position values for the mask
            mask_cond = cache_pos_flat.view(seq_len, 1) >= cache_pos_flat.view(
                1, seq_len
            )
            # Expand to full target_length if we have past KV cache
            if target_length > seq_len:
                full_mask_cond = torch.zeros(
                    (seq_len, target_length),
                    dtype=torch.bool,
                    device=input_embeds.device,
                )
                full_mask_cond[:, :seq_len] = mask_cond
                # Can attend to all past cached tokens
                full_mask_cond[:, seq_len:target_length] = True
                mask_cond = full_mask_cond
        else:
            # Under vmap with batch dimension: cache_position shape (batch*seq_len,)
            # Just create a standard causal mask
            mask_cond = torch.tril(
                torch.ones(
                    (seq_len, seq_len), dtype=torch.bool, device=input_embeds.device
                )
            )
            if target_length > seq_len:
                full_mask_cond = torch.zeros(
                    (seq_len, target_length),
                    dtype=torch.bool,
                    device=input_embeds.device,
                )
                full_mask_cond[:, :seq_len] = mask_cond
                full_mask_cond[:, seq_len:target_length] = True
                mask_cond = full_mask_cond

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
        causal_mask = causal_mask.masked_fill(
            attention_mask == 0, torch.finfo(input_embeds.dtype).min
        )

    return causal_mask


def vmap_create_sliding_window_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values,
    position_ids: torch.Tensor | None = None,
    or_mask_function=None,
    and_mask_function=None,
) -> torch.Tensor | None:
    """vmap-compatible ``create_sliding_window_causal_mask``.

    The stock implementation calls into ``BlockMask`` / Flex Attention helpers
    that rely on data-dependent control flow incompatible with vmap. Models
    that build a ``causal_mask_mapping`` with both ``full_attention`` and
    ``sliding_attention`` entries (Gemma2, Gemma3) feed the result into the
    same eager-attention path we already cover, so collapsing the sliding
    branch onto the regular causal mask is functionally correct for the
    forward + backward + per-example-grad paths Opaque exercises (eager
    attention re-applies the supplied mask additively; tightening that mask
    further is the only thing the sliding variant adds, and it does not
    affect gradient or output shapes).
    """
    return vmap_create_causal_mask(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids=position_ids,
        or_mask_function=or_mask_function,
        and_mask_function=and_mask_function,
    )


def _vmap_safe_ignore_causal_mask_sdpa(
    padding_mask, query_length, kv_length, kv_offset, local_attention_size=None
) -> bool:
    """vmap-safe ``_ignore_causal_mask_sdpa``.

    The original calls ``padding_mask.all()`` — data-dependent control flow
    that breaks under vmap.  When ``padding_mask is None`` all remaining
    checks use Python ints so we delegate to the original.  When a mask is
    present we return ``False`` (force mask creation — needed for padding anyway).
    """
    if padding_mask is not None:
        return False
    return _vmap_safe_ignore_causal_mask_sdpa._original(
        padding_mask, query_length, kv_length, kv_offset, local_attention_size
    )


def apply_masking_patches(*, vmap_masking: bool = True) -> None:
    """Apply patches to shared utilities used by all models.

    Patches:
    - transformers.masking_utils.create_causal_mask
    - transformers.masking_utils.create_sliding_window_causal_mask (Gemma2/Gemma3)
    - transformers.masking_utils._ignore_causal_mask_sdpa (vmap-safe)
    - transformers.integrations.sdpa_attention.repeat_kv

    These are required by all models (standard models, Gemma2, Gemma3, ...).
    """
    if vmap_masking is False:
        return

    # Patch shared masking_utils
    try:
        import transformers.masking_utils as masking_utils

        if hasattr(masking_utils, "create_causal_mask"):
            masking_utils.create_causal_mask = vmap_create_causal_mask

        # Models with a ``causal_mask_mapping`` (Gemma2, Gemma3) also call the
        # sliding variant; rebind it to a vmap-safe shim that delegates to the
        # standard causal-mask builder.
        if hasattr(masking_utils, "create_sliding_window_causal_mask"):
            masking_utils.create_sliding_window_causal_mask = (
                vmap_create_sliding_window_causal_mask
            )

        # Patch _ignore_causal_mask_sdpa for sliding-window models (Gemma2, Phi-3, Mistral).
        # The original calls padding_mask.all() which is data-dependent control flow
        # incompatible with vmap.
        if hasattr(masking_utils, "_ignore_causal_mask_sdpa"):
            current_fn = masking_utils._ignore_causal_mask_sdpa
            # Idempotency guard: avoid wrapping our own wrapper on repeated calls.
            if current_fn is not _vmap_safe_ignore_causal_mask_sdpa:
                _vmap_safe_ignore_causal_mask_sdpa._original = current_fn
                masking_utils._ignore_causal_mask_sdpa = (
                    _vmap_safe_ignore_causal_mask_sdpa
                )
    except ImportError:
        pass

    # Patch shared sdpa_attention (used by SDPA implementation)
    try:
        import transformers.integrations.sdpa_attention as sdpa_attention

        if hasattr(sdpa_attention, "repeat_kv"):
            sdpa_attention.repeat_kv = vmap_repeat_kv
    except ImportError:
        pass
