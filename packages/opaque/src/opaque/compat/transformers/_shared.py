# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared vmap-compatible utilities for HuggingFace transformers.

These patches are used by all models (standard and custom).
"""

import torch


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


# =============================================================================
# Batchify wrapper for model forward methods
# =============================================================================


def _disable_kv_cache(forward_fn):
    """Wrap forward to disable KV cache when no existing cache is passed.

    HuggingFace models default to ``config.use_cache = True``, which creates
    a ``DynamicCache`` on every forward pass — even during training where it
    is never used.  Under ``vmap``, these cache objects leak memory due to
    circular references (cache → tensors → autograd metadata → cache) that
    defeat Python's reference counting.

    This wrapper forces ``use_cache=False`` when ``past_key_values`` is
    ``None`` or an empty cache, matching the approach used by Unsloth
    (``if past_key_values is None and self.training: use_cache = False``).
    Unlike Unsloth, we don't gate on ``self.training`` to avoid side effects
    with LoRA modules — the condition ``past_key_values is None`` is
    sufficient since the cache is only useful for autoregressive generation
    with an existing cache.
    """
    import functools

    if getattr(forward_fn, "_opaque_cache_disabled", False):
        return forward_fn

    @functools.wraps(forward_fn)
    def wrapper(*args, **kwargs):
        past = kwargs.get("past_key_values")
        has_cached_data = past is not None and (
            not hasattr(past, "get_seq_length") or past.get_seq_length() > 0
        )
        if not has_cached_data:
            kwargs["use_cache"] = False
        return forward_fn(*args, **kwargs)

    wrapper._opaque_cache_disabled = True
    return wrapper


def _batchify_forward(original_forward):
    """Wrap a model ``forward`` to handle batchless inputs under vmap.

    Under ``vmap(grad(...))``, per-example inputs lack the batch dimension:
    ``input_ids`` is 1-D ``(seq,)`` instead of 2-D ``(batch, seq)``.
    HuggingFace models universally assume batched inputs, so this wrapper
    adds the batch dimension on entry and strips it on exit.

    This is analogous to PyTorch's *batchify* pattern in ``attention.cpp``
    for unbatched SDPA inputs.

    The wrapper is a no-op for normal (already-batched) inputs.

    Delegates to :func:`opaque.utils.functional.with_batch_dim`.
    """
    from opaque.utils.functional import with_batch_dim

    return with_batch_dim(
        original_forward,
        batch_kwargs={
            "input_ids": 2,
            "attention_mask": 2,
            "labels": 2,
            "position_ids": 2,
            "inputs_embeds": 3,
        },
        min_ndim=2,
    )


# =============================================================================
# Patch application
# =============================================================================

# All HF model modules that may contain CausalLM classes
_ALL_MODEL_MODULES = [
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.qwen3.modeling_qwen3",
    "transformers.models.phi3.modeling_phi3",
    "transformers.models.gemma.modeling_gemma",
    "transformers.models.gemma2.modeling_gemma2",
    "transformers.models.granite.modeling_granite",
    "transformers.models.cohere.modeling_cohere",
    "transformers.models.cohere2.modeling_cohere2",
    "transformers.models.gpt2.modeling_gpt2",
]


def _patch_causal_lm_classes(patch_fn) -> None:
    """Apply a patch function to all CausalLM/LMHeadModel forward methods."""
    import importlib

    for module_path in _ALL_MODEL_MODULES:
        try:
            module = importlib.import_module(module_path)
            for name in dir(module):
                if not (name.endswith("ForCausalLM") or name.endswith("LMHeadModel")):
                    continue
                cls = getattr(module, name)
                if isinstance(cls, type) and hasattr(cls, "forward"):
                    cls.forward = patch_fn(cls.forward)
        except (ImportError, RuntimeError):
            pass

    try:
        import peft

        for name in dir(peft):
            cls = getattr(peft, name, None)
            if (
                isinstance(cls, type)
                and hasattr(cls, "forward")
                and name.startswith("PeftModel")
            ):
                cls.forward = patch_fn(cls.forward)
    except ImportError:
        pass


def apply_kv_cache_patches() -> None:
    """Disable KV cache in model forward methods.

    Skippable via ``OPAQUE_SKIP_TRANSFORMERS_PATCHES=kv_cache``.
    """
    _patch_causal_lm_classes(_disable_kv_cache)


def apply_batchify_patches() -> None:
    """Apply batchify wrappers to all supported model classes.

    This must run AFTER both vmap patches, kernel patches, and kv_cache
    patches, because it must wrap the *final* version of forward.

    Skippable via ``OPAQUE_SKIP_TRANSFORMERS_PATCHES=batchify``.
    """
    _patch_causal_lm_classes(_batchify_forward)


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


def apply_shared_patches() -> None:
    """Apply patches to shared utilities used by all models.

    Patches:
    - transformers.masking_utils.create_causal_mask
    - transformers.masking_utils._ignore_causal_mask_sdpa (vmap-safe)
    - transformers.integrations.sdpa_attention.repeat_kv

    These are required by all models (standard models, Gemma2, etc.).
    """
    # Patch shared masking_utils
    try:
        import transformers.masking_utils as masking_utils

        if hasattr(masking_utils, "create_causal_mask"):
            masking_utils.create_causal_mask = vmap_create_causal_mask

        # Patch _ignore_causal_mask_sdpa for sliding-window models (Gemma2, Phi-3, Mistral).
        # The original calls padding_mask.all() which is data-dependent control flow
        # incompatible with vmap.
        if hasattr(masking_utils, "_ignore_causal_mask_sdpa"):
            _vmap_safe_ignore_causal_mask_sdpa._original = (
                masking_utils._ignore_causal_mask_sdpa
            )
            masking_utils._ignore_causal_mask_sdpa = _vmap_safe_ignore_causal_mask_sdpa
    except ImportError:
        pass

    # Patch shared sdpa_attention (used by SDPA implementation)
    try:
        import transformers.integrations.sdpa_attention as sdpa_attention

        if hasattr(sdpa_attention, "repeat_kv"):
            sdpa_attention.repeat_kv = vmap_repeat_kv
    except ImportError:
        pass
