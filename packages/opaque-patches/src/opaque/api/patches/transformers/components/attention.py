# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Model-level compatibility patching functions for vmap."""

from math import prod

import torch

from opaque.api.patches.kernels import (
    _can_use_triton_sliding_window_attention,
    _triton_sliding_window_attention,
)

_MIB = 1024**2
_ATTENTION_FALLBACK_WORKSPACE_BYTES = 256 * _MIB
_ATTENTION_MAX_WORKSPACE_BYTES = 512 * _MIB
_ATTENTION_FREE_MEMORY_FRACTION = 0.125
_ATTENTION_CHUNK_CANDIDATES = (1024, 512, 256, 128, 64, 32, 16, 1)
# At short lengths Triton's launch workspace exceeds Gemma2's dense score matrix.
_TRITON_SOFTCAP_MIN_SEQUENCE = 2048
_CUDA_SOFTCAP_FALLBACK_MAX_CHUNK = 64
# Test-only overrides; production always uses the workspace planner.
_GEMMA2_QUERY_CHUNK: int | None = None
_SDPA_QUERY_CHUNK: int | None = None


def _physical_vmap_factor(tensor: torch.Tensor) -> int:
    """Number of physical examples hidden by functorch wrappers."""
    if torch.compiler.is_compiling():
        return 1
    try:
        functorch = torch._C._functorch
        physical = tensor
        while functorch.is_functorch_wrapped_tensor(physical):
            physical = functorch.get_unwrapped(physical)
        return max(1, physical.numel() // tensor.numel())
    except (AttributeError, RuntimeError, ZeroDivisionError):
        return 1


def _attention_workspace_budget_bytes(device: torch.device) -> int:
    """Return a conservative temporary-workspace budget for attention."""
    free_bytes: int | None = None
    try:
        if device.type == "cuda":
            driver_free = int(torch.cuda.mem_get_info(device)[0])
            cached_free = max(
                0,
                int(torch.cuda.memory_reserved(device))
                - int(torch.cuda.memory_allocated(device)),
            )
            free_bytes = driver_free + cached_free
        elif device.type == "mps":
            total = int(torch.mps.recommended_max_memory())
            used = int(torch.mps.driver_allocated_memory())
            free_bytes = max(total - used, 0)
    except (AttributeError, RuntimeError, TypeError):
        free_bytes = None

    if free_bytes is None:
        return _ATTENTION_FALLBACK_WORKSPACE_BYTES
    return max(
        1,
        min(
            _ATTENTION_MAX_WORKSPACE_BYTES,
            int(free_bytes * _ATTENTION_FREE_MEMORY_FRACTION),
        ),
    )


def _attention_query_chunk_size(
    query: torch.Tensor,
    key: torch.Tensor,
    sliding_window: int,
    *,
    softcap: bool,
    bounded_keys: bool = True,
) -> int:
    """Choose the largest aligned query tile within the workspace budget."""
    query_length = query.shape[-2]
    if query_length <= 0:
        return 1

    leading = prod(query.shape[:-3]) * _physical_vmap_factor(query)
    grouped_heads = query.shape[-3] // key.shape[-3]
    score_bytes = 12 if softcap else max(4, query.element_size() * 2)
    backend_multiplier = 2 if query.device.type in {"cpu", "mps"} else 1
    budget = _attention_workspace_budget_bytes(query.device)
    candidates = _ATTENTION_CHUNK_CANDIDATES
    if softcap and query.device.type == "cuda":
        candidates = tuple(
            chunk for chunk in candidates if chunk <= _CUDA_SOFTCAP_FALLBACK_MAX_CHUNK
        )

    for chunk in candidates:
        chunk = min(chunk, query_length)
        key_extent = (
            min(key.shape[-2], sliding_window + chunk - 1)
            if bounded_keys
            else key.shape[-2]
        )
        score_workspace = (
            backend_multiplier
            * leading
            * grouped_heads
            * chunk
            * key_extent
            * score_bytes
        )
        activation_workspace = (
            leading * grouped_heads * chunk * query.shape[-1] * query.element_size() * 3
        )
        if score_workspace + activation_workspace <= budget:
            return chunk
    return 1


def _is_compact_padding_mask(
    attention_mask: torch.Tensor | None, query: torch.Tensor
) -> bool:
    """Whether ``attention_mask`` contains only per-key padding validity."""
    if attention_mask is None:
        return False
    expected_ndim = max(1, query.ndim - 2)
    return (
        attention_mask.ndim == expected_ndim
        and attention_mask.shape[-1] == query.shape[-2]
    )


def _compact_padding_slice(
    attention_mask: torch.Tensor,
    key_start: int,
    key_end: int,
) -> torch.Tensor:
    """Broadcast a compact padding mask over heads and query positions."""
    return attention_mask[..., None, None, key_start:key_end].to(torch.bool)


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


def _mask_for_query_group(
    attention_mask: torch.Tensor | None,
    query: torch.Tensor,
    query_head_start: int,
    query_head_end: int,
) -> torch.Tensor | None:
    if (
        attention_mask is not None
        and attention_mask.ndim >= query.ndim
        and attention_mask.shape[-3] == query.shape[-3]
    ):
        return attention_mask[..., query_head_start:query_head_end, :, :]
    return attention_mask


def _kv_group(
    states: torch.Tensor,
    group_index: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    state = states[..., group_index : group_index + 1, :, :]
    return state.expand(
        *state.shape[:-3], num_key_value_groups, state.shape[-2], state.shape[-1]
    )


def _grouped_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    is_causal: bool = False,
) -> torch.Tensor:
    num_key_value_groups = query.shape[-3] // key.shape[-3]
    if num_key_value_groups == 1:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )

    output_groups = []
    for key_value_head in range(key.shape[-3]):
        query_head_start = key_value_head * num_key_value_groups
        query_head_end = query_head_start + num_key_value_groups
        output_groups.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[..., query_head_start:query_head_end, :, :],
                _kv_group(key, key_value_head, num_key_value_groups),
                _kv_group(value, key_value_head, num_key_value_groups),
                attn_mask=_mask_for_query_group(
                    attention_mask, query, query_head_start, query_head_end
                ),
                dropout_p=dropout,
                scale=scaling,
                is_causal=is_causal,
            )
        )
    return torch.cat(output_groups, dim=-3)


def _can_use_native_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    is_causal: bool,
) -> bool:
    if query.device.type != "cuda" or torch._C._functorch.is_batchedtensor(query):
        return False
    params = torch.backends.cuda.SDPAParams(
        query, key, value, attention_mask, dropout, is_causal, True
    )
    return (
        torch.backends.cuda.flash_sdp_enabled()
        and torch.backends.cuda.can_use_flash_attention(params)
    )


def _apply_attention_mask(
    attn_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    q_len: int,
    kv_len: int,
) -> torch.Tensor:
    """Apply either an SDPA Boolean mask or an eager additive mask."""
    causal_mask = attention_mask[..., :q_len, :kv_len]
    if causal_mask.dtype == torch.bool:
        return attn_weights.masked_fill(
            ~causal_mask, torch.finfo(attn_weights.dtype).min
        )
    return attn_weights + causal_mask


def _chunked_sliding_window_mask(
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
    sliding_window: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Build one compact causal band for contiguous query and key chunks."""
    query_length = query_end - query_start
    key_length = key_end - key_start
    mask = torch.ones(
        (query_length, key_length), dtype=torch.bool, device=device
    ).tril_(diagonal=query_start - key_start)
    if sliding_window is not None:
        mask.triu_(diagonal=query_start - key_start - sliding_window + 1)
    return mask


def vmap_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run SDPA without flattening expanded grouped-query K/V heads."""
    num_key_value_groups = query.shape[-3] // key.shape[-3]
    if num_key_value_groups == 1:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            position_bias=position_bias,
            **kwargs,
        )

    query_length = query.shape[-2]
    key_value_length = key.shape[-2]
    is_causal = (
        is_causal if is_causal is not None else getattr(module, "is_causal", True)
    )
    is_causal = query_length > 1 and attention_mask is None and is_causal

    if is_causal and key_value_length > query_length:
        key = key[..., :query_length, :]
        value = value[..., :query_length, :]
        if position_bias is not None:
            position_bias = position_bias[..., :query_length]

    if position_bias is not None:
        from transformers.integrations.sdpa_attention import create_position_bias_mask

        attention_mask = create_position_bias_mask(
            position_bias, attention_mask, is_causal, query, key
        )
        is_causal = False

    if _can_use_native_gqa(query, key, value, attention_mask, dropout, bool(is_causal)):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            enable_gqa=True,
        )
    else:
        attn_output = _grouped_sdpa(
            query,
            key,
            value,
            attention_mask,
            dropout,
            scaling,
            bool(is_causal),
        )
    return attn_output.transpose(-3, -2).contiguous(), None


def _compact_sliding_window_sdpa(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sliding_window: int,
    dropout: float,
    scaling: float | None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a sliding-window prefill in bounded adaptive query chunks."""
    del module
    chunk_size = _SDPA_QUERY_CHUNK or _attention_query_chunk_size(
        query, key, sliding_window, softcap=False
    )
    output_chunks = []
    mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    for query_start in range(0, query.shape[-2], chunk_size):
        query_end = min(query_start + chunk_size, query.shape[-2])
        key_start = max(0, query_start - sliding_window + 1)
        query_size = query_end - query_start
        key_size = query_end - key_start
        diagonal = query_start - key_start
        cache_key = (query_size, key_size, diagonal)
        mask = mask_cache.get(cache_key)
        if mask is None:
            mask = _chunked_sliding_window_mask(
                query_start,
                query_end,
                key_start,
                query_end,
                sliding_window,
                query.device,
            )
            mask_cache[cache_key] = mask
        if attention_mask is not None:
            mask = mask & _compact_padding_slice(attention_mask, key_start, query_end)
        output_chunks.append(
            _grouped_sdpa(
                query[..., query_start:query_end, :],
                key[..., key_start:query_end, :],
                value[..., key_start:query_end, :],
                mask,
                dropout,
                scaling,
            )
        )
    return torch.cat(output_chunks, dim=-2)


def vmap_sdpa_attention_forward_sliding_window(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply compact SDPA sliding-window prefill without a full Q-by-K mask."""
    compact_padding = _is_compact_padding_mask(attention_mask, query)
    if (
        (attention_mask is None or compact_padding)
        and position_bias is None
        and sliding_window is not None
        and query.shape[-2] == key.shape[-2]
        and dropout == 0.0
        and query.shape[-2] > sliding_window
    ):
        kernel_scale = query.shape[-1] ** -0.5 if scaling is None else scaling
        padding_mask = attention_mask if compact_padding else None
        if _can_use_triton_sliding_window_attention(
            query,
            key,
            value,
            padding_mask,
            sliding_window,
            kernel_scale,
        ):
            output = _triton_sliding_window_attention(
                query,
                key,
                value,
                padding_mask,
                sliding_window,
                kernel_scale,
            )
            return output.transpose(-3, -2).contiguous(), None
        output = _compact_sliding_window_sdpa(
            module,
            query,
            key,
            value,
            sliding_window,
            dropout,
            scaling,
            attention_mask if compact_padding else None,
        )
        return output.transpose(-3, -2).contiguous(), None

    if compact_padding:
        dense_mask = _chunked_sliding_window_mask(
            0,
            query.shape[-2],
            0,
            key.shape[-2],
            sliding_window,
            query.device,
        )
        attention_mask = dense_mask & _compact_padding_slice(
            attention_mask, 0, key.shape[-2]
        )
    return vmap_sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        position_bias=position_bias,
        **kwargs,
    )


def _grouped_eager_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float,
    softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_key_value_groups = query.shape[-3] // key.shape[-3]
    output_groups = []
    weight_groups = []
    for key_value_head in range(key.shape[-3]):
        query_head_start = key_value_head * num_key_value_groups
        query_head_end = query_head_start + num_key_value_groups
        query_group = query[..., query_head_start:query_head_end, :, :]
        key_group = _kv_group(key, key_value_head, num_key_value_groups)
        value_group = _kv_group(value, key_value_head, num_key_value_groups)
        attn_weights = torch.matmul(query_group, key_group.transpose(-2, -1))
        attn_weights = attn_weights * scaling
        if softcap is not None:
            attn_weights = torch.tanh(attn_weights / softcap) * softcap

        group_mask = _mask_for_query_group(
            attention_mask, query, query_head_start, query_head_end
        )
        if group_mask is not None:
            attn_weights = _apply_attention_mask(
                attn_weights,
                group_mask,
                query_group.shape[-2],
                key_group.shape[-2],
            )

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        output_groups.append(torch.matmul(attn_weights, value_group))
        weight_groups.append(attn_weights)

    return torch.cat(output_groups, dim=-3), torch.cat(weight_groups, dim=-3)


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
    attn_output, attn_weights = _grouped_eager_attention(
        module, query, key, value, attention_mask, scaling, dropout
    )
    return attn_output.transpose(-3, -2).contiguous(), attn_weights


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
    scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
    attn_output, attn_weights = _grouped_eager_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout,
        softcap,
    )
    return attn_output.transpose(-3, -2).contiguous(), attn_weights


def _gemma2_mask_chunk(
    attention_mask: torch.Tensor,
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
) -> torch.Tensor:
    if attention_mask.shape[-2] == 1:
        return attention_mask[..., :, key_start:key_end]
    return attention_mask[..., query_start:query_end, key_start:key_end]


def _gemma2_softcap_probabilities(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    softcap: float,
    is_causal: bool,
    query_start: int,
    query_end: int,
    sliding_window: int | None,
    compact_padding: bool = False,
    compact_probability_mask: torch.Tensor | None = None,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compact_prefill = (
        (attention_mask is None or compact_padding)
        and is_causal
        and query.shape[-2] == key.shape[-2]
    )
    key_start = (
        max(0, query_start - sliding_window + 1)
        if compact_prefill and sliding_window is not None
        else 0
    )
    key_end = query_end if compact_prefill else key.shape[-2]
    query_chunk = query[..., query_start:query_end, :]
    key_chunk = key[..., key_start:key_end, :]
    logits = torch.matmul(query_chunk, key_chunk.transpose(-2, -1)) * scaling
    capped_logits = torch.tanh(logits / softcap) * softcap

    softmax_input = capped_logits
    probability_mask = None
    if compact_probability_mask is not None:
        probability_mask = compact_probability_mask
        softmax_input = capped_logits.masked_fill(
            ~probability_mask, torch.finfo(capped_logits.dtype).min
        )
    elif attention_mask is not None and not compact_padding:
        mask = _gemma2_mask_chunk(
            attention_mask, query_start, query_end, key_start, key_end
        )
        if mask.dtype == torch.bool:
            probability_mask = mask
            softmax_input = capped_logits.masked_fill(
                ~mask, torch.finfo(capped_logits.dtype).min
            )
        else:
            softmax_input = capped_logits + mask
    elif is_causal:
        probability_mask = _chunked_sliding_window_mask(
            query_start,
            query_end,
            key_start,
            key_end,
            sliding_window if compact_prefill else None,
            query.device,
        )
        if compact_padding:
            probability_mask = probability_mask & _compact_padding_slice(
                attention_mask, key_start, key_end
            )
        softmax_input = capped_logits.masked_fill(
            ~probability_mask, torch.finfo(capped_logits.dtype).min
        )
    elif compact_padding:
        probability_mask = _compact_padding_slice(attention_mask, key_start, key_end)
        softmax_input = capped_logits.masked_fill(
            ~probability_mask, torch.finfo(capped_logits.dtype).min
        )

    probabilities = torch.nn.functional.softmax(
        softmax_input, dim=-1, dtype=torch.float32
    )
    if probability_mask is not None:
        probabilities = probabilities.masked_fill(~probability_mask, 0.0)
    return key_start, key_end, query_chunk, key_chunk, capped_logits, probabilities


class _ChunkedGemma2Attention(torch.autograd.Function):
    """Softcapped GQA without retaining expanded K/V or full score matrices."""

    generate_vmap_rule = True

    @staticmethod
    def forward(
        query,
        key,
        value,
        attention_mask,
        scaling,
        softcap,
        is_causal,
        sliding_window,
        num_key_value_groups,
        compact_padding,
        query_chunk_size,
    ):
        output_chunks = []
        mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        for query_start in range(0, query.shape[-2], query_chunk_size):
            query_end = min(query_start + query_chunk_size, query.shape[-2])
            compact_probability_mask = None
            if (
                is_causal
                and (attention_mask is None or compact_padding)
                and query.shape[-2] == key.shape[-2]
            ):
                key_start = (
                    max(0, query_start - sliding_window + 1)
                    if sliding_window is not None
                    else 0
                )
                cache_key = (
                    query_end - query_start,
                    query_end - key_start,
                    query_start - key_start,
                )
                compact_probability_mask = mask_cache.get(cache_key)
                if compact_probability_mask is None:
                    compact_probability_mask = _chunked_sliding_window_mask(
                        query_start,
                        query_end,
                        key_start,
                        query_end,
                        sliding_window,
                        query.device,
                    )
                    mask_cache[cache_key] = compact_probability_mask
                if compact_padding:
                    compact_probability_mask = (
                        compact_probability_mask
                        & _compact_padding_slice(attention_mask, key_start, query_end)
                    )
            output_groups = []
            for key_value_head in range(key.shape[-3]):
                query_head_start = key_value_head * num_key_value_groups
                query_head_end = query_head_start + num_key_value_groups
                query_group = query[..., query_head_start:query_head_end, :, :]
                key_group = _kv_group(key, key_value_head, num_key_value_groups)
                value_group = _kv_group(value, key_value_head, num_key_value_groups)
                group_mask = _mask_for_query_group(
                    attention_mask, query, query_head_start, query_head_end
                )
                key_start, key_end, _, _, _, probabilities = (
                    _gemma2_softcap_probabilities(
                        query_group,
                        key_group,
                        group_mask,
                        scaling,
                        softcap,
                        is_causal,
                        query_start,
                        query_end,
                        sliding_window,
                        compact_padding,
                        compact_probability_mask,
                    )
                )
                output_groups.append(
                    torch.matmul(
                        probabilities.to(query.dtype),
                        value_group[..., key_start:key_end, :],
                    )
                )
            output_chunks.append(torch.cat(output_groups, dim=-3))
        return torch.cat(output_chunks, dim=-2)

    @staticmethod
    def setup_context(ctx, inputs, output):
        del output
        (
            query,
            key,
            value,
            attention_mask,
            scaling,
            softcap,
            is_causal,
            sliding_window,
            num_key_value_groups,
            compact_padding,
            query_chunk_size,
        ) = inputs
        if attention_mask is None:
            ctx.save_for_backward(query, key, value)
        else:
            ctx.save_for_backward(query, key, value, attention_mask)
        ctx.has_attention_mask = attention_mask is not None
        ctx.scaling = float(scaling)
        ctx.softcap = float(softcap)
        ctx.is_causal = is_causal
        ctx.sliding_window = sliding_window
        ctx.num_key_value_groups = num_key_value_groups
        ctx.compact_padding = compact_padding
        ctx.query_chunk_size = query_chunk_size

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_attention_mask:
            query, key, value, attention_mask = ctx.saved_tensors
        else:
            query, key, value = ctx.saved_tensors
            attention_mask = None

        grad_query_chunks = []
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        for query_start in range(0, query.shape[-2], ctx.query_chunk_size):
            query_end = min(query_start + ctx.query_chunk_size, query.shape[-2])
            compact_probability_mask = None
            if (
                ctx.is_causal
                and (attention_mask is None or ctx.compact_padding)
                and query.shape[-2] == key.shape[-2]
            ):
                key_start = (
                    max(0, query_start - ctx.sliding_window + 1)
                    if ctx.sliding_window is not None
                    else 0
                )
                cache_key = (
                    query_end - query_start,
                    query_end - key_start,
                    query_start - key_start,
                )
                compact_probability_mask = mask_cache.get(cache_key)
                if compact_probability_mask is None:
                    compact_probability_mask = _chunked_sliding_window_mask(
                        query_start,
                        query_end,
                        key_start,
                        query_end,
                        ctx.sliding_window,
                        query.device,
                    )
                    mask_cache[cache_key] = compact_probability_mask
                if ctx.compact_padding:
                    compact_probability_mask = (
                        compact_probability_mask
                        & _compact_padding_slice(attention_mask, key_start, query_end)
                    )
            grad_query_groups = []
            for key_value_head in range(key.shape[-3]):
                query_head_start = key_value_head * ctx.num_key_value_groups
                query_head_end = query_head_start + ctx.num_key_value_groups
                query_group = query[..., query_head_start:query_head_end, :, :]
                key_group = _kv_group(key, key_value_head, ctx.num_key_value_groups)
                value_group = _kv_group(value, key_value_head, ctx.num_key_value_groups)
                group_mask = _mask_for_query_group(
                    attention_mask, query, query_head_start, query_head_end
                )
                (
                    key_start,
                    key_end,
                    query_chunk,
                    key_chunk,
                    capped_logits,
                    probabilities,
                ) = _gemma2_softcap_probabilities(
                    query_group,
                    key_group,
                    group_mask,
                    ctx.scaling,
                    ctx.softcap,
                    ctx.is_causal,
                    query_start,
                    query_end,
                    ctx.sliding_window,
                    ctx.compact_padding,
                    compact_probability_mask,
                )
                grad_output_chunk = grad_output[
                    ...,
                    query_head_start:query_head_end,
                    query_start:query_end,
                    :,
                ]
                probabilities_input_dtype = probabilities.to(query.dtype)

                grad_value_group = torch.matmul(
                    probabilities_input_dtype.transpose(-2, -1), grad_output_chunk
                )
                grad_value[
                    ..., key_value_head : key_value_head + 1, key_start:key_end, :
                ].add_(grad_value_group.sum(dim=-3, keepdim=True))
                grad_probabilities = torch.matmul(
                    grad_output_chunk,
                    value_group[..., key_start:key_end, :].transpose(-2, -1),
                ).to(probabilities.dtype)
                grad_softmax_input = probabilities * (
                    grad_probabilities
                    - (grad_probabilities * probabilities).sum(dim=-1, keepdim=True)
                )
                grad_capped_logits = grad_softmax_input.to(capped_logits.dtype)
                if (
                    group_mask is not None
                    and group_mask.dtype == torch.bool
                    and not ctx.compact_padding
                ):
                    mask = _gemma2_mask_chunk(
                        group_mask, query_start, query_end, key_start, key_end
                    )
                    grad_capped_logits = grad_capped_logits.masked_fill(~mask, 0.0)
                grad_logits = grad_capped_logits * (
                    1.0 - (capped_logits / ctx.softcap) ** 2
                )
                grad_logits = grad_logits * ctx.scaling

                grad_query_groups.append(torch.matmul(grad_logits, key_chunk))
                grad_key_group = torch.matmul(
                    grad_logits.transpose(-2, -1), query_chunk
                )
                grad_key[
                    ..., key_value_head : key_value_head + 1, key_start:key_end, :
                ].add_(grad_key_group.sum(dim=-3, keepdim=True))

            grad_query_chunks.append(torch.cat(grad_query_groups, dim=-3))

        return (
            torch.cat(grad_query_chunks, dim=-2),
            grad_key,
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def vmap_sdpa_attention_forward_gemma2(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    sliding_window: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use chunked attention for Gemma2 softcap, which SDPA cannot express."""
    if softcap is not None:
        scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
        compact_padding = _is_compact_padding_mask(attention_mask, query)
        is_causal = kwargs.get("is_causal")
        if is_causal is None:
            is_causal = getattr(module, "is_causal", True)
        is_causal = (
            query.shape[-2] > 1
            and (attention_mask is None or compact_padding)
            and is_causal
        )

        if is_causal and key.shape[-2] > query.shape[-2]:
            key = key[..., : query.shape[-2], :]
            value = value[..., : query.shape[-2], :]

        if not module.training or dropout == 0.0:
            kernel_window = sliding_window or query.shape[-2]
            padding_mask = attention_mask if compact_padding else None
            if (
                is_causal
                and query.shape[-2] >= _TRITON_SOFTCAP_MIN_SEQUENCE
                and _can_use_triton_sliding_window_attention(
                    query,
                    key,
                    value,
                    padding_mask,
                    kernel_window,
                    scaling,
                    softcap,
                )
            ):
                attn_output = _triton_sliding_window_attention(
                    query,
                    key,
                    value,
                    padding_mask,
                    kernel_window,
                    scaling,
                    softcap,
                )
                return attn_output.transpose(-3, -2).contiguous(), None
            query_chunk_size = _GEMMA2_QUERY_CHUNK or _attention_query_chunk_size(
                query,
                key,
                sliding_window or query.shape[-2],
                softcap=True,
                bounded_keys=(
                    is_causal
                    and query.shape[-2] == key.shape[-2]
                    and sliding_window is not None
                ),
            )
            attn_output = _ChunkedGemma2Attention.apply(
                query,
                key,
                value,
                attention_mask,
                scaling,
                softcap,
                is_causal,
                sliding_window,
                query.shape[-3] // key.shape[-3],
                compact_padding,
                query_chunk_size,
            )
            return attn_output.transpose(-3, -2).contiguous(), None

        if is_causal:
            causal_mask = _chunked_sliding_window_mask(
                0,
                query.shape[-2],
                0,
                key.shape[-2],
                sliding_window,
                query.device,
            )
            if compact_padding:
                causal_mask = causal_mask & _compact_padding_slice(
                    attention_mask, 0, key.shape[-2]
                )
            attention_mask = causal_mask
        return vmap_eager_attention_forward_gemma2(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            softcap=softcap,
            **kwargs,
        )

    return vmap_sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


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
                if hasattr(self, "key_cache") and len(self.key_cache) > layer_idx:
                    kc = self.key_cache[layer_idx]
                    if kc is not None:
                        return kc.shape[-2]
                if hasattr(self, "seen_tokens"):
                    return self.seen_tokens
                return 0

            self.get_usable_length = get_usable_length

    return vmap_compatible_init
