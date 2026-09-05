# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Model-level compatibility patching functions for vmap."""

import torch

# Bound the largest temporary to (..., heads, 64, key_length), rather than
# materializing the full (..., heads, query_length, key_length) matrix.
_GEMMA2_QUERY_CHUNK = 64


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
        q_len = query.shape[-2]
        kv_len = key_states.shape[-2]
        attn_weights = _apply_attention_mask(
            attn_weights, attention_mask, q_len, kv_len
        )

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
    scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
    attn_weights = torch.matmul(query, key_states.transpose(-2, -1)) * scaling

    # Apply softcap (Gemma2-specific)
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        q_len = query.shape[-2]
        kv_len = key_states.shape[-2]
        attn_weights = _apply_attention_mask(
            attn_weights, attention_mask, q_len, kv_len
        )

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


def _gemma2_mask_chunk(
    attention_mask: torch.Tensor,
    query_start: int,
    query_end: int,
    key_length: int,
) -> torch.Tensor:
    if attention_mask.shape[-2] == 1:
        return attention_mask[..., :, :key_length]
    return attention_mask[..., query_start:query_end, :key_length]


def _gemma2_softcap_probabilities(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    softcap: float,
    is_causal: bool,
    query_start: int,
    query_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_chunk = query[..., query_start:query_end, :]
    logits = torch.matmul(query_chunk, key.transpose(-2, -1)) * scaling
    capped_logits = torch.tanh(logits / softcap) * softcap

    softmax_input = capped_logits
    probability_mask = None
    if attention_mask is not None:
        mask = _gemma2_mask_chunk(attention_mask, query_start, query_end, key.shape[-2])
        if mask.dtype == torch.bool:
            probability_mask = mask
            softmax_input = capped_logits.masked_fill(
                ~mask, torch.finfo(capped_logits.dtype).min
            )
        else:
            softmax_input = capped_logits + mask
    elif is_causal:
        query_positions = torch.arange(
            query_start, query_end, device=query.device
        ).unsqueeze(-1)
        key_positions = torch.arange(key.shape[-2], device=query.device)
        probability_mask = key_positions <= query_positions
        softmax_input = capped_logits.masked_fill(~probability_mask, -torch.inf)

    probabilities = torch.nn.functional.softmax(
        softmax_input, dim=-1, dtype=torch.float32
    )
    if probability_mask is not None:
        probabilities = probabilities.masked_fill(~probability_mask, 0.0)
    return query_chunk, capped_logits, probabilities


class _ChunkedGemma2Attention(torch.autograd.Function):
    """Softcapped attention without retaining a full query-by-key matrix."""

    generate_vmap_rule = True

    @staticmethod
    def forward(query, key, value, attention_mask, scaling, softcap, is_causal):
        output_chunks = []
        for query_start in range(0, query.shape[-2], _GEMMA2_QUERY_CHUNK):
            query_end = min(query_start + _GEMMA2_QUERY_CHUNK, query.shape[-2])
            _, _, probabilities = _gemma2_softcap_probabilities(
                query,
                key,
                attention_mask,
                scaling,
                softcap,
                is_causal,
                query_start,
                query_end,
            )
            output_chunks.append(torch.matmul(probabilities.to(query.dtype), value))
        return torch.cat(output_chunks, dim=-2)

    @staticmethod
    def setup_context(ctx, inputs, output):
        del output
        query, key, value, attention_mask, scaling, softcap, is_causal = inputs
        if attention_mask is None:
            ctx.save_for_backward(query, key, value)
        else:
            ctx.save_for_backward(query, key, value, attention_mask)
        ctx.has_attention_mask = attention_mask is not None
        ctx.scaling = float(scaling)
        ctx.softcap = float(softcap)
        ctx.is_causal = is_causal

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
        for query_start in range(0, query.shape[-2], _GEMMA2_QUERY_CHUNK):
            query_end = min(query_start + _GEMMA2_QUERY_CHUNK, query.shape[-2])
            query_chunk, capped_logits, probabilities = _gemma2_softcap_probabilities(
                query,
                key,
                attention_mask,
                ctx.scaling,
                ctx.softcap,
                ctx.is_causal,
                query_start,
                query_end,
            )
            grad_output_chunk = grad_output[..., query_start:query_end, :]
            probabilities_input_dtype = probabilities.to(query.dtype)

            grad_value.add_(
                torch.matmul(
                    probabilities_input_dtype.transpose(-2, -1), grad_output_chunk
                )
            )
            grad_probabilities = torch.matmul(
                grad_output_chunk, value.transpose(-2, -1)
            ).to(probabilities.dtype)
            grad_softmax_input = probabilities * (
                grad_probabilities
                - (grad_probabilities * probabilities).sum(dim=-1, keepdim=True)
            )
            grad_capped_logits = grad_softmax_input.to(capped_logits.dtype)
            if attention_mask is not None and attention_mask.dtype == torch.bool:
                mask = _gemma2_mask_chunk(
                    attention_mask, query_start, query_end, key.shape[-2]
                )
                grad_capped_logits = grad_capped_logits.masked_fill(~mask, 0.0)
            grad_logits = grad_capped_logits * (
                1.0 - (capped_logits / ctx.softcap) ** 2
            )
            grad_logits = grad_logits * ctx.scaling

            grad_query_chunks.append(torch.matmul(grad_logits, key))
            grad_key.add_(torch.matmul(grad_logits.transpose(-2, -1), query_chunk))

        return (
            torch.cat(grad_query_chunks, dim=-2),
            grad_key,
            grad_value,
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
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use chunked attention for Gemma2 softcap, which SDPA cannot express."""
    if softcap is not None:
        if not module.training or dropout == 0.0:
            scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
            is_causal = kwargs.get("is_causal")
            if is_causal is None:
                is_causal = getattr(module, "is_causal", True)
            is_causal = query.shape[-2] > 1 and attention_mask is None and is_causal

            if is_causal and key.shape[-2] > query.shape[-2]:
                key = key[..., : query.shape[-2], :]
                value = value[..., : query.shape[-2], :]

            key_states = vmap_repeat_kv(key, module.num_key_value_groups)
            value_states = vmap_repeat_kv(value, module.num_key_value_groups)
            attn_output = _ChunkedGemma2Attention.apply(
                query,
                key_states,
                value_states,
                attention_mask,
                scaling,
                softcap,
                is_causal,
            )
            return attn_output.transpose(-3, -2).contiguous(), None

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

    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    return sdpa_attention_forward(
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
