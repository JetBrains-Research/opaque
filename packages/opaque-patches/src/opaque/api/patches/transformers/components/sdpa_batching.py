# Copyright (c) 2026 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility batching rules for fused CUDA SDPA backward operators."""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch

log = logging.getLogger(__name__)

_EXPECTED_SCHEMAS = {
    "_scaled_dot_product_efficient_attention_backward": (
        "aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, "
        "Tensor query, Tensor key, Tensor value, Tensor attn_bias, Tensor out, "
        "Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, float dropout_p, "
        "bool[4] grad_input_mask, bool is_causal=False, *, float? scale=None) -> "
        "(Tensor, Tensor, Tensor, Tensor)"
    ),
    "_scaled_dot_product_flash_attention_backward": (
        "aten::_scaled_dot_product_flash_attention_backward(Tensor grad_out, "
        "Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, "
        "Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, "
        "float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, "
        "*, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)"
    ),
    "_scaled_dot_product_cudnn_attention_backward": (
        "aten::_scaled_dot_product_cudnn_attention_backward(Tensor grad_out, "
        "Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, "
        "Tensor philox_seed, Tensor philox_offset, Tensor attn_bias, Tensor cum_seq_q, "
        "Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, "
        "bool is_causal, *, float? scale=None) -> (Tensor, Tensor, Tensor)"
    ),
}

_RULES_INSTALLED = False
_RULES_LOCK = threading.Lock()
_LIBRARIES: list[torch.library.Library] = []


def _active_vmap_level(*tensors: torch.Tensor | None) -> int:
    levels = [
        torch._C._functorch.maybe_get_level(tensor)
        for tensor in tensors
        if tensor is not None and torch._C._functorch.is_batchedtensor(tensor)
    ]
    if not levels:
        raise RuntimeError(  # noqa: TRY003 - internal dispatch invariant
            "fused SDPA batching rule received no BatchedTensor"
        )
    return max(levels)


def _unwrap(tensor: torch.Tensor | None, level: int):
    if tensor is None:
        return None, None
    return torch._C._functorch._unwrap_batched(tensor, level)


def _batch_size(*tensor_dims: tuple[torch.Tensor | None, int | None]) -> int:
    sizes = [
        tensor.shape[bdim]
        for tensor, bdim in tensor_dims
        if tensor is not None and bdim is not None
    ]
    if not sizes:
        raise RuntimeError(  # noqa: TRY003 - internal dispatch invariant
            "fused SDPA batching rule received no batched tensor input"
        )
    if any(size != sizes[0] for size in sizes[1:]):
        raise RuntimeError(  # noqa: TRY003 - internal dispatch invariant
            f"fused SDPA inputs have different vmap sizes: {sizes}"
        )
    return sizes[0]


def _merge_vmap_and_model_batch(
    tensor: torch.Tensor, bdim: int | None, batch_size: int
) -> torch.Tensor:
    if bdim is None:
        tensor = tensor.unsqueeze(0).expand(batch_size, *tensor.shape)
    else:
        tensor = tensor.movedim(bdim, 0)
    return tensor.flatten(0, 1)


def _restore_vmap_batch(
    tensor: torch.Tensor | None, batch_size: int, level: int
) -> torch.Tensor | None:
    if tensor is None:
        return None
    tensor = tensor.unflatten(0, (batch_size, -1))
    return torch._C._functorch._add_batch_dim(tensor, 0, level)


def _unwrap_unbatched_aux(tensor: torch.Tensor, level: int, name: str) -> torch.Tensor:
    tensor, bdim = _unwrap(tensor, level)
    if bdim is not None:
        raise RuntimeError(  # noqa: TRY003 - internal dispatch invariant
            f"fused SDPA {name} must not carry a vmap dimension"
        )
    return tensor


def _efficient_attention_backward_batch_rule(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: list[bool],
    is_causal: bool = False,
    *,
    scale: float | None = None,
):
    level = _active_vmap_level(grad_out, query, key, value, attn_bias, out, logsumexp)
    grad_out, grad_out_bdim = _unwrap(grad_out, level)
    query, query_bdim = _unwrap(query, level)
    key, key_bdim = _unwrap(key, level)
    value, value_bdim = _unwrap(value, level)
    attn_bias, attn_bias_bdim = _unwrap(attn_bias, level)
    out, out_bdim = _unwrap(out, level)
    logsumexp, logsumexp_bdim = _unwrap(logsumexp, level)
    batch_size = _batch_size(
        (grad_out, grad_out_bdim),
        (query, query_bdim),
        (key, key_bdim),
        (value, value_bdim),
        (attn_bias, attn_bias_bdim),
        (out, out_bdim),
        (logsumexp, logsumexp_bdim),
    )

    merged = (
        _merge_vmap_and_model_batch(grad_out, grad_out_bdim, batch_size),
        _merge_vmap_and_model_batch(query, query_bdim, batch_size),
        _merge_vmap_and_model_batch(key, key_bdim, batch_size),
        _merge_vmap_and_model_batch(value, value_bdim, batch_size),
        (
            _merge_vmap_and_model_batch(attn_bias, attn_bias_bdim, batch_size)
            if attn_bias is not None
            else None
        ),
        _merge_vmap_and_model_batch(out, out_bdim, batch_size),
        _merge_vmap_and_model_batch(logsumexp, logsumexp_bdim, batch_size),
    )
    philox_seed = _unwrap_unbatched_aux(philox_seed, level, "Philox seed")
    philox_offset = _unwrap_unbatched_aux(philox_offset, level, "Philox offset")
    gradients = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
        *merged,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask,
        is_causal,
        scale=scale,
    )
    return tuple(
        _restore_vmap_batch(gradient, batch_size, level) for gradient in gradients
    )


def _flash_attention_backward_batch_rule(  # noqa: PLR0913 - ATen schema
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
):
    level = _active_vmap_level(grad_out, query, key, value, out, logsumexp)
    grad_out, grad_out_bdim = _unwrap(grad_out, level)
    query, query_bdim = _unwrap(query, level)
    key, key_bdim = _unwrap(key, level)
    value, value_bdim = _unwrap(value, level)
    out, out_bdim = _unwrap(out, level)
    logsumexp, logsumexp_bdim = _unwrap(logsumexp, level)
    batch_size = _batch_size(
        (grad_out, grad_out_bdim),
        (query, query_bdim),
        (key, key_bdim),
        (value, value_bdim),
        (out, out_bdim),
        (logsumexp, logsumexp_bdim),
    )

    merged = (
        _merge_vmap_and_model_batch(grad_out, grad_out_bdim, batch_size),
        _merge_vmap_and_model_batch(query, query_bdim, batch_size),
        _merge_vmap_and_model_batch(key, key_bdim, batch_size),
        _merge_vmap_and_model_batch(value, value_bdim, batch_size),
        _merge_vmap_and_model_batch(out, out_bdim, batch_size),
        _merge_vmap_and_model_batch(logsumexp, logsumexp_bdim, batch_size),
    )
    auxiliaries = (
        _unwrap_unbatched_aux(cum_seq_q, level, "cumulative query lengths"),
        _unwrap_unbatched_aux(cum_seq_k, level, "cumulative key lengths"),
        max_q,
        max_k,
        dropout_p,
        is_causal,
        _unwrap_unbatched_aux(philox_seed, level, "Philox seed"),
        _unwrap_unbatched_aux(philox_offset, level, "Philox offset"),
    )
    gradients = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        *merged, *auxiliaries, scale=scale
    )
    return tuple(
        _restore_vmap_batch(gradient, batch_size, level) for gradient in gradients
    )


def _cudnn_attention_backward_batch_rule(  # noqa: PLR0913, PLR0917 - ATen schema
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    attn_bias: torch.Tensor | None,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: float | None = None,
):
    level = _active_vmap_level(grad_out, query, key, value, out, logsumexp, attn_bias)
    grad_out, grad_out_bdim = _unwrap(grad_out, level)
    query, query_bdim = _unwrap(query, level)
    key, key_bdim = _unwrap(key, level)
    value, value_bdim = _unwrap(value, level)
    out, out_bdim = _unwrap(out, level)
    logsumexp, logsumexp_bdim = _unwrap(logsumexp, level)
    attn_bias, attn_bias_bdim = _unwrap(attn_bias, level)
    batch_size = _batch_size(
        (grad_out, grad_out_bdim),
        (query, query_bdim),
        (key, key_bdim),
        (value, value_bdim),
        (out, out_bdim),
        (logsumexp, logsumexp_bdim),
        (attn_bias, attn_bias_bdim),
    )

    merged = (
        _merge_vmap_and_model_batch(grad_out, grad_out_bdim, batch_size),
        _merge_vmap_and_model_batch(query, query_bdim, batch_size),
        _merge_vmap_and_model_batch(key, key_bdim, batch_size),
        _merge_vmap_and_model_batch(value, value_bdim, batch_size),
        _merge_vmap_and_model_batch(out, out_bdim, batch_size),
        _merge_vmap_and_model_batch(logsumexp, logsumexp_bdim, batch_size),
    )
    auxiliaries = (
        _unwrap_unbatched_aux(philox_seed, level, "Philox seed"),
        _unwrap_unbatched_aux(philox_offset, level, "Philox offset"),
        (
            _merge_vmap_and_model_batch(attn_bias, attn_bias_bdim, batch_size)
            if attn_bias is not None
            else None
        ),
        _unwrap_unbatched_aux(cum_seq_q, level, "cumulative query lengths"),
        _unwrap_unbatched_aux(cum_seq_k, level, "cumulative key lengths"),
        max_q,
        max_k,
        dropout_p,
        is_causal,
    )
    gradients = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(
        *merged, *auxiliaries, scale=scale
    )
    return tuple(
        _restore_vmap_batch(gradient, batch_size, level) for gradient in gradients
    )


_BATCH_RULES: dict[str, Any] = {
    "_scaled_dot_product_efficient_attention_backward": (
        _efficient_attention_backward_batch_rule
    ),
    "_scaled_dot_product_flash_attention_backward": (
        _flash_attention_backward_batch_rule
    ),
    "_scaled_dot_product_cudnn_attention_backward": (
        _cudnn_attention_backward_batch_rule
    ),
}


def _has_native_batch_rule(operator: str) -> bool:
    return torch._C._dispatch_has_kernel_for_dispatch_key(
        f"aten::{operator}", "FuncTorchBatched"
    )


def _has_expected_schema(operator: str) -> bool:
    try:
        schema = str(getattr(torch.ops.aten, operator).default._schema)
    except AttributeError:
        return False
    return schema == _EXPECTED_SCHEMAS[operator]


def install_fused_sdpa_batching_rules() -> None:
    """Install missing fused SDPA backward vmap rules when schemas are compatible."""
    global _RULES_INSTALLED

    if _RULES_INSTALLED:
        return
    with _RULES_LOCK:
        if _RULES_INSTALLED:
            return
        missing = [
            operator
            for operator in _BATCH_RULES
            if not _has_native_batch_rule(operator) and _has_expected_schema(operator)
        ]
        if missing:
            library = torch.library.Library("aten", "IMPL", "FuncTorchBatched")
            for operator in missing:
                library.impl(operator, _BATCH_RULES[operator])
            _LIBRARIES.append(library)
            log.debug("Installed fused SDPA batching rules for %s", ", ".join(missing))

        incompatible = [
            operator
            for operator in _BATCH_RULES
            if not _has_native_batch_rule(operator)
            and not _has_expected_schema(operator)
        ]
        if incompatible:
            log.debug(
                "Keeping PyTorch's SDPA vmap fallback for incompatible operators: %s",
                ", ".join(incompatible),
            )
        _RULES_INSTALLED = True
