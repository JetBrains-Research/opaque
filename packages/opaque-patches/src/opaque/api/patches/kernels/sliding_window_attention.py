# Copyright (c) 2026 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Private bounded-memory Triton sliding-window attention.

This module is intentionally not part of the public ``opaque.patches.kernels``
surface.  The forward pass uses blockwise online softmax.  Backward recomputes
the same score blocks and never stores an attention matrix.
"""

import math

import torch
import triton
import triton.language as tl

from opaque.exceptions import ConfigurationError

from ._utils import follow_autocast, torch_gpu_device, triton_tanh

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_MASK_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_BLOCK_M = 32
_BLOCK_N = 32
_MIN_DOT_SIZE = 16
_FOUR_WARP_MAX_BLOCK = 64
_MAX_HEAD_DIM = 256
_MIN_INPUT_NDIM = 3


@triton.jit
def _attention_forward_kernel(  # noqa: PLR0913, PLR0917
    Q,
    K,
    V,
    MASK,
    OUT,
    LSE,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_mb,
    stride_ms,
    stride_ob,
    stride_oh,
    stride_os,
    stride_lb,
    stride_lh,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    WINDOW,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    head_q = tl.program_id(1)
    batch = tl.program_id(2)
    head_kv = head_q // (H_Q // H_KV)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (
        Q
        + batch * stride_qb
        + head_q * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :]
    )
    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    lo = tl.maximum(0, start_m - WINDOW + 1)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(N_CTX, start_m + BLOCK_M)
    for start_n in tl.range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = (
            K
            + batch * stride_kb
            + head_kv * stride_kh
            + offs_n[:, None] * stride_ks
            + offs_d[None, :]
        )
        v_ptrs = (
            V
            + batch * stride_vb
            + head_kv * stride_vh
            + offs_n[:, None] * stride_vs
            + offs_d[None, :]
        )
        kv_mask = (offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        valid = (
            (offs_m[:, None] < N_CTX)
            & (offs_n[None, :] < N_CTX)
            & (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] > offs_m[:, None] - WINDOW)
        )
        if HAS_MASK:
            key_valid = tl.load(
                MASK + batch * stride_mb + offs_n * stride_ms,
                mask=offs_n < N_CTX,
                other=0,
            )
            valid &= key_valid[None, :] != 0

        raw_scores = tl.dot(q, tl.trans(k)) * SCALE
        if HAS_SOFTCAP:
            scores = SOFTCAP * triton_tanh(raw_scores / SOFTCAP)
        else:
            scores = raw_scores
        scores = tl.where(valid, scores, -float("inf"))

        tile_has_value = tl.sum(valid.to(tl.int32), axis=1) != 0
        tile_max = tl.max(scores, axis=1)
        m_new = tl.where(tile_has_value, tl.maximum(m_i, tile_max), m_i)
        alpha_shift = tl.where(tile_has_value, m_i - m_new, 0.0)
        score_shift = tl.where(valid, scores - m_new[:, None], -float("inf"))
        alpha = tl.exp(alpha_shift)
        p = tl.exp(score_shift)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_new

    has_value = l_i != 0.0
    denominator = tl.where(has_value, l_i, 1.0)
    safe_max = tl.where(has_value, m_i, 0.0)
    out = acc / denominator[:, None]
    lse = safe_max + tl.log(denominator)
    out_ptrs = (
        OUT
        + batch * stride_ob
        + head_q * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :]
    )
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.store(
        LSE + batch * stride_lb + head_q * stride_lh + offs_m,
        lse,
        mask=offs_m < N_CTX,
    )


@triton.jit
def _attention_backward_dq_kernel(  # noqa: PLR0913, PLR0917
    DO,
    Q,
    K,
    V,
    MASK,
    OUT,
    LSE,
    DQ,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_mb,
    stride_ms,
    stride_ob,
    stride_oh,
    stride_os,
    stride_lb,
    stride_lh,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    WINDOW,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    head_q = tl.program_id(1)
    batch = tl.program_id(2)
    head_kv = head_q // (H_Q // H_KV)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    md_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(
        Q
        + batch * stride_qb
        + head_q * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :],
        mask=md_mask,
        other=0.0,
    )
    do = tl.load(
        DO
        + batch * stride_dob
        + head_q * stride_doh
        + offs_m[:, None] * stride_dos
        + offs_d[None, :],
        mask=md_mask,
        other=0.0,
    )
    out = tl.load(
        OUT
        + batch * stride_ob
        + head_q * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :],
        mask=md_mask,
        other=0.0,
    )
    lse = tl.load(
        LSE + batch * stride_lb + head_q * stride_lh + offs_m,
        mask=offs_m < N_CTX,
        other=0.0,
    )
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    lo = tl.maximum(0, start_m - WINDOW + 1)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(N_CTX, start_m + BLOCK_M)
    for start_n in tl.range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nd_mask = (offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(
            K
            + batch * stride_kb
            + head_kv * stride_kh
            + offs_n[:, None] * stride_ks
            + offs_d[None, :],
            mask=nd_mask,
            other=0.0,
        )
        v = tl.load(
            V
            + batch * stride_vb
            + head_kv * stride_vh
            + offs_n[:, None] * stride_vs
            + offs_d[None, :],
            mask=nd_mask,
            other=0.0,
        )
        valid = (
            (offs_m[:, None] < N_CTX)
            & (offs_n[None, :] < N_CTX)
            & (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] > offs_m[:, None] - WINDOW)
        )
        if HAS_MASK:
            key_valid = tl.load(
                MASK + batch * stride_mb + offs_n * stride_ms,
                mask=offs_n < N_CTX,
                other=0,
            )
            valid &= key_valid[None, :] != 0

        raw_scores = tl.dot(q, tl.trans(k)) * SCALE
        if HAS_SOFTCAP:
            tanh_scores = triton_tanh(raw_scores / SOFTCAP)
            scores = SOFTCAP * tanh_scores
        else:
            scores = raw_scores
        p = tl.where(valid, tl.exp(scores - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if HAS_SOFTCAP:
            ds *= 1.0 - tanh_scores * tanh_scores
        dq += tl.dot(ds.to(q.dtype), k) * SCALE

    tl.store(
        DQ
        + batch * stride_dqb
        + head_q * stride_dqh
        + offs_m[:, None] * stride_dqs
        + offs_d[None, :],
        dq,
        mask=md_mask,
    )


@triton.jit
def _attention_backward_dkv_kernel(  # noqa: PLR0913, PLR0917
    DO,
    Q,
    K,
    V,
    MASK,
    OUT,
    LSE,
    DK,
    DV,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_mb,
    stride_ms,
    stride_ob,
    stride_oh,
    stride_os,
    stride_lb,
    stride_lh,
    stride_dkb,
    stride_dkh,
    stride_dks,
    stride_dvb,
    stride_dvh,
    stride_dvs,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    WINDOW,
    SCALE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    head_kv = tl.program_id(1)
    batch = tl.program_id(2)

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    nd_mask = (offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
    k = tl.load(
        K
        + batch * stride_kb
        + head_kv * stride_kh
        + offs_n[:, None] * stride_ks
        + offs_d[None, :],
        mask=nd_mask,
        other=0.0,
    )
    v = tl.load(
        V
        + batch * stride_vb
        + head_kv * stride_vh
        + offs_n[:, None] * stride_vs
        + offs_d[None, :],
        mask=nd_mask,
        other=0.0,
    )
    if HAS_MASK:
        key_valid = tl.load(
            MASK + batch * stride_mb + offs_n * stride_ms,
            mask=offs_n < N_CTX,
            other=0,
        )
    else:
        key_valid = offs_n < N_CTX

    dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    q_hi = tl.minimum(N_CTX, start_n + BLOCK_N + WINDOW - 1)

    for group_offset in range(GROUP_SIZE):
        head_q = head_kv * GROUP_SIZE + group_offset
        for start_m in tl.range(start_n, q_hi, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            md_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
            q = tl.load(
                Q
                + batch * stride_qb
                + head_q * stride_qh
                + offs_m[:, None] * stride_qs
                + offs_d[None, :],
                mask=md_mask,
                other=0.0,
            )
            do = tl.load(
                DO
                + batch * stride_dob
                + head_q * stride_doh
                + offs_m[:, None] * stride_dos
                + offs_d[None, :],
                mask=md_mask,
                other=0.0,
            )
            out = tl.load(
                OUT
                + batch * stride_ob
                + head_q * stride_oh
                + offs_m[:, None] * stride_os
                + offs_d[None, :],
                mask=md_mask,
                other=0.0,
            )
            lse = tl.load(
                LSE + batch * stride_lb + head_q * stride_lh + offs_m,
                mask=offs_m < N_CTX,
                other=0.0,
            )
            delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
            valid = (
                (offs_m[:, None] < N_CTX)
                & (offs_n[None, :] < N_CTX)
                & (offs_n[None, :] <= offs_m[:, None])
                & (offs_n[None, :] > offs_m[:, None] - WINDOW)
                & (key_valid[None, :] != 0)
            )
            raw_scores = tl.dot(q, tl.trans(k)) * SCALE
            if HAS_SOFTCAP:
                tanh_scores = triton_tanh(raw_scores / SOFTCAP)
                scores = SOFTCAP * tanh_scores
            else:
                scores = raw_scores
            p = tl.where(valid, tl.exp(scores - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            if HAS_SOFTCAP:
                ds *= 1.0 - tanh_scores * tanh_scores
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * SCALE
            dv += tl.dot(tl.trans(p.to(q.dtype)), do)

    tl.store(
        DK
        + batch * stride_dkb
        + head_kv * stride_dkh
        + offs_n[:, None] * stride_dks
        + offs_d[None, :],
        dk,
        mask=nd_mask,
    )
    tl.store(
        DV
        + batch * stride_dvb
        + head_kv * stride_dvh
        + offs_n[:, None] * stride_dvs
        + offs_d[None, :],
        dv,
        mask=nd_mask,
    )


def _canonical(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().reshape(
        -1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]
    )


def _canonical_mask(mask: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
    if mask.numel() == 0:
        return mask
    return mask.to(dtype=torch.bool).contiguous().reshape(batch, seq_len)


def _launch_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    padding_mask: torch.Tensor,
    sliding_window: int,
    scale: float,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = query.shape
    q = _canonical(query)
    k = _canonical(key)
    v = _canonical(value)
    batch, heads_q, seq_len, head_dim = q.shape
    heads_kv = k.shape[1]
    mask = _canonical_mask(padding_mask, batch, seq_len)
    out = torch.empty_like(q)
    lse = torch.empty((batch, heads_q, seq_len), dtype=torch.float32, device=q.device)
    block_d = max(_MIN_DOT_SIZE, triton.next_power_of_2(head_dim))
    num_warps = 4 if block_d <= _FOUR_WARP_MAX_BLOCK else 8

    with torch_gpu_device(q.device):
        _attention_forward_kernel[(triton.cdiv(seq_len, _BLOCK_M), heads_q, batch)](
            q,
            k,
            v,
            mask,
            out,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            mask.stride(0) if mask.numel() else 0,
            mask.stride(1) if mask.numel() else 0,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            H_Q=heads_q,
            H_KV=heads_kv,
            N_CTX=seq_len,
            HEAD_DIM=head_dim,
            WINDOW=min(sliding_window, seq_len),
            SCALE=scale,
            SOFTCAP=softcap,
            HAS_MASK=mask.numel() != 0,
            HAS_SOFTCAP=softcap > 0.0,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
    return out.reshape(original_shape), lse.reshape(original_shape[:-1])


def _launch_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    padding_mask: torch.Tensor,
    sliding_window: int,
    scale: float,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _canonical(query)
    k = _canonical(key)
    v = _canonical(value)
    do = _canonical(grad_output)
    out = _canonical(output)
    batch, heads_q, seq_len, head_dim = q.shape
    heads_kv = k.shape[1]
    lse_flat = lse.contiguous().reshape(batch, heads_q, seq_len)
    mask = _canonical_mask(padding_mask, batch, seq_len)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_d = max(_MIN_DOT_SIZE, triton.next_power_of_2(head_dim))
    num_warps = 4 if block_d <= _FOUR_WARP_MAX_BLOCK else 8
    common = {
        "H_Q": heads_q,
        "H_KV": heads_kv,
        "N_CTX": seq_len,
        "HEAD_DIM": head_dim,
        "WINDOW": min(sliding_window, seq_len),
        "SCALE": scale,
        "SOFTCAP": softcap,
        "HAS_MASK": mask.numel() != 0,
        "HAS_SOFTCAP": softcap > 0.0,
        "BLOCK_M": _BLOCK_M,
        "BLOCK_N": _BLOCK_N,
        "BLOCK_D": block_d,
        "num_warps": num_warps,
    }
    tensors = (do, q, k, v, mask, out, lse_flat)
    strides = (
        do.stride(0),
        do.stride(1),
        do.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        mask.stride(0) if mask.numel() else 0,
        mask.stride(1) if mask.numel() else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse_flat.stride(0),
        lse_flat.stride(1),
    )
    with torch_gpu_device(q.device):
        _attention_backward_dq_kernel[(triton.cdiv(seq_len, _BLOCK_M), heads_q, batch)](
            *tensors,
            dq,
            *strides,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            **common,
        )
        _attention_backward_dkv_kernel[
            (triton.cdiv(seq_len, _BLOCK_N), heads_kv, batch)
        ](
            *tensors,
            dk,
            dv,
            *strides,
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            GROUP_SIZE=heads_q // heads_kv,
            **common,
        )
    return (
        dq.reshape(query.shape),
        dk.reshape(key.shape),
        dv.reshape(value.shape),
    )


def _move_vmap_dim(tensor: torch.Tensor, batch_dim: int | None) -> torch.Tensor:
    if batch_dim is None:
        raise ConfigurationError(*("Q, K, and V must all be vmapped",))
    return tensor.movedim(batch_dim, 0)


def _vmap_mask(
    mask: torch.Tensor,
    batch_dim: int | None,
    batch_size: int,
) -> torch.Tensor:
    if mask.numel() == 0:
        return mask
    if batch_dim is None:
        return mask.unsqueeze(0).expand(batch_size, *mask.shape)
    return mask.movedim(batch_dim, 0)


class _SlidingWindowAttentionBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        grad_output,
        query,
        key,
        value,
        output,
        lse,
        padding_mask,
        sliding_window,
        scale,
        softcap,
    ):
        return _launch_backward(
            grad_output,
            query,
            key,
            value,
            output,
            lse,
            padding_mask,
            sliding_window,
            scale,
            softcap,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError(
            "Double backward is not supported for sliding-window attention"
        )

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_output,
        query,
        key,
        value,
        output,
        lse,
        padding_mask,
        sliding_window,
        scale,
        softcap,
    ):
        tensor_dims = in_dims[:6]
        if len(set(tensor_dims)) != 1 or tensor_dims[0] is None:
            raise ConfigurationError(
                *(
                    "Sliding-window attention backward requires matching vmap "
                    f"dimensions for gradient, Q, K, V, output, and LSE; got {in_dims}",
                )
            )
        if any(dim is not None for dim in in_dims[7:]):
            raise ConfigurationError(*("Attention scalar arguments cannot be vmapped",))
        batch_dim = tensor_dims[0]
        moved = [
            _move_vmap_dim(tensor, batch_dim)
            for tensor in (grad_output, query, key, value, output, lse)
        ]
        mask = _vmap_mask(padding_mask, in_dims[6], info.batch_size)
        grads = _launch_backward(
            *moved,
            mask,
            sliding_window,
            scale,
            softcap,
        )
        return grads, (0, 0, 0)


class _SlidingWindowAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        query,
        key,
        value,
        padding_mask,
        sliding_window,
        scale,
        softcap,
    ):
        return _launch_forward(
            query,
            key,
            value,
            padding_mask,
            sliding_window,
            scale,
            softcap,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        query, key, value, padding_mask, sliding_window, scale, softcap = inputs
        result, lse = output
        ctx.save_for_backward(query, key, value, result, lse, padding_mask)
        ctx.sliding_window = sliding_window
        ctx.scale = scale
        ctx.softcap = softcap
        ctx.mark_non_differentiable(lse)

    @staticmethod
    def backward(ctx, grad_output, _grad_lse):
        query, key, value, output, lse, padding_mask = ctx.saved_tensors
        dq, dk, dv = _SlidingWindowAttentionBackward.apply(
            grad_output.contiguous(),
            query,
            key,
            value,
            output,
            lse,
            padding_mask,
            ctx.sliding_window,
            ctx.scale,
            ctx.softcap,
        )
        return dq, dk, dv, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims,
        query,
        key,
        value,
        padding_mask,
        sliding_window,
        scale,
        softcap,
    ):
        q_dim, k_dim, v_dim, mask_dim, window_dim, scale_dim, softcap_dim = in_dims
        if q_dim is None or not (q_dim == k_dim == v_dim):
            raise ConfigurationError(
                *(
                    "Sliding-window attention requires matching vmap dimensions "
                    f"for Q, K, and V; got {in_dims[:3]}",
                )
            )
        if any(dim is not None for dim in (window_dim, scale_dim, softcap_dim)):
            raise ConfigurationError(*("Attention scalar arguments cannot be vmapped",))
        q = _move_vmap_dim(query, q_dim)
        k = _move_vmap_dim(key, k_dim)
        v = _move_vmap_dim(value, v_dim)
        mask = _vmap_mask(padding_mask, mask_dim, info.batch_size)
        output, lse = _launch_forward(
            q,
            k,
            v,
            mask,
            sliding_window,
            scale,
            softcap,
        )
        return (output, lse), (0, 0)


def _can_use_triton_sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    padding_mask: torch.Tensor | None,
    sliding_window: int,
    scale: float,
    softcap: float | None = None,
) -> bool:
    """Return whether the private Triton implementation supports these inputs.

    The argument contract matches :func:`_triton_sliding_window_attention`.
    This predicate never launches a kernel and returns ``False`` for unsupported
    devices, dtypes, layouts, masks, or scalar options.
    """
    try:
        if not all(torch.is_tensor(tensor) for tensor in (query, key, value)):
            return False
        if (
            query.ndim < _MIN_INPUT_NDIM
            or key.ndim != query.ndim
            or value.ndim != query.ndim
        ):
            return False
        if not (query.is_cuda and key.is_cuda and value.is_cuda):
            return False
        if not (query.device == key.device == value.device):
            return False
        active_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else query.dtype
        )
        if active_dtype not in _SUPPORTED_DTYPES:
            return False
        if not torch.is_autocast_enabled("cuda") and not (
            query.dtype == key.dtype == value.dtype
        ):
            return False
        if query.shape[:-3] != key.shape[:-3] or key.shape != value.shape:
            return False
        heads_q, seq_len, head_dim = query.shape[-3:]
        heads_kv, key_seq_len, key_head_dim = key.shape[-3:]
        if (
            query.numel() == 0
            or seq_len == 0
            or head_dim == 0
            or head_dim > _MAX_HEAD_DIM
            or key_seq_len != seq_len
            or key_head_dim != head_dim
            or heads_kv == 0
            or heads_q % heads_kv != 0
        ):
            return False
        if (
            isinstance(sliding_window, bool)
            or not isinstance(sliding_window, int)
            or sliding_window <= 0
        ):
            return False
        if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
            return False
        if softcap is not None and (
            not isinstance(softcap, (int, float))
            or not math.isfinite(float(softcap))
            or float(softcap) <= 0.0
        ):
            return False
        if active_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            return False
        return padding_mask is None or (
            torch.is_tensor(padding_mask)
            and padding_mask.is_cuda
            and padding_mask.device == query.device
            and padding_mask.dtype in _MASK_DTYPES
            and padding_mask.shape == (*query.shape[:-3], seq_len)
        )
    except (RuntimeError, TypeError, ValueError):
        return False


def _triton_sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    padding_mask: torch.Tensor | None,
    sliding_window: int,
    scale: float,
    softcap: float | None = None,
) -> torch.Tensor:
    """Run private causal, equal-length sliding-window attention on CUDA.

    Args:
        query: Tensor shaped ``(..., query_heads, sequence, head_dim)``.
        key: Tensor shaped ``(..., kv_heads, sequence, head_dim)``.
        value: Tensor with the same shape as ``key``.
        padding_mask: Optional bool or 0/1 tensor shaped ``(..., sequence)``.
        sliding_window: Number of causal keys visible to each query.
        scale: Multiplier applied to each unmodified Q/K dot product.
        softcap: Optional positive Gemma2-style logit softcap.

    Floating inputs follow the active CUDA autocast dtype before dispatch.
    Call the eligibility predicate first and use the caller's dense fallback
    when it returns ``False``.
    """
    if not _can_use_triton_sliding_window_attention(
        query,
        key,
        value,
        padding_mask,
        sliding_window,
        scale,
        softcap,
    ):
        raise ConfigurationError(
            *(
                "Inputs are not eligible for private Triton sliding-window "
                "attention; call _can_use_triton_sliding_window_attention first "
                "and use the PyTorch attention path when it returns False.",
            )
        )
    query, key, value = follow_autocast(query, key, value)
    mask = (
        torch.empty(0, dtype=torch.bool, device=query.device)
        if padding_mask is None
        else padding_mask.to(dtype=torch.bool)
    )
    result, _lse = _SlidingWindowAttention.apply(
        query,
        key,
        value,
        mask,
        int(sliding_window),
        float(scale),
        0.0 if softcap is None else float(softcap),
    )
    return result
