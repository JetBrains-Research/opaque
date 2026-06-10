# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Triton RoPE kernels derive from the Unsloth project
# (Apache-2.0; https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rope_embedding.py)
# and have been adapted to Opaque's vmap-friendly new-style autograd dispatch.
# See NOTICE in the repository root.
"""RoPE (Rotary Position Embedding) kernels with vmap support for DP-SGD.

Ported from unsloth/kernels/rope_embedding.py with new-style autograd API.

RoPE applies position-dependent rotation to Q and K tensors:
  Q_rot = Q * cos + rotate_half(Q) * sin
where rotate_half swaps and negates the first half of the last dimension.

Three implementations:
1. Fast_RoPE_Embedding: Triton kernel for single tensor (Q or K)
2. Fast_RoPE_Embedding_QK: Triton kernel for Q+K together (with GQA support)
3. Slow_RoPE_Embedding: Pure PyTorch fallback (always vmap-compatible)
"""

import triton
import triton.language as tl
import torch
from ._utils import (
    calculate_settings,
    ensure_cuda_tensors,
    follow_autocast,
    torch_gpu_device,
)

ROPE_GROUP_SIZE: int = 4


# ============================================================================
# Triton Kernels
# ============================================================================


@triton.jit
def _rope_embedding_kernel(
    Q,
    Q_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RoPE embedding kernel for single tensor."""
    ROPE_GROUP_SIZE: tl.constexpr = 4
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(
        sin + (row_position % seqlen) * sin_row_stride + col_offsets,
        mask=mask,
        other=0,
    )
    cos1 = tl.load(
        cos + (row_position % seqlen) * cos_row_stride + col_offsets,
        mask=mask,
        other=0,
    )

    if BACKWARD_PASS:
        sin1 = -sin1

    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = (
            row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim
        )

        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)


@triton.jit
def _rope_embedding_qk_kernel(
    Q,
    Q_batch_stride,
    Q_head_stride,
    Q_seq_stride,
    K,
    K_batch_stride,
    K_head_stride,
    K_seq_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    rope_embedding_indices,
    seqlen,
    head_dim: tl.constexpr,
    n_heads_K: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    HAS_ROPE_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RoPE embedding kernel for Q and K together with GQA support."""
    row_position = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    if HAS_ROPE_INDICES:
        rot_position = tl.load(
            rope_embedding_indices + row_position,
            eviction_policy="evict_first",
        ).to(tl.int32)
    else:
        rot_position = row_position % seqlen

    cos_ptr = cos + rot_position * cos_row_stride
    sin_ptr = sin + rot_position * sin_row_stride
    sin1 = tl.load(sin_ptr + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos_ptr + col_offsets, mask=mask, other=0)

    if BACKWARD_PASS:
        sin1 = -sin1

    batch_id = row_position // seqlen
    seq_index = row_position - batch_id * seqlen

    # Process Q
    q_ptr = (
        Q
        + batch_id * Q_batch_stride
        + head_position * Q_head_stride
        + seq_index * Q_seq_stride
    )
    q0 = tl.load(q_ptr + col_offsets, mask=mask, other=0)
    q1 = tl.load(q_ptr + half_head_dim + col_offsets, mask=mask, other=0)
    tl.store(q_ptr + col_offsets, q0 * cos1 - q1 * sin1, mask=mask)
    tl.store(q_ptr + half_head_dim + col_offsets, q1 * cos1 + q0 * sin1, mask=mask)

    # Process K (only if head_position < n_heads_K for GQA)
    if head_position < n_heads_K:
        k_ptr = (
            K
            + batch_id * K_batch_stride
            + head_position * K_head_stride
            + seq_index * K_seq_stride
        )
        k0 = tl.load(k_ptr + col_offsets, mask=mask, other=0)
        k1 = tl.load(k_ptr + half_head_dim + col_offsets, mask=mask, other=0)
        tl.store(k_ptr + col_offsets, k0 * cos1 - k1 * sin1, mask=mask)
        tl.store(k_ptr + half_head_dim + col_offsets, k1 * cos1 + k0 * sin1, mask=mask)


# The kernels are already decorated with @triton.jit
# Just create aliases for the heuristics versions
_rope_embedding_kernel_heuristics = _rope_embedding_kernel
_rope_embedding_qk_kernel_heuristics = _rope_embedding_qk_kernel


# ============================================================================
# Autograd Functions with vmap support
# ============================================================================


class _RoPEBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support.

    RoPE backward uses the same kernel as forward with BACKWARD_PASS=True.
    """

    @staticmethod
    def forward(grad_Q, cos, sin):
        batch, seq_len, n_heads, head_dim = grad_Q.shape
        # In-place: overwrite grad_Q buffer (internal gradient, safe to mutate)
        dQ = grad_Q.reshape(batch * seq_len, n_heads * head_dim).contiguous()
        n_rows = dQ.shape[0]

        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        with torch_gpu_device(dQ.device):
            _rope_embedding_kernel_heuristics[(n_rows, n_groups)](
                dQ,
                dQ.stride(0),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                seq_len,
                head_dim,
                n_heads,
                BACKWARD_PASS=True,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return dQ.reshape(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for RoPE")

    @staticmethod
    def vmap(info, in_dims, grad_Q, cos, sin):
        grad_Q_bdim, cos_bdim, sin_bdim = in_dims

        assert cos_bdim is None and sin_bdim is None
        if grad_Q_bdim != 0:
            # The merge below assumes the vmap batch is the leading dim;
            # any other placement would interleave rows across examples and
            # rotate with the wrong positions.
            raise ValueError(f"grad_Q should be batched at dim 0, got {grad_Q_bdim}")

        head_dim = grad_Q.shape[-1]
        n_heads = grad_Q.shape[-2]
        seq_len = grad_Q.shape[-3]

        # Merge all leading dims into rows — works for both 4D and 5D input
        dQ = grad_Q.reshape(-1, n_heads * head_dim).contiguous()
        n_rows = dQ.shape[0]

        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        with torch_gpu_device(dQ.device):
            _rope_embedding_kernel_heuristics[(n_rows, n_groups)](
                dQ,
                dQ.stride(0),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                seq_len,
                head_dim,
                n_heads,
                BACKWARD_PASS=True,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return dQ.reshape(grad_Q.shape), grad_Q_bdim


class Opaque_RoPE(torch.autograd.Function):
    """Fast RoPE embedding with new-style API and vmap support.

    Input shape: (batch, seq_len, n_heads, head_dim)
    Applies: Q_rot = Q * cos - rotate_half(Q) * sin
    """

    @staticmethod
    def forward(Q, cos, sin):
        """Forward pass - apply RoPE to Q tensor.

        Args:
            Q: Input tensor (batch, seq_len, n_heads, head_dim)
            cos: Cosine cache (seq_len, head_dim) or (1, 1, seq_len, head_dim)
            sin: Sine cache (seq_len, head_dim) or (1, 1, seq_len, head_dim)

        Returns:
            Q_rot: Rotated Q tensor, same shape as input
        """
        cos = cos.squeeze()
        sin = sin.squeeze()

        batch, seq_len, n_heads, head_dim = Q.shape
        Q_out = Q.clone().reshape(batch * seq_len, n_heads * head_dim)
        n_rows, n_cols = Q_out.shape

        assert seq_len <= cos.shape[0], f"seq_len {seq_len} > cos length {cos.shape[0]}"

        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)

        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        with torch_gpu_device(Q.device):
            _rope_embedding_kernel_heuristics[(n_rows, n_groups)](
                Q_out,
                Q_out.stride(0),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                seq_len,
                head_dim,
                n_heads,
                BACKWARD_PASS=False,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return Q_out.reshape(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def setup_context(ctx, inputs, output):
        Q, cos, sin = inputs
        # Save cos/sin from inputs (squeezed to match what forward used)
        ctx.save_for_backward(cos.squeeze(), sin.squeeze())
        # Recompute metadata from input shapes
        batch, seq_len, n_heads, head_dim = Q.shape
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = div + (mod != 0)
        ctx.shape = Q.shape

    @staticmethod
    def backward(ctx, grad_Q):
        cos, sin = ctx.saved_tensors
        dQ = _RoPEBackward.apply(grad_Q, cos, sin)
        return dQ, None, None

    @staticmethod
    def vmap(info, in_dims, Q, cos, sin):
        """Custom vmap rule for DP-SGD.

        Calls Triton forward kernel directly with merged batch dims.
        """
        Q_bdim, cos_bdim, sin_bdim = in_dims

        if Q_bdim != 0:
            raise ValueError(f"Q should be batched at dim 0, got {Q_bdim}")
        if cos_bdim is not None or sin_bdim is not None:
            raise ValueError("cos and sin should not be batched")

        cos_sq = cos.squeeze()
        sin_sq = sin.squeeze()

        batched_shape = Q.shape
        head_dim = Q.shape[-1]
        n_heads = Q.shape[-2]
        seq_len = Q.shape[-3]

        # Merge all leading dims into rows — works for both 4D and 5D input
        Q_out = Q.reshape(-1, n_heads * head_dim).contiguous()
        n_rows = Q_out.shape[0]

        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        with torch_gpu_device(Q.device):
            _rope_embedding_kernel_heuristics[(n_rows, n_groups)](
                Q_out,
                Q_out.stride(0),
                cos_sq,
                cos_sq.stride(0),
                sin_sq,
                sin_sq.stride(0),
                seq_len,
                head_dim,
                n_heads,
                BACKWARD_PASS=False,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return Q_out.reshape(batched_shape), Q_bdim


class _RoPE_QK_Backward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(grad_Q, grad_K, cos, sin, rope_ptr, has_indices, seq_len):
        head_dim = grad_Q.shape[-1]
        batch = grad_Q.shape[0]
        n_heads_Q = grad_Q.shape[1]
        n_heads_K = grad_K.shape[1]

        if not has_indices:
            rope_ptr = cos.new_empty(1, dtype=torch.int32)

        # In-place: overwrite grad buffers (internal gradients, safe to mutate)
        dQ_out = grad_Q.contiguous()
        dK_out = grad_K.contiguous()

        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        with torch_gpu_device(dQ_out.device):
            _rope_embedding_qk_kernel_heuristics[(batch * seq_len, n_heads_Q)](
                dQ_out,
                dQ_out.stride(0),
                dQ_out.stride(1),
                dQ_out.stride(2),
                dK_out,
                dK_out.stride(0),
                dK_out.stride(1),
                dK_out.stride(2),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                rope_ptr,
                seq_len,
                head_dim=head_dim,
                n_heads_K=n_heads_K,
                BACKWARD_PASS=True,
                HAS_ROPE_INDICES=has_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return dQ_out, dK_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for RoPE_QK")

    @staticmethod
    def vmap(info, in_dims, grad_Q, grad_K, cos, sin, rope_ptr, has_indices, seq_len):
        gQ_bdim, gK_bdim, cos_bdim, sin_bdim, rp_bdim, hi_bdim, sl_bdim = in_dims

        assert cos_bdim is None and sin_bdim is None
        assert hi_bdim is None and sl_bdim is None
        if gQ_bdim != 0 or gK_bdim != 0:
            # The leading-dim collapse below assumes the vmap batch is dim 0;
            # any other placement would interleave rows across examples and
            # rotate with the wrong positions.
            raise ValueError(
                f"grad_Q/grad_K should be batched at dim 0, got {gQ_bdim}/{gK_bdim}"
            )

        head_dim = grad_Q.shape[-1]
        n_heads_Q = grad_Q.shape[-3]
        n_heads_K = grad_K.shape[-3]

        # Collapse all leading dims — works for both 4D and 5D input
        total_batch = grad_Q[..., 0, 0, 0].numel()
        dQ = grad_Q.reshape(total_batch, n_heads_Q, seq_len, head_dim).contiguous()
        dK = grad_K.reshape(total_batch, n_heads_K, seq_len, head_dim).contiguous()

        if not has_indices:
            rope_ptr_local = cos.new_empty(1, dtype=torch.int32)
        else:
            rope_ptr_local = rope_ptr

        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        with torch_gpu_device(dQ.device):
            _rope_embedding_qk_kernel_heuristics[(total_batch * seq_len, n_heads_Q)](
                dQ,
                dQ.stride(0),
                dQ.stride(1),
                dQ.stride(2),
                dK,
                dK.stride(0),
                dK.stride(1),
                dK.stride(2),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                rope_ptr_local,
                seq_len,
                head_dim=head_dim,
                n_heads_K=n_heads_K,
                BACKWARD_PASS=True,
                HAS_ROPE_INDICES=has_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return (
            (dQ.reshape(grad_Q.shape), dK.reshape(grad_K.shape)),
            (gQ_bdim, gK_bdim),
        )


class Opaque_RoPE_QK(torch.autograd.Function):
    """Fast RoPE embedding for Q and K together with GQA support.

    Input shapes:
        Q: (batch, n_heads_Q, seq_len, head_dim)
        K: (batch, n_heads_K, seq_len, head_dim)

    Supports different number of heads for Q and K (GQA).
    """

    @staticmethod
    def forward(Q, K, cos, sin, rope_indices=None):
        """Forward pass - apply RoPE to Q and K tensors.

        Args:
            Q: Query tensor (batch, n_heads_Q, seq_len, head_dim)
            K: Key tensor (batch, n_heads_K, seq_len, head_dim)
            cos: Cosine cache
            sin: Sine cache
            rope_indices: Optional position indices for non-contiguous positions

        Returns:
            Tuple of (Q_rot, K_rot)
        """
        has_indices = rope_indices is not None
        cos = cos.squeeze()
        sin = sin.squeeze()

        batch, n_heads_Q, seq_len, head_dim = Q.shape
        _, n_heads_K, _, _ = K.shape

        Q_out = Q.clone()
        K_out = K.clone()

        if has_indices:
            rope_ptr = rope_indices.reshape(-1).to(dtype=torch.int32, device=Q.device)
        else:
            rope_ptr = cos.new_empty(1, dtype=torch.int32)

        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        with torch_gpu_device(Q.device):
            _rope_embedding_qk_kernel_heuristics[(batch * seq_len, n_heads_Q)](
                Q_out,
                Q_out.stride(0),
                Q_out.stride(1),
                Q_out.stride(2),
                K_out,
                K_out.stride(0),
                K_out.stride(1),
                K_out.stride(2),
                cos,
                cos.stride(0),
                sin,
                sin.stride(0),
                rope_ptr,
                seq_len,
                head_dim=head_dim,
                n_heads_K=n_heads_K,
                BACKWARD_PASS=False,
                HAS_ROPE_INDICES=has_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return Q_out, K_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        Q, K, cos, sin, rope_indices = inputs

        has_indices = rope_indices is not None
        if has_indices:
            rope_ptr = rope_indices.reshape(-1).to(dtype=torch.int32, device=Q.device)
        else:
            rope_ptr = cos.new_empty(1, dtype=torch.int32)

        ctx.save_for_backward(
            cos.squeeze(),
            sin.squeeze(),
            rope_ptr if has_indices else torch.empty(0, device=Q.device),
        )
        BLOCK_SIZE, num_warps = calculate_settings(Q.shape[-1])
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.has_indices = has_indices
        ctx.seq_len = Q.shape[2]
        ctx.n_heads_Q = Q.shape[1]
        ctx.n_heads_K = K.shape[1]

    @staticmethod
    def backward(ctx, grad_Q, grad_K):
        cos, sin, rope_ptr = ctx.saved_tensors
        dQ, dK = _RoPE_QK_Backward.apply(
            grad_Q, grad_K, cos, sin, rope_ptr, ctx.has_indices, ctx.seq_len
        )
        return dQ, dK, None, None, None

    @staticmethod
    def vmap(info, in_dims, Q, K, cos, sin, rope_indices):
        """Custom vmap rule for DP-SGD.

        Calls Triton forward kernel directly with merged batch dims.
        """
        Q_bdim, K_bdim, cos_bdim, sin_bdim, rope_bdim = in_dims

        if Q_bdim != 0 or K_bdim != 0:
            raise ValueError("Q and K should be batched at dim 0")
        if cos_bdim is not None or sin_bdim is not None:
            raise ValueError("cos and sin should not be batched")
        if rope_bdim is not None:
            raise ValueError("rope_indices should not be batched")

        cos_sq = cos.squeeze()
        sin_sq = sin.squeeze()

        has_indices = rope_indices is not None
        head_dim = Q.shape[-1]
        seq_len = Q.shape[-2]
        n_heads_Q = Q.shape[-3]
        n_heads_K = K.shape[-3]

        # Collapse all leading dims — works for both 4D and 5D input
        total_batch = Q[..., 0, 0, 0].numel()
        Q_out = Q.reshape(total_batch, n_heads_Q, seq_len, head_dim).contiguous()
        K_out = K.reshape(total_batch, n_heads_K, seq_len, head_dim).contiguous()

        if has_indices:
            rope_ptr = rope_indices.reshape(-1).to(dtype=torch.int32, device=Q.device)
        else:
            rope_ptr = cos_sq.new_empty(1, dtype=torch.int32)

        BLOCK_SIZE, num_warps = calculate_settings(head_dim)

        with torch_gpu_device(Q.device):
            _rope_embedding_qk_kernel_heuristics[(total_batch * seq_len, n_heads_Q)](
                Q_out,
                Q_out.stride(0),
                Q_out.stride(1),
                Q_out.stride(2),
                K_out,
                K_out.stride(0),
                K_out.stride(1),
                K_out.stride(2),
                cos_sq,
                cos_sq.stride(0),
                sin_sq,
                sin_sq.stride(0),
                rope_ptr,
                seq_len,
                head_dim=head_dim,
                n_heads_K=n_heads_K,
                BACKWARD_PASS=False,
                HAS_ROPE_INDICES=has_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return (
            (Q_out.reshape(Q.shape), K_out.reshape(K.shape)),
            (Q_bdim, K_bdim),
        )


def _slow_rope_cos_sin(cos, sin, position_ids):
    """Select cos/sin rows by position (HF apply_rotary_pos_emb semantics).

    Returns (bs, 1, seq_len, head_dim) caches that broadcast over the heads
    dim of a (batch, n_heads, seq_len, head_dim) tensor. Identity when
    position_ids is None.
    """
    if position_ids is None:
        return cos, sin
    cos = cos.squeeze(1).squeeze(0)  # (seq_len, head_dim)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)  # (bs, 1, seq_len, head_dim)
    sin = sin[position_ids].unsqueeze(1)
    return cos, sin


class Opaque_SlowRoPE(torch.autograd.Function):
    """Pure PyTorch RoPE embedding (slower but always vmap-compatible).

    Use this as fallback when Triton kernels have issues.
    """

    @staticmethod
    def forward(Q, cos, sin, position_ids=None):
        """Forward pass using pure PyTorch operations.

        Args:
            Q: Input tensor (batch, n_heads, seq_len, head_dim)
            cos: Cosine cache
            sin: Sine cache
            position_ids: Optional position indices

        Returns:
            Q_rot: Rotated Q tensor
        """
        cos, sin = _slow_rope_cos_sin(cos, sin, position_ids)

        half = Q.shape[-1] // 2
        Q_out = Q.clone()
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
        Q_out = Q_out * cos + RH_Q * sin

        return Q_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        Q, cos, sin, position_ids = inputs
        # Save the position-indexed caches: backward must rotate with the
        # same per-token angles the forward used.
        cos, sin = _slow_rope_cos_sin(cos, sin, position_ids)
        ctx.save_for_backward(cos, sin)

    @staticmethod
    def backward(ctx, grad_Q):
        cos, sin = ctx.saved_tensors

        half = grad_Q.shape[-1] // 2
        RH_dQ = torch.cat((grad_Q[..., half:], -grad_Q[..., :half]), dim=-1)
        dQ = grad_Q * cos + RH_dQ * sin

        return dQ, None, None, None

    @staticmethod
    def vmap(info, in_dims, Q, cos, sin, position_ids):
        """Custom vmap rule - trivial since it's pure PyTorch."""
        Q_bdim, cos_bdim, sin_bdim, pos_bdim = in_dims

        if Q_bdim != 0:
            raise ValueError(f"Q should be batched at dim 0, got {Q_bdim}")
        if cos_bdim is not None or sin_bdim is not None:
            raise ValueError("cos and sin should not be batched")
        if pos_bdim is not None:
            raise ValueError("position_ids should not be batched")

        output = Opaque_SlowRoPE.apply(Q, cos, sin, position_ids)
        return output, Q_bdim


# ============================================================================
# Convenience wrappers
# ============================================================================


def opaque_rope(Q, cos, sin):
    """Apply RoPE embedding with vmap support.

    Args:
        Q: Input tensor (batch, seq_len, n_heads, head_dim)
        cos: Cosine cache
        sin: Sine cache

    Returns:
        Q_rot: Rotated tensor, same shape as input
    """
    ensure_cuda_tensors(Q, cos, sin, fn_name="opaque_rope")
    Q, cos, sin = follow_autocast(Q, cos, sin)
    return Opaque_RoPE.apply(Q, cos, sin)


def opaque_rope_qk(Q, K, cos, sin, rope_indices=None):
    """Apply RoPE embedding to Q and K with vmap support.

    Args:
        Q: Query tensor (batch, n_heads_Q, seq_len, head_dim)
        K: Key tensor (batch, n_heads_K, seq_len, head_dim)
        cos: Cosine cache
        sin: Sine cache
        rope_indices: Optional position indices

    Returns:
        Tuple of (Q_rot, K_rot)
    """
    tensors = [Q, K, cos, sin]
    if rope_indices is not None:
        tensors.append(rope_indices)
    ensure_cuda_tensors(*tensors, fn_name="opaque_rope_qk")
    Q, K, cos, sin = follow_autocast(Q, K, cos, sin)
    return Opaque_RoPE_QK.apply(Q, K, cos, sin, rope_indices)


def opaque_slow_rope(Q, cos, sin, position_ids=None):
    """Apply RoPE embedding using pure PyTorch (always vmap-compatible).

    Args:
        Q: Input tensor (batch, n_heads, seq_len, head_dim)
        cos: Cosine cache
        sin: Sine cache
        position_ids: Optional position indices

    Returns:
        Q_rot: Rotated tensor, same shape as input
    """
    Q, cos, sin = follow_autocast(Q, cos, sin)
    return Opaque_SlowRoPE.apply(Q, cos, sin, position_ids)
