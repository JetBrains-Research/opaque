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
from .utils import calculate_settings, torch_gpu_device

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
        offs_q2 = row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim

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

class NewStyleRoPEEmbedding(torch.autograd.Function):
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
        batch, seq_len, n_heads, head_dim = ctx.shape

        dQ = grad_Q.clone().reshape(batch * seq_len, n_heads * head_dim)
        n_rows = dQ.shape[0]

        with torch_gpu_device(dQ.device):
            _rope_embedding_kernel_heuristics[(n_rows, ctx.n_groups)](
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
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )

        return dQ.reshape(batch, seq_len, n_heads, head_dim), None, None

    @staticmethod
    def vmap(info, in_dims, Q, cos, sin):
        """Custom vmap rule for DP-SGD.

        When vmap is applied, Q has shape (vmap_batch, batch, seq_len, n_heads, head_dim).
        We reshape to merge vmap_batch into batch, apply RoPE, then reshape back.
        """
        Q_bdim, cos_bdim, sin_bdim = in_dims

        if Q_bdim != 0:
            raise ValueError(f"Q should be batched at dim 0, got {Q_bdim}")
        if cos_bdim is not None or sin_bdim is not None:
            raise ValueError("cos and sin should not be batched")

        # Q shape: (vmap_batch, batch, seq_len, n_heads, head_dim)
        vmap_batch = Q.shape[0]
        original_shape = Q.shape
        # Merge vmap_batch into batch dimension
        Q_merged = Q.reshape(-1, *Q.shape[2:])  # (vmap_batch * batch, seq_len, n_heads, head_dim)

        # Apply RoPE to merged tensor
        Q_rot = NewStyleRoPEEmbedding.apply(Q_merged, cos, sin)

        # Reshape back to original vmap shape
        Q_rot = Q_rot.reshape(original_shape)

        return Q_rot, Q_bdim


class NewStyleRoPEEmbeddingQK(torch.autograd.Function):
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

        Q_out = Q.clone() if not Q.is_contiguous() else Q.clone()
        K_out = K.clone() if not K.is_contiguous() else K.clone()

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

        ctx.save_for_backward(cos.squeeze(), sin.squeeze(), rope_ptr if has_indices else torch.empty(0, device=Q.device))
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
        batch, _, _, head_dim = grad_Q.shape

        if not ctx.has_indices:
            rope_ptr = cos.new_empty(1, dtype=torch.int32)

        dQ_out = grad_Q.clone() if not grad_Q.is_contiguous() else grad_Q.clone()
        dK_out = grad_K.clone() if not grad_K.is_contiguous() else grad_K.clone()

        with torch_gpu_device(dQ_out.device):
            _rope_embedding_qk_kernel_heuristics[(batch * ctx.seq_len, ctx.n_heads_Q)](
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
                ctx.seq_len,
                head_dim=head_dim,
                n_heads_K=ctx.n_heads_K,
                BACKWARD_PASS=True,
                HAS_ROPE_INDICES=ctx.has_indices,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )

        return dQ_out, dK_out, None, None, None

    @staticmethod
    def vmap(info, in_dims, Q, K, cos, sin, rope_indices):
        """Custom vmap rule for DP-SGD.

        Handles vmap batch dimension by merging into the existing batch dim.
        """
        Q_bdim, K_bdim, cos_bdim, sin_bdim, rope_bdim = in_dims

        if Q_bdim != 0 or K_bdim != 0:
            raise ValueError("Q and K should be batched at dim 0")
        if cos_bdim is not None or sin_bdim is not None:
            raise ValueError("cos and sin should not be batched")
        if rope_bdim is not None:
            raise ValueError("rope_indices should not be batched")

        # Q/K shape: (vmap_batch, batch, n_heads, seq_len, head_dim)
        original_Q_shape = Q.shape
        original_K_shape = K.shape

        # Merge vmap_batch into batch
        Q_merged = Q.reshape(-1, *Q.shape[2:])
        K_merged = K.reshape(-1, *K.shape[2:])

        Q_rot, K_rot = NewStyleRoPEEmbeddingQK.apply(Q_merged, K_merged, cos, sin, rope_indices)

        # Reshape back
        Q_rot = Q_rot.reshape(original_Q_shape)
        K_rot = K_rot.reshape(original_K_shape)

        return (Q_rot, K_rot), (Q_bdim, K_bdim)


class NewStyleSlowRoPEEmbedding(torch.autograd.Function):
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
        if position_ids is not None:
            cos = cos.squeeze(1).squeeze(0)
            sin = sin.squeeze(1).squeeze(0)
            cos = cos[position_ids].unsqueeze(2)
            sin = sin[position_ids].unsqueeze(2)

        half = Q.shape[-1] // 2
        Q_out = Q.clone()
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
        Q_out = Q_out * cos + RH_Q * sin

        return Q_out

    @staticmethod
    def setup_context(ctx, inputs, output):
        Q, cos, sin, position_ids = inputs
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

        output = NewStyleSlowRoPEEmbedding.apply(Q, cos, sin, position_ids)
        return output, Q_bdim


# ============================================================================
# Convenience wrappers
# ============================================================================

def rope_embedding_vmap(Q, cos, sin):
    """Apply RoPE embedding with vmap support.

    Args:
        Q: Input tensor (batch, seq_len, n_heads, head_dim)
        cos: Cosine cache
        sin: Sine cache

    Returns:
        Q_rot: Rotated tensor, same shape as input
    """
    return NewStyleRoPEEmbedding.apply(Q, cos, sin)


def rope_embedding_qk_vmap(Q, K, cos, sin, rope_indices=None):
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
    return NewStyleRoPEEmbeddingQK.apply(Q, K, cos, sin, rope_indices)


def slow_rope_embedding_vmap(Q, cos, sin, position_ids=None):
    """Apply RoPE embedding using pure PyTorch (always vmap-compatible).

    Args:
        Q: Input tensor (batch, n_heads, seq_len, head_dim)
        cos: Cosine cache
        sin: Sine cache
        position_ids: Optional position indices

    Returns:
        Q_rot: Rotated tensor, same shape as input
    """
    return NewStyleSlowRoPEEmbedding.apply(Q, cos, sin, position_ids)
