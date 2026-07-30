# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused grouped-GEMM MoE expert FFN — sparse (O(T*K)) perf path for ``opaque_moe``.

A re-derivation of Liger's ``LigerFusedMoEFunction`` made vmap- and
vmap(grad)-safe for Opaque's per-sample-gradient DP-SGD path. Same Function/vmap
contract as the dense :class:`~opaque.api.patches.kernels.moe.Opaque_MoE`
baseline, so ``opaque_moe`` dispatches CUDA+bf16/fp16 here and keeps the dense
torch path for CPU/fp32.

Strategy (per the MoE-port lessons): tokens are sorted by expert and run through
``torch._grouped_mm`` for the three mode-1 (offset-grouped along the token dim)
GEMMs — forward up-proj, forward down-proj, backward ``dx``. The mode-2
per-group weight grads (``dW1``/``dW2``) can't use ``_grouped_mm`` (it asserts
16-byte-aligned group sizes — unusable for data-dependent MoE groups), so a
single custom Triton kernel :func:`_grouped_AtB` computes ``out[g] = A_g^T @ B_g``
over the variable-size row groups. All reductions accumulate in fp32.

Per-sample weight grads under ``vmap(grad)`` (DP-SGD) use **virtual experts**:
sample ``b``'s tokens for real expert ``e`` are assigned to group ``b*E + e``, so
the grouped weight-grad naturally lands in a ``(B, E, ...)`` buffer — never
summed across the batch. The forward/dx GEMMs index the *shared* weights by the
real expert (``group % E``).
"""

# ``I`` is the per-expert intermediate dim throughout (paired with ``2I`` for the
# fused gate+up projection); single-letter tensor-shape names are intentional here.
# ruff: noqa: E741

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ._utils import follow_autocast, torch_gpu_device

# ---------------------------------------------------------------------------
# Custom Triton kernel: grouped A^T @ B (the mode-2 per-group weight grad)
# ---------------------------------------------------------------------------


@triton.jit
def _grouped_AtB_kernel(
    A_ptr,  # (M, P)  rows grouped by expert via seg_offs
    B_ptr,  # (M, Q)
    seg_ptr,  # (G+1,) int32 — exclusive prefix offsets; group g is rows [seg[g], seg[g+1])
    out_ptr,  # (G, P, Q) — out[g] = A[group g]^T @ B[group g]
    P: tl.constexpr,
    Q: tl.constexpr,
    stride_a_m,
    stride_a_p: tl.constexpr,
    stride_b_m,
    stride_b_q: tl.constexpr,
    stride_o_g,
    stride_o_p,
    stride_o_q: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over P (output rows)
    BLOCK_N: tl.constexpr,  # tile over Q (output cols)
    BLOCK_K: tl.constexpr,  # tile over the group's token rows (contraction)
):
    """Grid: (G * ceil(P/BLOCK_M), ceil(Q/BLOCK_N)). Early-exit on empty groups."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n_p_tiles: tl.constexpr = (P + BLOCK_M - 1) // BLOCK_M
    # int64 guards expert/row offset overflow at large G*P*Q (mirrors Liger #1246).
    g = (pid0 // n_p_tiles).to(tl.int64)
    p_tile = pid0 % n_p_tiles

    seg_start = tl.load(seg_ptr + g)
    seg_end = tl.load(seg_ptr + g + 1)
    M_e = seg_end - seg_start
    if M_e == 0:
        return

    p_idx = p_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    q_idx = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    p_mask = p_idx < P
    q_mask = q_idx < Q

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, M_e, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        k_mask = k_off < M_e
        rows = (seg_start + k_off).to(tl.int64)

        # A_blk[p, k] = A[row_k, p_idx_p]  -> (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + rows[None, :] * stride_a_m + p_idx[:, None] * stride_a_p
        a_blk = tl.load(a_ptrs, mask=k_mask[None, :] & p_mask[:, None], other=0.0)
        # B_blk[k, q] = B[row_k, q_idx_q]  -> (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + rows[:, None] * stride_b_m + q_idx[None, :] * stride_b_q
        b_blk = tl.load(b_ptrs, mask=k_mask[:, None] & q_mask[None, :], other=0.0)

        acc = tl.dot(a_blk, b_blk, acc=acc)

    o_ptrs = (
        out_ptr
        + g * stride_o_g
        + p_idx[:, None] * stride_o_p
        + q_idx[None, :] * stride_o_q
    )
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=p_mask[:, None] & q_mask[None, :]
    )


def _grouped_AtB(A, B, seg_offs, G):
    """``out[g] = A[group g]^T @ B[group g]`` for row-groups defined by ``seg_offs``
    (exclusive prefix, length G+1). ``A`` (M,P), ``B`` (M,Q) -> ``out`` (G,P,Q).
    fp32 accumulate, cast to ``A.dtype``. The mode-2 weight grad torch can't do."""
    A = A.contiguous()
    B = B.contiguous()
    P, Q = A.shape[1], B.shape[1]
    out = torch.zeros(G, P, Q, dtype=A.dtype, device=A.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    def grid(meta):
        return (G * triton.cdiv(P, meta["BLOCK_M"]), triton.cdiv(Q, meta["BLOCK_N"]))

    with torch_gpu_device(A.device):
        _grouped_AtB_kernel[grid](
            A,
            B,
            seg_offs,
            out,
            P=P,
            Q=Q,
            stride_a_m=A.stride(0),
            stride_a_p=A.stride(1),
            stride_b_m=B.stride(0),
            stride_b_q=B.stride(1),
            stride_o_g=out.stride(0),
            stride_o_p=out.stride(1),
            stride_o_q=out.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    return out


# ---------------------------------------------------------------------------
# Routing + grouped-GEMM helpers
# ---------------------------------------------------------------------------


def _grouped_mm(A, Bw, ends):
    """``torch._grouped_mm`` mode-1: ``A`` (M,Kc) grouped along M by ``ends`` @
    ``Bw`` (G,Kc,Nc) -> (M,Nc). ``ends`` = int32 cumulative group ends.

    ``Bw`` may be a non-contiguous ``.mT`` view — grouped_mm consumes it directly,
    avoiding a transient full-weight copy (matters at MoE scale)."""
    return torch._grouped_mm(A.contiguous(), Bw, offs=ends)


def _route_sort(expert_of_row, n_groups):
    """Sort (token,k) rows by group id. Returns the sort permutation and the
    int32 cumulative-end offsets for :func:`_grouped_mm` (length ``n_groups``)."""
    sort_idx = torch.argsort(expert_of_row, stable=True)
    ends = torch.bincount(expert_of_row, minlength=n_groups).cumsum(0).to(torch.int32)
    return sort_idx, ends


def _seg_offsets(group_of_row, n_groups):
    """Exclusive-prefix offsets (length ``n_groups+1``) for :func:`_grouped_AtB`."""
    counts = torch.bincount(group_of_row, minlength=n_groups)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)


def _fused_moe_forward(x_flat, W1, W2, expert_of_row, tok_of_row, tw_row, E, N, H, I):
    """Sparse grouped MoE forward. ``x_flat`` (N,H); shared weights ``W1`` (E,2I,H),
    ``W2`` (E,H,I); routing flattened to (N*K,) rows, grouped by real expert."""
    dt = x_flat.dtype
    sort_idx, ends = _route_sort(expert_of_row, E)
    tok_s = tok_of_row[sort_idx]
    x_s = x_flat[tok_s]  # (NK, H)

    gate_up = _grouped_mm(x_s, W1.mT, ends)  # (NK, 2I)
    g, u = gate_up[:, :I], gate_up[:, I:]
    h = F.silu(g.float()).to(dt) * u  # silu in fp32, matches dense
    y = _grouped_mm(h, W2.mT, ends)  # (NK, H)

    yw = (y * tw_row[sort_idx].unsqueeze(-1)).float()
    out = torch.zeros(N, H, dtype=torch.float32, device=x_flat.device)
    out.index_add_(0, tok_s, yw)  # fp32 expert-sum reduction
    return out.to(dt)


def _fused_moe_backward(
    grad_flat,
    x_flat,
    W1,
    W2,
    real_eor,
    tok_of_row,
    tw_row,
    group_of_row,
    n_groups,
    N,
    H,
    I,
    K,
    compute_wgrad=True,
):
    """Manual grouped MoE backward. Returns ``dx`` (N,H), ``dW1`` (n_groups,2I,H),
    ``dW2`` (n_groups,H,I), ``dtw`` (N,K).

    The mode-1 GEMMs (forward recompute, ``dh``, ``dx``) use **real-expert**
    grouping with the shared weights (``E <= 1024`` groups — within
    ``torch._grouped_mm``'s cap). The mode-2 per-group weight grads use
    ``group_of_row`` (== real expert for the summed path, or the virtual expert
    ``b*E+e`` for the per-sample DP path) and run through the custom
    :func:`_grouped_AtB` Triton kernel, which has no group cap and needs no
    repeated-weight buffer.

    ``compute_wgrad=False`` skips the mode-2 weight grads (returns
    ``dW1=dW2=None``) for frozen expert weights: their per-sample
    ``(B, E, ...)`` buffers are the largest backward allocation and autograd
    discards them anyway."""
    dt = x_flat.dtype
    E = W1.shape[0]
    sort_idx, ends = _route_sort(real_eor, E)
    tok_s = tok_of_row[sort_idx]
    x_s = x_flat[tok_s]  # (NK, H), real-expert order

    # Recompute forward intermediates (vmap rules have no ctx; cheap vs the GEMMs).
    gate_up = _grouped_mm(x_s, W1.mT, ends)
    g, u = gate_up[:, :I], gate_up[:, I:]
    sig = torch.sigmoid(g.float())
    silu = (g.float() * sig).to(dt)
    h = silu * u
    y = _grouped_mm(h, W2.mT, ends)  # (NK, H)

    go_s = grad_flat[tok_s]  # (NK, H)
    tw_s = tw_row[sort_idx]
    dy = (tw_s.unsqueeze(-1) * go_s).to(dt)  # (NK, H)

    # dtw: ∂L/∂routing_weight = sum_h grad_out * y, scattered back to (N, K).
    dtw_row = (go_s.float() * y.float()).sum(-1)  # (NK,)
    dtw = torch.zeros(N * K, dtype=torch.float32, device=x_flat.device)
    dtw[sort_idx] = dtw_row
    dtw = dtw.reshape(N, K).to(dt)

    dh = _grouped_mm(dy, W2, ends)  # (NK, I)
    dsilu = (sig * (1.0 + g.float() * (1.0 - sig))).to(dt)
    dgu = torch.cat([dh * u * dsilu, dh * silu], dim=-1)  # (NK, 2I)

    dx_s = _grouped_mm(dgu, W1, ends)  # (NK, H)
    dx = torch.zeros(N, H, dtype=torch.float32, device=x_flat.device)
    dx.index_add_(0, tok_s, dx_s.float())  # fp32 token reduction
    dx = dx.to(dt)

    # Frozen experts: autograd discards the mode-2 weight grads, so skip the
    # ``(n_groups, ...)`` allocations.
    if not compute_wgrad:
        return dx, None, None, dtw

    # Per-group weight grads (mode-2): re-sort the real-expert-ordered rows into
    # ``group_of_row`` order (a no-op permutation when groups == real experts),
    # then the custom kernel keeps each group separate — NOT summed across groups.
    vperm = torch.argsort(group_of_row[sort_idx], stable=True)
    seg = _seg_offsets(group_of_row, n_groups)
    dW1 = _grouped_AtB(dgu[vperm], x_s[vperm], seg, n_groups)  # (n_groups, 2I, H)
    dW2 = _grouped_AtB(dy[vperm], h[vperm], seg, n_groups)  # (n_groups, H, I)
    return dx, dW1, dW2, dtw


def _flat_routing(N, K, device):
    """``tok_of_row`` (N*K,): the token index for each flattened (token,k) row."""
    return torch.arange(N, device=device).repeat_interleave(K)


# ---------------------------------------------------------------------------
# Autograd Functions (two-Function pattern for vmap(grad) — see swiglu.py)
# ---------------------------------------------------------------------------


class _FusedMoEBackward(torch.autograd.Function):
    """Backward as an autograd.Function so ``vmap(grad)`` routes here (DP-SGD).
    No double backward."""

    @staticmethod
    def forward(
        grad_out, x, gate_up_proj, down_proj, top_k_index, top_k_weights, compute_wgrad
    ):
        N, H = x.shape
        K = top_k_index.shape[-1]
        E = gate_up_proj.shape[0]
        I = gate_up_proj.shape[1] // 2
        eor = top_k_index.reshape(-1)
        tor = _flat_routing(N, K, x.device)
        # Summed weight grads: group == real expert.
        dx, dW1, dW2, dtw = _fused_moe_backward(
            grad_out,
            x,
            gate_up_proj,
            down_proj,
            eor,
            tor,
            top_k_weights.reshape(-1),
            group_of_row=eor,
            n_groups=E,
            N=N,
            H=H,
            I=I,
            K=K,
            compute_wgrad=compute_wgrad,
        )
        return dx, dW1, dW2, dtw

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for fused MoE")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_out,
        x,
        gate_up_proj,
        down_proj,
        top_k_index,
        top_k_weights,
        compute_wgrad,
    ):
        # DP-SGD contract: grad_out/x/top_k batched at 0; shared weights unbatched.
        B, T, H = x.shape
        K = top_k_index.shape[-1]
        E = gate_up_proj.shape[0]
        I = gate_up_proj.shape[1] // 2
        N = B * T

        xf = x.reshape(N, H)
        gf = grad_out.reshape(N, H)
        tif = top_k_index.reshape(N, K)
        twf = top_k_weights.reshape(N, K)

        tor = _flat_routing(N, K, x.device)
        eor = tif.reshape(
            -1
        )  # (NK,) real expert ids — used for the shared-weight GEMMs
        # Virtual experts: sample b's tokens for real expert e -> group b*E + e, so
        # the per-group weight grad lands per-sample (never summed across the batch).
        virtual = (tor // T) * E + eor  # (NK,)

        dx, dW1, dW2, dtw = _fused_moe_backward(
            gf,
            xf,
            gate_up_proj,
            down_proj,
            eor,
            tor,
            twf.reshape(-1),
            group_of_row=virtual,
            n_groups=B * E,
            N=N,
            H=H,
            I=I,
            K=K,
            compute_wgrad=compute_wgrad,
        )
        if not compute_wgrad:
            # Frozen experts: emit a single unbatched zero weight grad with
            # ``out_dim=None`` so vmap broadcasts it, instead of materialising the
            # per-sample ``(B, E, ...)`` buffers.
            return (
                (
                    dx.reshape(B, T, H),
                    gate_up_proj.new_zeros(gate_up_proj.shape),
                    down_proj.new_zeros(down_proj.shape),
                    dtw.reshape(B, T, K),
                ),
                (0, None, None, 0),
            )
        # dx/dtw per-token (merged batch); dW1/dW2 per-sample (kept, for DP-SGD).
        return (
            (
                dx.reshape(B, T, H),
                dW1.reshape(B, E, 2 * I, H),
                dW2.reshape(B, E, H, I),
                dtw.reshape(B, T, K),
            ),
            (0, 0, 0, 0),
        )


class Opaque_FusedMoE(torch.autograd.Function):
    """Sparse grouped-GEMM MoE expert FFN with vmap-grad support (DP-SGD). Same
    ``.apply`` signature as the dense ``Opaque_MoE`` and Liger's fused op."""

    @staticmethod
    def forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        N, H = x.shape
        K = top_k_index.shape[-1]
        I = gate_up_proj.shape[1] // 2
        eor = top_k_index.reshape(-1)
        tor = _flat_routing(N, K, x.device)
        return _fused_moe_forward(
            x,
            gate_up_proj,
            down_proj,
            eor,
            tor,
            top_k_weights.reshape(-1),
            E=gate_up_proj.shape[0],
            N=N,
            H=H,
            I=I,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_out):
        # needs_input_grad: (x, gate_up_proj, down_proj, top_k_index, top_k_weights).
        # Skip the mode-2 weight grads when neither expert weight requires grad.
        compute_wgrad = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        dx, dW1, dW2, dtw = _FusedMoEBackward.apply(
            grad_out, *ctx.saved_tensors, compute_wgrad
        )
        # inputs: x, gate_up_proj, down_proj, top_k_index (int, no grad), top_k_weights
        return dx, dW1, dW2, None, dtw

    @staticmethod
    def vmap(info, in_dims, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        # Forward is token-independent: merge the vmap batch into the token dim.
        B, T, H = x.shape
        K = top_k_index.shape[-1]
        I = gate_up_proj.shape[1] // 2
        N = B * T
        xf = x.reshape(N, H)
        tif = top_k_index.reshape(N, K)
        twf = top_k_weights.reshape(N, K)
        eor = tif.reshape(-1)
        tor = _flat_routing(N, K, x.device)
        out = _fused_moe_forward(
            xf,
            gate_up_proj,
            down_proj,
            eor,
            tor,
            twf.reshape(-1),
            E=gate_up_proj.shape[0],
            N=N,
            H=H,
            I=I,
        )
        return out.reshape(B, T, H), 0


def opaque_fused_moe(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Sparse grouped-GEMM MoE expert FFN (CUDA bf16/fp16). Autograd +
    ``vmap(grad)`` (DP-SGD) flow through the two-Function pair above."""
    x, gate_up_proj, down_proj, top_k_weights = follow_autocast(
        x, gate_up_proj, down_proj, top_k_weights
    )
    return Opaque_FusedMoE.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
