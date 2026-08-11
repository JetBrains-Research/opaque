# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Sparse grouped-GEMM MoE expert FFN — non-Triton (MPS/CPU) perf path.

The pure-PyTorch sibling of ``fused_moe.py``: same sparse O(T*K) strategy and the
same Function/vmap contract as the dense :class:`~opaque.api.patches.kernels.moe.Opaque_MoE`
baseline, but built on ``torch._grouped_mm`` instead of Triton so it runs on
Apple MPS (and CPU). ``opaque_moe`` dispatches here on non-CUDA hosts when
``torch._grouped_mm`` is available, avoiding the dense path's O(T*E) blowup
(every token through every expert).

Tokens are sorted by expert and run through ``torch._grouped_mm`` for the three
mode-1 (offset-grouped along the token dim) GEMMs — forward up-proj, forward
down-proj, backward ``dx``. The mode-2 per-group weight grads (``dW1``/``dW2``)
are ``out[g] = A_g^T @ B_g`` (contraction grouped along the *token* axis).
``torch._grouped_mm``'s 2D×2D layout expresses this — its 16-byte rule is on
matrix *strides*, not group sizes — but :func:`_grouped_AtB` does it with an
explicit per-group loop (G = E, or B*E for the per-sample DP path — both small)
so it stays safe inside the vmap rules. All reductions accumulate in fp32.

Per-sample weight grads under ``vmap(grad)`` (DP-SGD) use **virtual experts**:
sample ``b``'s tokens for real expert ``e`` go to group ``b*E + e``, so the
grouped weight-grad lands per-sample in a ``(B, E, ...)`` buffer — never summed
across the batch. The forward/dx GEMMs index the shared weights by real expert.
"""

# ``I`` is the per-expert intermediate dim throughout (paired with ``2I`` for the
# fused gate+up projection); single-letter tensor-shape names are intentional.
# ruff: noqa: E741

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_mm_available() -> bool:
    """True when ``torch._grouped_mm`` exists (the sparse path's GEMM backend)."""
    return hasattr(torch, "_grouped_mm")


# ---------------------------------------------------------------------------
# Routing + grouped-GEMM helpers
# ---------------------------------------------------------------------------


def _grouped_mm(A, Bw, ends):
    """``torch._grouped_mm`` mode-1: ``A`` (M,Kc) grouped along M by ``ends`` @
    ``Bw`` (G,Kc,Nc) -> (M,Nc). ``ends`` = int32 cumulative group ends."""
    return torch._grouped_mm(A.contiguous(), Bw, offs=ends)


def _grouped_AtB(A, B, seg_offs, G):
    """``out[g] = A[group g]^T @ B[group g]`` for row-groups defined by ``seg_offs``
    (exclusive prefix, length G+1). ``A`` (M,P), ``B`` (M,Q) -> ``out`` (G,P,Q).

    Per-group matmul loop (G is E or B*E — small). fp32 accumulate, cast to
    ``A.dtype``. Same op as ``torch._grouped_mm(A.mT, B, offs)`` (2D×2D layout);
    done as an explicit loop because these functions run inside vmap rules
    (regular tensors), so a Python loop + in-place group writes are safe.
    """
    P, Q = A.shape[1], B.shape[1]
    out = torch.zeros(G, P, Q, dtype=A.dtype, device=A.device)
    bounds = seg_offs.tolist()
    Af, Bf = A.float(), B.float()
    for g in range(G):
        lo, hi = bounds[g], bounds[g + 1]
        if hi > lo:
            out[g] = (Af[lo:hi].t() @ Bf[lo:hi]).to(A.dtype)
    return out


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


def _flat_routing(N, K, device):
    """``tok_of_row`` (N*K,): the token index for each flattened (token,k) row."""
    return torch.arange(N, device=device).repeat_interleave(K)


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

    Mode-1 GEMMs (forward recompute, ``dh``, ``dx``) use real-expert grouping with
    the shared weights. The mode-2 per-group weight grads use ``group_of_row``
    (== real expert for the summed path, or the virtual expert ``b*E+e`` for the
    per-sample DP path) through :func:`_grouped_AtB`.

    ``compute_wgrad=False`` skips the mode-2 weight grads entirely (returns
    ``dW1=dW2=None``). When the expert weights are frozen (DP-SGD LoRA on
    attention only), those per-sample ``(n_groups, ...)`` buffers are pure waste —
    the same OOM the Triton ``_fused_moe_backward`` avoids — so this mirrors that
    skip for the non-Triton MPS/CPU grouped path."""
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

    # Frozen experts (DP-SGD LoRA on attention only): the mode-2 weight grads are
    # discarded by autograd, so skip the two giant ``(n_groups, ...)`` allocations.
    if not compute_wgrad:
        return dx, None, None, dtw

    # Per-group weight grads (mode-2): re-sort the real-expert-ordered rows into
    # ``group_of_row`` order, then keep each group separate (never summed across).
    vperm = torch.argsort(group_of_row[sort_idx], stable=True)
    seg = _seg_offsets(group_of_row, n_groups)
    dW1 = _grouped_AtB(dgu[vperm], x_s[vperm], seg, n_groups)  # (n_groups, 2I, H)
    dW2 = _grouped_AtB(dy[vperm], h[vperm], seg, n_groups)  # (n_groups, H, I)
    return dx, dW1, dW2, dtw


# ---------------------------------------------------------------------------
# Autograd Functions (two-Function pattern for vmap(grad) — see moe.py)
# ---------------------------------------------------------------------------


class _GroupedMoEBackward(torch.autograd.Function):
    """Backward as an autograd.Function so ``vmap(grad)`` routes here (DP-SGD)."""

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
        raise NotImplementedError("Double backward not supported for grouped MoE")

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
        eor = tif.reshape(-1)
        # Virtual experts: sample b's tokens for real expert e -> group b*E + e.
        virtual = (tor // T) * E + eor
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
            # per-sample ``(B, E, ...)`` buffers that OOM at large batch scale.
            return (
                (
                    dx.reshape(B, T, H),
                    gate_up_proj.new_zeros(gate_up_proj.shape),
                    down_proj.new_zeros(down_proj.shape),
                    dtw.reshape(B, T, K),
                ),
                (0, None, None, 0),
            )
        return (
            (
                dx.reshape(B, T, H),
                dW1.reshape(B, E, 2 * I, H),
                dW2.reshape(B, E, H, I),
                dtw.reshape(B, T, K),
            ),
            (0, 0, 0, 0),
        )


class Opaque_GroupedMoE(torch.autograd.Function):
    """Sparse grouped-GEMM MoE expert FFN (non-Triton). Same ``.apply`` signature
    as the dense ``Opaque_MoE`` and the Triton ``Opaque_FusedMoE``."""

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
        # Frozen experts (DP-SGD LoRA on attention only) => skip the per-sample
        # mode-2 weight grads; mirrors Opaque_FusedMoE / _fused_moe_backward.
        compute_wgrad = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        dx, dW1, dW2, dtw = _GroupedMoEBackward.apply(
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
