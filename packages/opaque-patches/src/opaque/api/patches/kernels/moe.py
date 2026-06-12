# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Mixture-of-Experts expert-FFN kernel with vmap support for DP-SGD.

Signature mirrors Liger's ``LigerFusedMoEFunction.apply``. Layout matches HF v5
stacked-weight experts: ``x`` (..,H), ``gate_up_proj`` (E,2I,H), ``down_proj``
(E,H,I), ``top_k_index`` / ``top_k_weights`` (..,K) -> ``out`` (..,H).

Two ``autograd.Function``s (forward + backward), each with a ``vmap`` rule — the
Opaque pattern for ``vmap(grad)`` (DP-SGD per-sample gradients). The vmap rules
stay vectorized: per-token grads merge the vmap batch into the token dim;
per-sample expert-weight grads use a batched einsum over the vmap dim (kept, NOT
summed across samples). The expert loop is over the static expert count.

This dense (every-token-through-every-expert) formulation is the correctness
baseline; a fused Triton grouped-GEMM kernel can replace the internals behind the
same Function/vmap contract for the sparse-compute speedup. (A pure-torch sparse
gather is *slower* here — per-expert gather/scatter overhead dominates — so the
sparsity win specifically needs the fused kernel.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _route_weights(top_k_index, top_k_weights, num_experts):
    """(.., K) routing -> (.., E) dense per-expert weight (0 for unrouted)."""
    experts = torch.arange(num_experts, device=top_k_index.device)
    onehot = (top_k_index.unsqueeze(-1) == experts).to(top_k_weights.dtype)
    return (top_k_weights.unsqueeze(-1) * onehot).sum(dim=-2)


def _moe_forward(x, gate_up_proj, down_proj, w_te):
    """Dense-masked MoE forward; ``w_te`` (..,E). Token-independent (leading dims
    broadcast). fp32 expert-sum reduction."""
    intermediate = gate_up_proj.shape[1] // 2
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(gate_up_proj.shape[0]):
        gate_up = F.linear(x, gate_up_proj[e])
        h = F.silu(gate_up[..., :intermediate].float()).to(x.dtype) * gate_up[..., intermediate:]
        out = out + (F.linear(h, down_proj[e]) * w_te[..., e : e + 1]).float()
    return out.to(x.dtype)


def _moe_backward(grad_out, x, gate_up_proj, down_proj, w_te, batch_dims):
    """Manual MoE backward. ``batch_dims`` (0 or 1) = leading dims kept on the
    expert-weight grads: 0 -> summed (E,..) for the training graph; 1 -> per-sample
    (B,E,..) under vmap. Per-token grads (dx, dw_te) always keep their leading dims."""
    intermediate = gate_up_proj.shape[1] // 2
    dx = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    dw_te = torch.zeros(w_te.shape, dtype=torch.float32, device=x.device)
    dgate_up, ddown = [], []
    bspec = "b" if batch_dims == 1 else ""
    for e in range(gate_up_proj.shape[0]):
        gate_up = F.linear(x, gate_up_proj[e])
        gate, up = gate_up[..., :intermediate], gate_up[..., intermediate:]
        sig = torch.sigmoid(gate.float())
        silu = (gate.float() * sig).to(gate.dtype)
        h = silu * up
        y = F.linear(h, down_proj[e])

        dy = (w_te[..., e : e + 1] * grad_out).to(grad_out.dtype)
        dw_te[..., e] = (grad_out.float() * y.float()).sum(dim=-1)
        dh = F.linear(dy, down_proj[e].t())
        dsilu = (sig * (1.0 + gate.float() * (1.0 - sig))).to(gate.dtype)
        dgu = torch.cat([dh * up * dsilu, dh * silu], dim=-1)
        dx = dx + F.linear(dgu, gate_up_proj[e].t()).float()
        # weight grads: sum the token dim, keep the batch dim per ``batch_dims``.
        ddown.append(torch.einsum(f"{bspec}th,{bspec}ti->{bspec}hi", dy.float(), h.float()))
        dgate_up.append(torch.einsum(f"{bspec}tj,{bspec}th->{bspec}jh", dgu.float(), x.float()))

    dgate_up = torch.stack(dgate_up, dim=batch_dims).to(gate_up_proj.dtype)
    ddown = torch.stack(ddown, dim=batch_dims).to(down_proj.dtype)
    return dx.to(x.dtype), dgate_up, ddown, dw_te


def _dw_te_to_dtw(dw_te, top_k_index, dtype):
    """Map (.., E) routing-weight grad to (.., K) via the selected experts."""
    return torch.gather(dw_te, -1, top_k_index.long()).to(dtype)


class _MoEBackward(torch.autograd.Function):
    """Backward as an autograd.Function so ``vmap(grad)`` routes here. No double backward."""

    @staticmethod
    def forward(grad_out, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        w_te = _route_weights(top_k_index, top_k_weights, gate_up_proj.shape[0])
        dx, dgate_up, ddown, dw_te = _moe_backward(
            grad_out, x, gate_up_proj, down_proj, w_te, batch_dims=0
        )
        return dx, dgate_up, ddown, _dw_te_to_dtw(dw_te, top_k_index, top_k_weights.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for MoE")

    @staticmethod
    def vmap(info, in_dims, grad_out, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        w_te = _route_weights(top_k_index, top_k_weights, gate_up_proj.shape[0])
        dx, dgate_up, ddown, dw_te = _moe_backward(
            grad_out, x, gate_up_proj, down_proj, w_te, batch_dims=1
        )
        dtw = _dw_te_to_dtw(dw_te, top_k_index, top_k_weights.dtype)
        # dx/dtw batched at 0 (per-token); dgate_up/ddown batched at 0 (per-sample
        # grads of the shared weights — exactly what DP-SGD needs).
        return (dx, dgate_up, ddown, dtw), (0, 0, 0, 0)


class Opaque_MoE(torch.autograd.Function):
    """MoE expert FFN with vmap-grad support (DP-SGD). Same ``.apply`` signature as
    Liger's fused op, so a Triton kernel can replace the dense internals."""

    @staticmethod
    def forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        w_te = _route_weights(top_k_index, top_k_weights, gate_up_proj.shape[0])
        return _moe_forward(x, gate_up_proj, down_proj, w_te)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_out):
        dx, dgate_up, ddown, dtw = _MoEBackward.apply(grad_out, *ctx.saved_tensors)
        # inputs: x, gate_up_proj, down_proj, top_k_index (int, no grad), top_k_weights
        return dx, dgate_up, ddown, None, dtw

    @staticmethod
    def vmap(info, in_dims, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        # Forward is token-independent: merge the vmap batch into the token dim.
        w_te = _route_weights(top_k_index, top_k_weights, gate_up_proj.shape[0])
        return _moe_forward(x, gate_up_proj, down_proj, w_te), 0


def opaque_moe(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """MoE expert FFN. Autograd + ``vmap(grad)`` (DP-SGD) flow through the
    two-Function pair above.

    Always dispatches to the sparse grouped-GEMM Triton path
    (:func:`opaque_fused_moe`) when the tensors are CUDA bf16/fp16 with Triton
    present, and to the dense torch ``Opaque_MoE`` otherwise (CPU, fp32). The two
    are numerically equivalent — forward is bit-identical, backward matches within
    the bf16 floor (see ``test_kernel_precision``) — so the fused path runs
    whenever the hardware supports it; no opt-out knob.
    """
    if _TRITON_AVAILABLE and x.is_cuda:
        from ._utils import follow_autocast

        x, gate_up_proj, down_proj, top_k_weights = follow_autocast(
            x, gate_up_proj, down_proj, top_k_weights
        )
        if x.dtype in (torch.bfloat16, torch.float16):
            from .fused_moe import Opaque_FusedMoE

            return Opaque_FusedMoE.apply(
                x, gate_up_proj, down_proj, top_k_index, top_k_weights
            )
    return Opaque_MoE.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)


def torch_reference_moe(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Pure-PyTorch autograd-composed MoE reference (oracle for tests)."""
    w_te = _route_weights(top_k_index, top_k_weights, gate_up_proj.shape[0])
    return _moe_forward(x, gate_up_proj, down_proj, w_te)
