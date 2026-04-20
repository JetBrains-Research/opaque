"""Fused Linear + Cross-Entropy Loss with native vmap support.

Computes CE(hidden_states @ weight.T, labels) without materializing the full
logit matrix. Ported from Apple's cut_cross_entropy Triton kernels (ICLR 2025),
simplified to remove bias/Kahan/filtering/VocabOrdering/dLSE/shift and to
support our standard Opaque_Foo + _FooBackward vmap dispatch pattern.

Shift is handled in Python (pre-shift) so the kernel sees flat pre-shifted
tokens. This makes vmap merge a trivial reshape instead of requiring per-sample
loops.

Mathematical decomposition:
    CE(e, c, t) = -e·c[t] + log(Σ_v exp(e·c[v]))
                = neg_dot(e, c[t]) + LSE(e @ c.T)

Memory savings: O(B) + O(V) intermediate storage instead of O(B * V).
For LLaMA-3 (128K vocab), this avoids materializing ~1 GB of logits per sample.
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton import Config

from .utils import (
    _build_flat_valids,
    b_bin_fn,
    ensure_cuda_tensors,
    tl_lock_add,
    tl_logaddexp,
    tl_softcapping,
    tl_softcapping_grad,
)


# =============================================================================
# Autotune configs (subset of CCE defaults, sufficient for our use case)
# =============================================================================


def _get_autotune_configs():
    return [
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 128}, num_warps=4, num_stages=2
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 32}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=8, num_stages=3
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 256, "BLOCK_D": 32}, num_warps=8, num_stages=3
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=8, num_stages=3
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 128}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 64}, num_warps=4, num_stages=4
        ),
    ]


# Use fixed best config (same as CCE default) — set CCE_AUTOTUNE=1 to enable search
_AUTOTUNE = _os.getenv("CCE_AUTOTUNE", "0") != "0"


# =============================================================================
# Forward Triton Kernel
# =============================================================================


def _linear_ce_forward_kernel(
    E,
    C,
    LSE,
    NegCorrectLogit,
    Locks,
    Valids,
    Targets,
    softcap,
    B,
    V,
    D,
    BMax,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_vb,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_pid_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_pid_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(
            tl.int64
        )

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    # Tiled matmul: logits[b,v] = Σ_d E[b,d] * C[v,d]
    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))
        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision="ieee")
        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    accum = accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
    logits = tl.where(offs_v[None, :] < V, accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    logits = logits.cast(tl.float32)

    # Store neg_correct_logit = -logits[b, target[b]] (no shift — targets pre-shifted)
    this_targets = tl.load(Targets + offs_b, mask=offs_b < BMax, other=V + 1)
    direct_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    neg_correct_logit_ptrs = NegCorrectLogit + direct_offs_b
    neg_correct_logit_ptrs = tl.broadcast_to(
        neg_correct_logit_ptrs[:, None], (BLOCK_B, BLOCK_V)
    )
    tl.store(
        neg_correct_logit_ptrs, -logits, mask=this_targets[:, None] == offs_v[None, :]
    )

    # Per-block LSE: max + log(sum(exp(logits - max)))
    this_mx = tl.max(logits, axis=1)
    this_lse = this_mx + tl.log(tl.sum(tl.exp(logits - this_mx[:, None]), axis=1))

    o_mask = direct_offs_b < B
    lse_ptrs = LSE + direct_offs_b

    # Lock-based atomic logaddexp across V blocks
    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    lse = tl_logaddexp(lse, this_lse)
    tl.store(lse_ptrs, lse, mask=o_mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(this_locks, 0)


_linear_ce_forward_kernel = triton.jit(_linear_ce_forward_kernel)
_linear_ce_forward_kernel = triton.heuristics(
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "GROUP_B": lambda args: 8,
    }
)(_linear_ce_forward_kernel)

if _AUTOTUNE:
    _linear_ce_forward_kernel = triton.autotune(
        configs=_get_autotune_configs(),
        key=["V", "D", "B_BIN"],
        restore_value=["LSE"],
    )(_linear_ce_forward_kernel)
else:
    # Fixed best config (matches CCE default)
    _linear_ce_forward_kernel = triton.heuristics(
        {
            k: (lambda args, _v=v: _v)
            for k, v in Config(
                dict(BLOCK_B=128, BLOCK_V=128, BLOCK_D=32), num_warps=4, num_stages=4
            )
            .all_kwargs()
            .items()
        }
    )(_linear_ce_forward_kernel)


# =============================================================================
# Backward Triton Kernels
# =============================================================================


@triton.jit
def _mm_backward(
    do,
    da_ptrs,
    partial_mask_a,
    da_lock_ptr,
    n_locks,
    b_ptrs,
    partial_mask_b,
    stride_ad,
    stride_bd,
    D,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    d_inds = tl.arange(0, BLOCK_D)[None, :].to(tl.int64)

    b_ptrs = b_ptrs + d_inds * stride_bd
    da_ptrs = da_ptrs + d_inds * stride_ad

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        if EVEN_D:
            mask = partial_mask_b
        else:
            mask = partial_mask_b & (d_inds < (D - d * BLOCK_D))

        b = tl.load(b_ptrs, mask=mask, other=0.0)
        da_i = tl.dot(do, b, input_precision="ieee").to(da_ptrs.dtype.element_ty)

        if EVEN_D:
            mask = partial_mask_a
        else:
            mask = partial_mask_a & (d_inds < (D - d * BLOCK_D))

        lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
        this_da_lock_ptr = da_lock_ptr + lock_offset

        tl_lock_add(da_ptrs, da_i, mask, this_da_lock_ptr)

        b_ptrs += BLOCK_D * stride_bd
        da_ptrs += BLOCK_D * stride_ad


def _linear_ce_backward_kernel(
    E,
    C,
    LSE,
    dOut,
    Valids,
    softcap,
    Targets,
    dE,
    dELocks,
    dC,
    dCLocks,
    B,
    D,
    V,
    BMax,
    n_de_locks_0,
    n_de_locks_1,
    n_dc_locks_0,
    n_dc_locks_1,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_vb,
    # Per-sample dC parameters
    tokens_per_sample,
    num_dc_samples,
    dc_sample_stride,
    dc_locks_sample_stride,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MM_BACK_BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    ITEM_DO: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    COMPUTE_DC: tl.constexpr,
    COMPUTE_DE: tl.constexpr,
    PER_SAMPLE_DC: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_v_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_v_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(
            tl.int64
        )

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    # Recompute logits via tiled matmul E @ C^T
    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))
        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision="ieee")
        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    accum = accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
    if HAS_SOFTCAP:
        accum = tl_softcapping(accum, softcap)

    # Load LSE for CE gradient computation
    if HAS_VALIDS:
        direct_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
        lse = tl.load(LSE + direct_offs_b, mask=direct_offs_b < B, other=float("inf"))
    else:
        lse = tl.load(LSE + offs_b, mask=offs_b < B, other=float("inf"))

    accum = accum.cast(tl.float32)
    # d_accum = softmax(logits) - one_hot(target) = exp(logits - LSE) - one_hot
    d_accum = tl.exp(accum - lse[:, None])
    d_accum = tl.where(offs_v[None, :] < V, d_accum, 0.0)

    # Subtract one-hot at target positions
    targets = tl.load(Targets + offs_b, mask=offs_b < BMax, other=V + 1)
    is_target = targets[:, None] == offs_v[None, :]
    d_accum += tl.where(is_target, -1.0, 0.0)

    # Scale by upstream gradient
    if ITEM_DO:
        d_out = tl.load(dOut)
    else:
        d_out = tl.load(dOut + offs_b, mask=offs_b < BMax, other=0.0)[:, None]

    d_accum = d_accum * d_out

    # Softcap gradient
    if HAS_SOFTCAP:
        d_accum = tl_softcapping_grad(d_accum, accum, softcap)

    d_accum = d_accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")

    # dE = d_accum @ C via _mm_backward with locks
    if COMPUTE_DE:
        lock_offset = (pid_b // tl.cdiv(B, BLOCK_B * n_de_locks_0)) * n_de_locks_1
        _mm_backward(
            d_accum,
            dE + (offs_b[:, None] * stride_eb),
            offs_b[:, None] < BMax,
            dELocks + lock_offset,
            n_de_locks_1,
            C + offs_v[:, None] * stride_cv,
            offs_v[:, None] < V,
            stride_ed,
            stride_cd,
            D,
            MM_BACK_BLOCK_D,
            MM_BACK_EVEN_D,
        )

    # dC = d_accum^T @ E via _mm_backward with locks
    if COMPUTE_DC:
        if PER_SAMPLE_DC:
            # Per-sample dC: each token's contribution goes to its sample's dC buffer.
            # Compute sample_id from token position: sample_id = offs_b // tokens_per_sample
            sample_ids = offs_b // tokens_per_sample  # (BLOCK_B,)

            for s in range(num_dc_samples):
                s_mask = sample_ids == s  # (BLOCK_B,)
                count_s = tl.sum(s_mask.to(tl.int32))
                if count_s > 0:
                    masked_d = tl.where(
                        s_mask[:, None], d_accum, tl.zeros_like(d_accum)
                    )
                    dc_s_base = dC + s * dc_sample_stride
                    dc_locks_s_base = dCLocks + s * dc_locks_sample_stride

                    lock_offset = (
                        pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)
                    ) * n_dc_locks_1
                    _mm_backward(
                        tl.trans(masked_d),
                        dc_s_base + (offs_v[:, None] * stride_cv),
                        offs_v[:, None] < V,
                        dc_locks_s_base + lock_offset,
                        n_dc_locks_1,
                        E + (offs_b[:, None] * stride_eb),
                        offs_b[:, None] < BMax,
                        stride_cd,
                        stride_ed,
                        D,
                        MM_BACK_BLOCK_D,
                        MM_BACK_EVEN_D,
                    )
        else:
            lock_offset = (pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)) * n_dc_locks_1
            _mm_backward(
                tl.trans(d_accum),
                dC + (offs_v[:, None] * stride_cv),
                offs_v[:, None] < V,
                dCLocks + lock_offset,
                n_dc_locks_1,
                E + (offs_b[:, None] * stride_eb),
                offs_b[:, None] < BMax,
                stride_cd,
                stride_ed,
                D,
                MM_BACK_BLOCK_D,
                MM_BACK_EVEN_D,
            )


def _back_block_d(args) -> int:
    return 2 * args["BLOCK_D"]


_linear_ce_backward_kernel = triton.jit(_linear_ce_backward_kernel)
_linear_ce_backward_kernel = triton.heuristics(
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "MM_BACK_BLOCK_D": lambda args: _back_block_d(args),
        "MM_BACK_EVEN_D": lambda args: (args["D"] % _back_block_d(args)) == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "ITEM_DO": lambda args: args["dOut"].numel() == 1,
        "GROUP_B": lambda args: 8,
        "COMPUTE_DC": lambda args: args["dC"] is not None,
        "COMPUTE_DE": lambda args: args["dE"] is not None,
        "PER_SAMPLE_DC": lambda args: args["num_dc_samples"] > 1,
    }
)(_linear_ce_backward_kernel)

if _AUTOTUNE:
    _linear_ce_backward_kernel = triton.autotune(
        configs=_get_autotune_configs(),
        key=["V", "D", "B_BIN"],
        reset_to_zero=["dE", "dC"],
    )(_linear_ce_backward_kernel)
else:
    _linear_ce_backward_kernel = triton.heuristics(
        {
            k: (lambda args, _v=v: _v)
            for k, v in Config(
                dict(BLOCK_B=128, BLOCK_V=128, BLOCK_D=32), num_warps=4, num_stages=4
            )
            .all_kwargs()
            .items()
        }
    )(_linear_ce_backward_kernel)


# =============================================================================
# Python wrapper helpers
# =============================================================================


@dataclass(slots=True)
class LSEReturn:
    lse: torch.Tensor
    neg_correct_logit: torch.Tensor


def _forward_impl(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    valids: torch.Tensor | None,
    softcap: float | None,
) -> LSEReturn:
    """Launch forward kernel. e=(B,D) or (N,D), c=(V,D), targets=(N,)."""
    assert e.is_contiguous()
    assert e.shape[1] == c.shape[1]

    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B = e.shape[0]

    V, D = c.shape

    lse = e.new_full((B,), -torch.inf, dtype=torch.float32)
    neg_correct_logit = e.new_full((B,), 0.0, dtype=torch.float32)
    locks = e.new_full((triton.cdiv(B, 128),), 0, dtype=torch.uint32)

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _linear_ce_forward_kernel[grid](
        e,
        c,
        lse,
        neg_correct_logit,
        locks,
        valids,
        targets,
        softcap,
        B,
        V,
        D,
        e.size(0),  # BMax
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if valids is None else valids.stride(0),
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
    )

    return LSEReturn(lse, neg_correct_logit)


def _backward_impl(
    do: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    lse: torch.Tensor,
    targets: torch.Tensor,
    valids: torch.Tensor | None,
    softcap: float | None,
    compute_de: bool,
    compute_dc: bool,
    num_dc_samples: int = 1,
    tokens_per_sample: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Launch backward kernel.

    Args:
        num_dc_samples: When > 1, produces per-sample dC of shape (num_dc_samples, V, D).
            Tokens are split by sample_id = token_position // tokens_per_sample.
        tokens_per_sample: Number of tokens per sample (required when num_dc_samples > 1).

    Returns (de, dc) cast to input dtype.
    """
    assert e.is_contiguous()
    assert c.shape[1] == e.shape[1]

    if valids is not None:
        B = valids.size(0)
    else:
        B = e.size(0)

    V, D = c.shape
    nd_locks = triton.cdiv(D, 64)

    de = torch.zeros_like(e, dtype=torch.float32) if compute_de else None

    if compute_dc:
        if num_dc_samples > 1:
            # Per-sample dC: (num_dc_samples, V, D)
            dc = torch.zeros(num_dc_samples, V, D, dtype=torch.float32, device=e.device)
            dc_sample_stride = V * D  # stride between samples (contiguous)
        else:
            dc = torch.zeros_like(c, dtype=torch.float32)
            dc_sample_stride = 0
    else:
        dc = None
        dc_sample_stride = 0

    if de is not None:
        de_locks = e.new_zeros((triton.cdiv(B, 128), nd_locks), dtype=torch.int32)
        de_lock_sizes = de_locks.size()
    else:
        de_locks = None
        de_lock_sizes = (None, None)

    if dc is not None:
        if num_dc_samples > 1:
            dc_locks = e.new_zeros(
                (num_dc_samples, triton.cdiv(V, 128), nd_locks), dtype=torch.int32
            )
            dc_lock_sizes = (triton.cdiv(V, 128), nd_locks)
            dc_locks_sample_stride = triton.cdiv(V, 128) * nd_locks
        else:
            dc_locks = e.new_zeros((triton.cdiv(V, 128), nd_locks), dtype=torch.int32)
            dc_lock_sizes = dc_locks.size()
            dc_locks_sample_stride = 0
    else:
        dc_locks = None
        dc_lock_sizes = (None, None)
        dc_locks_sample_stride = 0

    do = do.contiguous()
    lse = lse.contiguous()

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _linear_ce_backward_kernel[grid](
        e,
        c,
        lse,
        do,
        valids,
        softcap,
        targets,
        de,
        de_locks,
        dc,
        dc_locks,
        B,
        D,
        V,
        e.size(0),  # BMax
        *de_lock_sizes,
        *dc_lock_sizes,
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if valids is None else valids.stride(0),
        tokens_per_sample=tokens_per_sample,
        num_dc_samples=num_dc_samples,
        dc_sample_stride=dc_sample_stride,
        dc_locks_sample_stride=dc_locks_sample_stride,
        B_BIN=b_bin_fn(B),
    )

    if de is not None:
        de = de.to(dtype=e.dtype)
    if dc is not None:
        dc = dc.to(dtype=c.dtype)

    return de, dc


# =============================================================================
# Backward autograd.Function (for vmap(grad()) dispatch)
# =============================================================================


class _LinearCEBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(
        grad_out,
        hidden_states,
        weight,
        labels,
        softcap,
        ignore_index,
        compute_dc,
    ):
        # Pre-shift and flatten
        e = hidden_states[..., :-1, :].contiguous().flatten(0, -2)  # (N, D)
        targets = labels[..., 1:].contiguous().flatten()  # (N,)
        valids = _build_flat_valids(targets, ignore_index)

        # Recompute LSE (activation checkpointing style)
        lse = _forward_impl(e, weight, targets, valids, softcap).lse

        de, dc = _backward_impl(
            grad_out,
            e,
            weight,
            lse,
            targets,
            valids,
            softcap,
            compute_de=True,
            compute_dc=compute_dc,
        )
        if dc is None:
            dc = weight.new_zeros(weight.shape)
        return de, dc

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError(
            "Double backward not supported for LinearCrossEntropyLoss"
        )

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_out,
        hidden_states,
        weight,
        labels,
        softcap,
        ignore_index,
        compute_dc,
    ):
        (grad_bdim, h_bdim, w_bdim, lab_bdim, sc_bdim, ii_bdim, dc_bdim) = in_dims

        assert w_bdim is None, "weight should not be batched"
        assert sc_bdim is None, "softcap should not be batched"
        assert ii_bdim is None, "ignore_index should not be batched"
        assert dc_bdim is None, "compute_dc should not be batched"

        B_vmap = hidden_states.shape[0]
        D = hidden_states.shape[-1]
        V = weight.shape[0]

        # Pre-shift and merge all samples into one flat batch
        h_shifted = hidden_states[..., :-1, :].contiguous()  # (B_vmap, ..., seq-1, D)
        t_shifted = labels[..., 1:].contiguous()  # (B_vmap, ..., seq-1)
        tokens_per_sample = h_shifted[0].numel() // D

        e = h_shifted.reshape(-1, D)  # (B_vmap * tokens_per_sample, D)
        targets = t_shifted.reshape(-1)  # (B_vmap * tokens_per_sample,)

        valids = _build_flat_valids(targets, ignore_index)

        # Expand per-sample scalar grad to per-token grad
        if grad_bdim is not None:
            do = grad_out.repeat_interleave(
                torch.tensor(tokens_per_sample, device=grad_out.device)
            )
        else:
            do = grad_out.expand(B_vmap * tokens_per_sample)

        # Single merged forward
        lse = _forward_impl(e, weight, targets, valids, softcap).lse

        # Single merged backward with per-sample dC (if needed):
        # de is merged (all samples), dc is per-sample via kernel-level sample masking.
        # This is 1 forward + 1 backward = 2 kernel launches instead of B_vmap × 2.
        de, dc = _backward_impl(
            do,
            e,
            weight,
            lse,
            targets,
            valids,
            softcap,
            compute_de=True,
            compute_dc=compute_dc,
            num_dc_samples=B_vmap if compute_dc else 1,
            tokens_per_sample=tokens_per_sample if compute_dc else 0,
        )

        # Reshape de from (B_vmap * tokens_per_sample, D) to (B_vmap, tokens_per_sample, D)
        # Don't pad here — backward() handles reshape + padding
        de = de.reshape(B_vmap, tokens_per_sample, D)

        if dc is None:
            dc = weight.new_zeros((B_vmap, V, D))

        # dc is already (B_vmap, V, D) from per-sample kernel (or zeros if skipped)
        return (de, dc), (0, 0)


# =============================================================================
# Main autograd.Function
# =============================================================================


class Opaque_LinearCrossEntropyLoss(torch.autograd.Function):
    """Fused linear projection + cross-entropy loss with vmap support.

    Computes the NLL sum:
        nll_sum = Σ_valid_tokens CE(hidden_states @ weight.T, labels)

    Without materializing the full (batch*seq, vocab) logit matrix.

    Returns unreduced nll_sum — caller handles reduction (mean, num_items_in_batch).
    """

    @staticmethod
    def forward(
        hidden_states,
        weight,
        labels,
        ignore_index=-100,
        logit_softcapping=0,
    ):
        softcap = logit_softcapping if logit_softcapping != 0 else None

        # Pre-shift and flatten
        e = hidden_states[..., :-1, :].contiguous().flatten(0, -2)  # (N, D)
        targets = labels[..., 1:].contiguous().flatten()  # (N,)
        valids = _build_flat_valids(targets, ignore_index)

        lse_ret = _forward_impl(e, weight, targets, valids, softcap)

        # NLL = neg_correct_logit + lse = -e·c[t] + log(Σ exp(e·c[v]))
        nll = lse_ret.neg_correct_logit.add_(lse_ret.lse)
        return nll.sum()

    @staticmethod
    def setup_context(ctx, inputs, output):
        (hidden_states, weight, labels, ignore_index, logit_softcapping) = inputs

        ctx.save_for_backward(hidden_states, weight, labels)
        ctx.softcap = logit_softcapping if logit_softcapping != 0 else None
        ctx.ignore_index = ignore_index

    @staticmethod
    def backward(ctx, grad_loss):
        hidden_states, weight, labels = ctx.saved_tensors
        # needs_input_grad[1] = weight needs grad. In DP-SGD LoRA training,
        # weight is frozen → skip dC to save ~1/3 of backward kernel time.
        compute_dc = ctx.needs_input_grad[1]

        de, dc = _LinearCEBackward.apply(
            grad_loss,
            hidden_states,
            weight,
            labels,
            ctx.softcap,
            ctx.ignore_index,
            compute_dc,
        )

        # de is (shifted_seq, D) — reshape and pad to match hidden_states shape
        shifted_shape = list(hidden_states.shape)
        shifted_shape[-2] -= 1
        de = de.reshape(shifted_shape)
        pad_shape = list(hidden_states.shape)
        pad_shape[-2] = 1
        de = torch.cat([de, de.new_zeros(pad_shape)], dim=-2)

        return de, dc, None, None, None

    @staticmethod
    def vmap(
        info, in_dims, hidden_states, weight, labels, ignore_index, logit_softcapping
    ):
        """Custom vmap rule for DP-SGD — single merged kernel call."""
        (h_bdim, w_bdim, lab_bdim, ii_bdim, sc_bdim) = in_dims

        if h_bdim != 0:
            raise ValueError(f"hidden_states should be batched at dim 0, got {h_bdim}")
        if lab_bdim != 0:
            raise ValueError(f"labels should be batched at dim 0, got {lab_bdim}")
        assert w_bdim is None, "weight should not be batched"
        assert ii_bdim is None, "ignore_index should not be batched"
        assert sc_bdim is None, "logit_softcapping should not be batched"

        softcap = logit_softcapping if logit_softcapping != 0 else None

        B_vmap = hidden_states.shape[0]
        D = hidden_states.shape[-1]

        # Pre-shift and merge all samples into one flat batch
        h_shifted = hidden_states[..., :-1, :].contiguous()
        t_shifted = labels[..., 1:].contiguous()
        tokens_per_sample = h_shifted[0].numel() // D

        e = h_shifted.reshape(-1, D)
        targets = t_shifted.reshape(-1)

        valids = _build_flat_valids(targets, ignore_index)

        # Single forward call for entire merged batch
        lse_ret = _forward_impl(e, weight, targets, valids, softcap)
        nll = lse_ret.neg_correct_logit.add_(lse_ret.lse)

        # Split per-sample: scatter NLLs back to sample buckets
        if valids is not None:
            sample_ids = valids.long() // tokens_per_sample
            nll_sums = torch.zeros(B_vmap, device=nll.device, dtype=nll.dtype)
            nll_sums.scatter_add_(0, sample_ids, nll)
        else:
            nll_sums = nll.reshape(B_vmap, tokens_per_sample).sum(dim=1)

        return nll_sums, 0


def opaque_linear_cross_entropy_loss(
    hidden_states,
    weight,
    labels,
    num_items_in_batch=None,
    ignore_index=-100,
    logit_softcapping=0,
):
    """Convenience wrapper for fused linear + cross-entropy loss.

    The kernel returns nll_sum (unreduced). This wrapper divides by
    num_items_in_batch (if given) or count of valid tokens.

    Any weight scaling (Granite divisive, Cohere multiplicative) should be
    applied to the weight tensor before calling this function, so that
    autograd correctly propagates gradients to the original weight.

    Args:
        hidden_states: (..., hidden_dim) embeddings from backbone
        weight: (vocab_size, hidden_dim) lm_head weight (pre-scaled if needed)
        labels: (...,) target token IDs (-100 = ignore)
        num_items_in_batch: optional denominator for loss averaging
        ignore_index: label value to ignore
        logit_softcapping: Gemma2 softcap value (0 = disabled)

    Returns:
        loss: scalar tensor
    """
    ensure_cuda_tensors(
        hidden_states,
        weight,
        labels,
        fn_name="opaque_linear_cross_entropy_loss",
    )
    nll_sum = Opaque_LinearCrossEntropyLoss.apply(
        hidden_states,
        weight,
        labels,
        ignore_index,
        logit_softcapping,
    )

    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(nll_sum.device)
        return nll_sum / num_items_in_batch

    shifted_labels = labels[..., 1:].contiguous().flatten()
    n_valid = (shifted_labels != ignore_index).sum().float().clamp(min=1)
    return nll_sum / n_valid
